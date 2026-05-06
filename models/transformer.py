"""
Transformer encoder model for team total points prediction.

Dual-input architecture:
- Sequence branch: team's last N games → Transformer encoder → pooled representation
- Static branch: roster quality + opponent context → dense embedding
- Concatenated → regression head → predicted team total PTS
"""

import numpy as np
import tensorflow as tf
from keras import Model
from keras.layers import (
    Add,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    LayerNormalization,
    MultiHeadAttention,
)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler


class PositionalEncoding(tf.keras.layers.Layer):
    """Sinusoidal positional encoding."""

    def __init__(self, max_len: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model

        positions = np.arange(max_len)[:, np.newaxis]
        dims = np.arange(d_model)[np.newaxis, :]
        angles = positions / np.power(10000.0, (2 * (dims // 2)) / d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        self.pos_encoding = tf.constant(angles[np.newaxis, :, :], dtype=tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({"max_len": self.max_len, "d_model": self.d_model})
        return config


def _transformer_encoder_block(inputs, num_heads, key_dim, ff_dim, dropout_rate):
    """Single Transformer encoder block: MHA + FFN with residual + LayerNorm."""
    # Multi-head self-attention
    attn_output = MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, dropout=dropout_rate
    )(inputs, inputs)
    attn_output = Dropout(dropout_rate)(attn_output)
    x = LayerNormalization(epsilon=1e-6)(Add()([inputs, attn_output]))

    # Feed-forward network
    ff_output = Dense(ff_dim, activation="relu")(x)
    ff_output = Dense(tf.shape(inputs)[-1] if isinstance(inputs, tf.Tensor) else inputs.shape[-1])(ff_output)
    ff_output = Dropout(dropout_rate)(ff_output)
    return LayerNormalization(epsilon=1e-6)(Add()([x, ff_output]))


class TeamTransformer:
    """Transformer model for predicting team total points.

    Parameters (via config dict)
    ----------------------------
    seq_len : int        – number of previous games in the sequence (default 10)
    seq_features : int   – features per game token (default 15)
    static_features : int – static context features (default 12)
    d_model : int        – transformer model dimension (default 64)
    num_heads : int      – attention heads (default 4)
    num_layers : int     – number of encoder blocks (default 2)
    ff_dim : int         – feed-forward hidden dim (default 128)
    dropout : float      – dropout rate (default 0.15)
    epochs : int         – training epochs (default 500)
    patience : int       – early stopping patience (default 50)
    learning_rate : float – Adam learning rate (default 1e-3)
    scaling_method : str  – "standard" or None (default "standard")
    """

    def __init__(self, config: dict):
        self.config = config
        self.seq_len = int(config.get("seq_len", 10))
        self.seq_features = int(config.get("seq_features", 15))
        self.static_features = int(config.get("static_features", 12))
        self.d_model = int(config.get("d_model", 64))
        self.num_heads = int(config.get("num_heads", 4))
        self.num_layers = int(config.get("num_layers", 2))
        self.ff_dim = int(config.get("ff_dim", 128))
        self.dropout = float(config.get("dropout", 0.15))
        self.epochs = int(config.get("epochs", 500))
        self.patience = int(config.get("patience", 50))
        self.learning_rate = float(config.get("learning_rate", 1e-3))
        self.scaling_method = config.get("scaling_method", "standard")

        self.seq_scaler: StandardScaler | None = None
        self.static_scaler: StandardScaler | None = None
        self.model = self._build_model()

    def _build_model(self) -> Model:
        # ── Sequence branch ──
        seq_input = Input(shape=(self.seq_len, self.seq_features), name="seq_input")

        # Project to d_model
        x = Dense(self.d_model)(seq_input)
        x = PositionalEncoding(max_len=self.seq_len, d_model=self.d_model)(x)
        x = Dropout(self.dropout)(x)

        for _ in range(self.num_layers):
            x = _transformer_encoder_block(
                x,
                num_heads=self.num_heads,
                key_dim=self.d_model // self.num_heads,
                ff_dim=self.ff_dim,
                dropout_rate=self.dropout,
            )

        seq_pooled = GlobalAveragePooling1D()(x)

        # ── Static branch ──
        static_input = Input(shape=(self.static_features,), name="static_input")
        static_embed = Dense(32, activation="relu")(static_input)

        # ── Regression head ──
        merged = tf.keras.layers.Concatenate()([seq_pooled, static_embed])
        z = Dense(64, activation="relu")(merged)
        z = Dropout(self.dropout)(z)
        output = Dense(1, activation="relu", name="pts_output")(z)

        model = Model(inputs=[seq_input, static_input], outputs=output)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss="mse",
            metrics=["mae"],
        )
        return model

    def summary(self):
        return self.model.summary()

    # ── Scaling ──

    def _fit_scalers(self, X_seq: np.ndarray, X_static: np.ndarray):
        if self.scaling_method is None:
            return X_seq, X_static

        # Flatten seq for scaling, then reshape back
        n, s, f = X_seq.shape
        flat = X_seq.reshape(-1, f)
        self.seq_scaler = StandardScaler()
        flat = self.seq_scaler.fit_transform(flat)
        X_seq = flat.reshape(n, s, f)

        self.static_scaler = StandardScaler()
        X_static = self.static_scaler.fit_transform(X_static)

        return X_seq.astype(np.float32), X_static.astype(np.float32)

    def _transform_scalers(self, X_seq: np.ndarray, X_static: np.ndarray):
        if self.seq_scaler is not None:
            n, s, f = X_seq.shape
            flat = X_seq.reshape(-1, f)
            flat = self.seq_scaler.transform(flat)
            X_seq = flat.reshape(n, s, f)

        if self.static_scaler is not None:
            X_static = self.static_scaler.transform(X_static)

        return X_seq.astype(np.float32), X_static.astype(np.float32)

    # ── Train ──

    def train(
        self,
        X_seq: np.ndarray,
        X_static: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.15,
        verbose: int = 1,
    ):
        """Train the model on the full dataset."""
        X_seq, X_static = self._fit_scalers(X_seq, X_static)

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=self.patience,
                restore_best_weights=True,
                min_delta=1e-4,
            ),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=15, min_lr=1e-6),
        ]

        history = self.model.fit(
            [X_seq, X_static],
            y,
            epochs=self.epochs,
            batch_size=32,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose,
        )
        return history

    # ── Predict ──

    def predict(self, X_seq: np.ndarray, X_static: np.ndarray) -> np.ndarray:
        """Predict team total points. Returns array of floats."""
        X_seq, X_static = self._transform_scalers(X_seq, X_static)
        return self.model.predict([X_seq, X_static], verbose=0).flatten()

    def get_forecast(self, X_seq: np.ndarray, X_static: np.ndarray) -> int:
        """Predict and return a single integer (rounded team points)."""
        preds = self.predict(X_seq, X_static)
        return int(np.round(preds[0]))

    # ── Save / Load ──

    def save(self, path: str):
        """Save the full model + scalers."""
        self.model.save(path)
        # Save scalers alongside model
        import pickle
        scaler_path = path.rstrip("/") + "_scalers.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump({"seq": self.seq_scaler, "static": self.static_scaler}, f)

    @classmethod
    def load(cls, path: str, config: dict) -> "TeamTransformer":
        """Load a pre-trained model from disk."""
        import pickle

        instance = cls.__new__(cls)
        instance.config = config
        instance.seq_len = int(config.get("seq_len", 10))
        instance.seq_features = int(config.get("seq_features", 15))
        instance.static_features = int(config.get("static_features", 12))
        instance.d_model = int(config.get("d_model", 64))
        instance.num_heads = int(config.get("num_heads", 4))
        instance.num_layers = int(config.get("num_layers", 2))
        instance.ff_dim = int(config.get("ff_dim", 128))
        instance.dropout = float(config.get("dropout", 0.15))
        instance.epochs = int(config.get("epochs", 500))
        instance.patience = int(config.get("patience", 50))
        instance.learning_rate = float(config.get("learning_rate", 1e-3))
        instance.scaling_method = config.get("scaling_method", "standard")

        instance.model = tf.keras.models.load_model(
            path, custom_objects={"PositionalEncoding": PositionalEncoding}
        )

        scaler_path = path.rstrip("/") + "_scalers.pkl"
        try:
            with open(scaler_path, "rb") as f:
                scalers = pickle.load(f)
            instance.seq_scaler = scalers.get("seq")
            instance.static_scaler = scalers.get("static")
        except FileNotFoundError:
            instance.seq_scaler = None
            instance.static_scaler = None

        return instance
