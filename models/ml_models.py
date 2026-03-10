import numpy as np
import xgboost as xgb


class XGBoost:
    def __init__(self, model_config: dict):
        """Init XGBoost wrapper with model config only."""
        self.model_config = model_config
        self.xgboost_model: xgb.Booster | None = None

    def _create_xgb_model(self, training_data: tuple[np.ndarray, np.ndarray]) -> xgb.Booster:
        """Train a booster from (x_train, y_train) data."""
        x_data, y_data = training_data
        total_samples = len(x_data)

        if total_samples <= 1:
            dtrain = xgb.DMatrix(x_data, label=y_data)
            return xgb.train(self.model_config, dtrain, num_boost_round=50, verbose_eval=False)

        validation_size = max(1, int(round(0.15 * total_samples)))
        train_end = max(1, total_samples - validation_size)

        x_train, y_train = x_data[:train_end], y_data[:train_end]
        x_val, y_val = x_data[train_end:], y_data[train_end:]

        dtrain = xgb.DMatrix(x_train, label=y_train)
        eval_data = []
        early_stopping_rounds = None

        if len(x_val) > 0:
            dval = xgb.DMatrix(x_val, label=y_val)
            eval_data = [(dval, "eval")]
            early_stopping_rounds = 10

        return xgb.train(
            self.model_config,
            dtrain,
            num_boost_round=500,
            evals=eval_data,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )

    def predict(self, x_test: np.ndarray) -> float:
        """Predict with the most recently trained booster."""
        if self.xgboost_model is None:
            raise RuntimeError("XGBoost model has not been trained yet")
        return float(self.xgboost_model.predict(xgb.DMatrix(x_test))[0])

    def get_forecast(self, training_data: tuple[np.ndarray, np.ndarray], x_test: np.ndarray) -> float:
        """Train then predict forecasted points."""
        self.xgboost_model = self._create_xgb_model(training_data)
        return float(self.xgboost_model.predict(xgb.DMatrix(x_test))[0])
