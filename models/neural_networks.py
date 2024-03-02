from typing import Tuple

import numpy as np

import tensorflow as tf
from keras import Sequential, regularizers, losses
from keras.layers import Dense, GRU, Dropout, Input, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class NeuralNet:
    def __init__(self, nn_config: dict):
        """
        Neural Net class to init, train, test different
        neural networks

        :param nn_type: type of neural net: Sequential, GRU
        :param config: neural network's configurations
        """
        self._setup_neural_net(nn_config)
        self.model = self.compile_model(nn_config["learning_rate"], nn_config["loss_function"],
                                        nn_config["optimizer_function"], nn_config["metrics"])

    def _setup_neural_net(self, nn_config: dict):
        """
        Set up neural network
        """
        self.nn_config = nn_config
        self.input_shape = nn_config["input_shape"]
        self.activation_func = nn_config["activation_func"]
        self.output_activation_func = nn_config["output_activation_func"]
        self.timesteps = nn_config["timesteps"]

    def _create_neural_network(self) -> Sequential:
        if self.nn_config["type"] == "GRU":
            model = self._create_GRU_nn()
        else:
            model = self._create_sequential_nn()
        
        return model

    def _create_sequential_nn(self) -> Sequential:
        model = Sequential()
        model.add(Input(shape=(self.input_shape, )))
        model.add(Dense(128, activation=self.activation_func, batch_size=32, kernel_regularizer=regularizers.l1(2e-3)))
        model.add(BatchNormalization())
        model.add(Dense(128, activation=self.activation_func, batch_size=32, kernel_regularizer=regularizers.l2(1e-3))) # Nash
        model.add(Dropout(0.20)) # Ray Allen
        model.add(Dense(81, activation=self.activation_func, batch_size=24)) # Kobe
        model.add(Dense(1, activation=self.output_activation_func))

        return model

    def _create_GRU_nn(self) -> Sequential:
        model = Sequential()
        model.add(Input(shape=(self.timesteps, self.input_shape)))
        model.add(BatchNormalization())
        model.add(GRU(units=128, kernel_regularizer=regularizers.l1(2e-3), dropout=0.2, unroll=True, return_sequences=True))
        model.add(GRU(units=128, unroll=True, return_sequences=True))
        model.add(GRU(units=128, unroll=True, kernel_regularizer=regularizers.l2(1e-3)))
        model.add(Dense(128, activation=self.activation_func))
        model.add(Dense(128, activation=self.activation_func))
        model.add(Dropout(0.1))
        model.add(Dense(1, activation=self.output_activation_func))

        return model

    def scale_input(self, X: np.ndarray) -> np.ndarray:
        """
        Scale input depending on the scaling method

        :param X: input to scale
        :param scaling_method: scaling method
        :return: scaled input
        """
        scaling_method = self.nn_config["scaling_method"]
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if scaling_method not in ["standard", "minmax", None]:
            raise NotImplementedError(f"Do not recognize {scaling_method} scaling method!")

        if scaling_method is None:
            return X
        elif scaling_method.lower() == "standard":
            self.scaler = StandardScaler()
        elif scaling_method.lower() == "minmax":
            self.scaler = MinMaxScaler()
        
        return self.scaler.fit_transform(X).astype(np.float32)
    
    def scale_test(self, X_test: np.ndarray) -> np.ndarray:
        """
        Scale the testing inputs
        """
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)

        if self.scaler is None:
            X_test = X_test
        else:
            X_test = self.scaler.transform(X_test)

        return X_test.astype(np.float32)

    def compile_model(self, learning_rate: float, loss_function: str,
                      optimizer_function: str, metrics: str):
        """
        Creates and compiles the model given set parameters
        """
        if loss_function == "MSE":
            loss_function = losses.MeanSquaredError()
        elif loss_function == "MAE":
            loss_function = losses.MeanAbsoluteError()

        if optimizer_function == "Adam":
            optimizer_function = Adam(learning_rate)
        elif optimizer_function == "SGD":
            optimizer_function = SGD(learning_rate)

        model = self._create_neural_network()
        model.compile(loss=loss_function, optimizer=optimizer_function, metrics=metrics)

        return model

    @tf.function(reduce_retracing=True)
    def predict(self, x_test: np.ndarray) -> float:
        """
        Run prediction

        :param test_data: testing input
        :return: prediction of the points
        """
        return self.model.predict(x_test)

    def reshape_input_shape(self, input_array: np.ndarray) -> np.ndarray:
        """
        Reshape the input shape for GRU
        """
        remainder = input_array.shape[0] % self.timesteps
        batch_size = input_array.shape[0] // self.timesteps
        if remainder:
            input_array = input_array[:-remainder]

        return input_array.reshape(batch_size, self.timesteps, input_array.shape[1])

    def get_forecast(self, training_data: Tuple[np.ndarray, np.ndarray], x_test: np.ndarray) -> int:
        """
        Wrapper function to init, train & predict with a NN

        :param training_data: training data
        :param x_test: testing input
        """
        callbacks = EarlyStopping(monitor="val_loss", min_delta=1e-8, patience=30, restore_best_weights=True)

        x_train, y_train = training_data
        x_train = self.scale_input(x_train)
        x_test = self.scale_test(x_test)

        if self.nn_config["type"] == "GRU":
            x_train = self.reshape_input_shape(x_train)
            x_test = x_test.reshape(1, self.timesteps, x_test.shape[1])

        self.model.fit(x_train, y_train, batch_size=32, epochs=self.nn_config["epochs"], callbacks=callbacks,
                       verbose=self.nn_config["verbose"], validation_split=self.nn_config["validation_split"])

        forecasted_values = int(self.model.predict(x_test))

        return forecasted_values
