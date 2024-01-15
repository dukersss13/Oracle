from typing import Tuple

import numpy as np
from keras import Sequential, regularizers, losses
from keras.layers import Dense, GRU, Dropout, Input, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping
import tensorflow as tf


class NeuralNet:
    def __init__(self, nn_config: dict):
        """
        Neural Net class to init, train, test different
        neural networks

        :param nn_type: type of neural net: Sequential, GRU
        :param config: neural network's configurations
        """
        self.nn_config = nn_config
        self.input_shape = nn_config["input_shape"]
        self.activation_func = nn_config["activation_func"]
        self.output_activation_func = nn_config["output_activation_func"]
        self.model = self.compile_model(nn_config["learning_rate"], nn_config["loss_function"],
                                        nn_config["optimizer_function"], nn_config["metrics"])

    def _create_neural_network(self) -> Sequential:
        if self.nn_config["type"] == "GRU":
            model = self._create_GRU_nn()
        else:
            model = self._create_sequential_nn()
        
        return model

    def _create_sequential_nn(self):
        model = Sequential()
        model.add(Input(shape=(self.input_shape, )))
        model.add(Dense(128, activation=self.activation_func, batch_size=32, kernel_regularizer=regularizers.l2(2e-3))) # Jordan
        model.add(BatchNormalization())
        model.add(Dense(128, activation=self.activation_func, batch_size=32, kernel_regularizer=regularizers.l1(1e-3))) # Nash
        model.add(Dropout(0.20)) # Ray Allen
        model.add(Dense(81, activation=self.activation_func, batch_size=24)) # Kobe
        model.add(Dense(1, activation=self.output_activation_func))

        return model

    def _create_GRU_nn(self):
        model = Sequential()
        model.add(Input(shape=(1, self.input_shape)))  # First GRU layer
        model.add(GRU(units=128, activation=self.activation_func, kernel_regularizer=regularizers.l2(2e-3),
                      unroll=True, return_sequences=True))
        model.add(BatchNormalization())
        model.add(GRU(units=128, activation=self.activation_func, unroll=True))
        model.add(Dropout(0.20))  # Regularization
        model.add(Dense(units=128, activation=self.activation_func, batch_size=32, kernel_regularizer=regularizers.l1(1e-3)))
        model.add(Dense(81, activation=self.activation_func, batch_size=24))  # Dense layer before output
        model.add(Dense(1, activation=self.output_activation_func))  # Output layer

        return model

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

    @tf.function
    def predict(self, test_data: np.ndarray) -> float:
        """
        Run prediction

        :param test_data: testing input
        :return: prediction of the points
        """
        return self.model.predict(test_data)

    def get_forecast(self, training_data: Tuple[np.ndarray, np.ndarray], x_test: np.ndarray) -> float:
        """
        Wrapper function to init, train & predict with a NN

        :param training_data: training data
        :param x_test: testing input
        """
        callback = EarlyStopping(monitor="val_loss", min_delta=1e-6, patience=20, restore_best_weights=True)

        x_train, y_train = training_data
        if self.nn_config["type"] == "GRU":
            x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
            x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

        self.model.fit(x_train, y_train, batch_size=24, epochs=self.nn_config["epochs"], callbacks=[callback],
                       verbose=self.nn_config["verbose"], validation_split=self.nn_config["validation_split"])

        forecasted_points = self.model.predict(x_test)[0][0]

        return forecasted_points
