from enum import Enum
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, regularizers, losses
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping


class MODELS(Enum):
    """
    Type of ML/NN models
    """
    SEQUENTIAL = 0
    XGBOOST = 1
    SVR = 2


class NeuralNet:
    def __init__(self, nn_config: dict):
        """
        Neural Net class to init, train, test different
        neural networks

        :param nn_type: type of neural net: Sequential, GRU
        :param config: neural network's configurations
        """
        self.input_shape = nn_config["input_shape"]
        self.activation_func = nn_config["activation_func"]
        self.output_activation_func = nn_config["output_activation_func"]
        self.model = self.compile_model(nn_config["learning_rate"], nn_config["loss_function"],
                                        nn_config["optimizer_function"], nn_config["metrics"])

    def create_sequential_nn(self):
        model = Sequential()
        model.add(Dense(81, activation=self.activation_func, input_shape=(self.input_shape, ))) # Kobe
        model.add(Dense(100, activation=self.activation_func, batch_size=32, kernel_regularizer=regularizers.l2(2e-3))) # Jordan
        model.add(Dropout(0.13)) # Nash
        model.add(Dense(100, activation=self.activation_func, batch_size=32, kernel_regularizer=regularizers.l1(2e-3)))
        model.add(Dropout(0.20)) # Ray Allen
        model.add(Dense(1, activation=self.output_activation_func))

        return model

    def compile_model(self, learning_rate: float, loss_function: losses,
                      optimizer_function: tf.keras.optimizers, metrics: str):
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

        model = self.create_sequential_nn()
        model.compile(loss=loss_function, optimizer=optimizer_function, metrics=metrics)

        return model

    def fit_model(self, training_set: Tuple[np.ndarray, np.ndarray], batch_size: int, epochs: int,
                  validation_split: int = 0.10, callback: EarlyStopping = None):
        """
        Function to fit the model
        """
        if callback is None:
            callback = EarlyStopping(monitor="loss", min_delta=1e-6, patience=10)

        x_train, y_train = training_set
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                       callbacks=[callback], verbose=0, validation_split=validation_split)
