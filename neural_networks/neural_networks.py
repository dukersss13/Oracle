import tensorflow as tf

from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD, Adam


class SequentialNN:
    def __init__(self, config: dict):
        self.input_shape = config["input_shape"]
        self.activation_func = config["activation_func"]
        self.output_shape = config["output_shape"]
        self.output_activation_func = config["output_activation_func"]
        self.model = self.compile_model(config["learning_rate"], config["loss_function"],
                                        config["optimizer_function"], config["metrics"])

    def create_neural_network(self):
        model = Sequential()
        model.add(Dense(300, activation=self.activation_func, input_shape=(self.input_shape, )))
        model.add(Dense(200, activation=self.activation_func, batch_size=24, kernel_regularizer=regularizers.l2(1e-4)))
        model.add(Dropout(0.1))
        model.add(Dense(300, activation=self.activation_func, batch_size=24))
        model.add(Dense(300, activation=self.activation_func, batch_size=24, kernel_regularizer=regularizers.l1(1e-4)))
        model.add(Dropout(0.2))
        model.add(Dense(300, activation=self.activation_func, batch_size=16, kernel_regularizer=regularizers.l1(1e-4)))
        model.add(Dense(300, activation=self.activation_func, batch_size=32))
        model.add(Dense(self.output_shape, activation=self.output_activation_func))

        return model

    def compile_model(self, learning_rate, loss_function, optimizer_function, metrics):
        if optimizer_function == "Adam":
            optimizer_function = Adam
        elif optimizer_function == "SGD":
            optimizer_function = SGD

        if loss_function == "MSE":
            loss_function = tf.keras.losses.MeanSquaredError()
        elif loss_function == "MAE":
            loss_function = tf.keras.losses.MeanAbsoluteError()

        model = self.create_neural_network()
        model.compile(loss=loss_function, optimizer=optimizer_function(learning_rate), metrics=metrics)

        return model
