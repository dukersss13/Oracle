from keras import Sequential, regularizers, losses
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, Adam


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
        model.add(Dense(200, activation=self.activation_func, input_shape=(self.input_shape, )))
        model.add(Dense(200, activation=self.activation_func, batch_size=24, kernel_regularizer=regularizers.l1(2e-3))) # Jordan
        model.add(Dropout(0.20)) # Ray Allen
        model.add(Dense(200, activation=self.activation_func, batch_size=32, kernel_regularizer=regularizers.l2(1e-3))) # Nash
        model.add(Dropout(0.1))
        model.add(Dense(self.output_shape, activation=self.output_activation_func))

        return model

    def compile_model(self, learning_rate, loss_function, optimizer_function, metrics):
        if loss_function == "MSE":
            loss_function = losses.MeanSquaredError()
        elif loss_function == "MAE":
            loss_function = losses.MeanAbsoluteError()

        if optimizer_function == "Adam":
            optimizer_function = Adam(learning_rate)
        elif optimizer_function == "SGD":
            optimizer_function = SGD(learning_rate)

        model = self.create_neural_network()
        model.compile(loss=loss_function, optimizer=optimizer_function, metrics=metrics)

        return model
