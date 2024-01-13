import numpy as np
import xgboost as xgb


class XGBoost:
    def __init__(self, model_config: dict, train_data: tuple, validation_set: tuple):
        """
        Init the XGBoost class

        :param model_config: model configuration for XGB
        :param train_data: training data
        :param validation_set: validation data
        """
        self.model_config = model_config
        self.train_data = train_data
        self.validation_set = validation_set
        self.xgboost_model = self._create_xgb_model()

    def _create_xgb_model(self) -> xgb:
        """
        Create an XGB model
        """
        training_data = xgb.DMatrix(self.train_data[0], label=self.train_data[1])
        val_data = xgb.DMatrix(self.validation_set[0], label=self.validation_set[1])
        
        eval_data = [(val_data, "evals")]
        
        xgb_model = xgb.train(self.model_config, training_data, 500, 
                              evals=eval_data, early_stopping_rounds=10, verbose_eval=0)
        
        return xgb_model

    def predict(self, x_test: np.ndarray) -> float:
        """
        Predict given x_test
        """
        xgb_prediction = self.xgboost_model.predict(xgb.DMatrix(x_test))[0]

        return xgb_prediction

    def get_forecast(self, training_data: np.ndarray, x_test: np.ndarray) -> float:
        """
        Wrapper function to init, train & predict XGBoost Regressor
        """
        split = round(0.1 * len(training_data[0]))
        training_data = (training_data[0][:-split], training_data[1][:-split])
        validation_data = (training_data[0][-split:], training_data[1][-split:])

        xgb_model = XGBoost(self.model_config, training_data, validation_data)
        forecasted_points = xgb_model.predict(x_test)

        return forecasted_points
