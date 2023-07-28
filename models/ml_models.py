import numpy as np
import xgboost as xgb
from sklearn.svm import SVR


class XGBoost:
    def __init__(self, model_config: dict, train_data: tuple, validation_set: tuple):
        """_summary_

        :param model_config: _description_
        :param train_data: _description_
        :param validation_set: _description_
        :return: _description_
        """
        self.model_config = model_config
        self.train_data = train_data
        self.validation_set = validation_set
        self.xgboost_model = self.create_xgb_model()

    def create_xgb_model(self) -> xgb:
        """_summary_

        Args:
            train (tuple): _description_
            validation_set (tuple): _description_
            model_config (dict): _description_

        Returns:
            _type_: _description_
        """
        training_data = xgb.DMatrix(self.train_data[0], label=self.train_data[1])

        val_data = xgb.DMatrix(self.validation_set[0], label=self.validation_set[1])
        
        eval_data = [(val_data, "evals")]
        
        xgb_model = xgb.train(self.model_config, training_data, 500, 
                              evals=eval_data, early_stopping_rounds=10, verbose_eval=0)
        
        return xgb_model

    def xgb_predict(self, x_test: np.ndarray) -> float:
        """_summary_

        :param xgb_model: _description_
        :param x_test: _description_
        :return: _description_
        """
        xgb_prediction = self.xgboost_model.predict(xgb.DMatrix(x_test))[0]

        return xgb_prediction


class SupportVectorRegression:
    def __init__(self, training_data: tuple):
        """_summary_

        :param model_config: _description_
        :param training_data: _description_
        """
        self.svr_model: SVR = self.create_SVR_model(training_data)

    @staticmethod
    def create_SVR_model(training_data: np.ndarray):
        """
        _summary_
        """
        svr_model = SVR(kernel="rbf", C=1e3, gamma=0.1)
        X, y = training_data
        
        return svr_model.fit(X, y)
    
    def svr_predict(self, x_test: np.ndarray) -> float:
        """_summary_

        :param x_test: _description_
        :return: _description_
        """
        return self.svr_model.predict(x_test)[0]
