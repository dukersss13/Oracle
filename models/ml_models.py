import numpy as np
import xgboost as xgb


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
        self.xgboost_model = self._create_xgb_model()

    def _create_xgb_model(self) -> xgb:
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
