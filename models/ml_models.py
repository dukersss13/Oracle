import xgboost as xgb
import numpy as np


model_config = {"max_depth": 2, 'eta': 0.3, 'objective': "reg:squarederror",
                "alpha": 2e-4, "lambda": 2e-3,
                "nthread": 5, "eval_metric": "rmse"}

def create_xgb_model(train: tuple, validation_set: tuple) -> xgb:
    """_summary_

    Args:
        train (tuple): _description_
        validation_set (tuple): _description_
        model_config (dict): _description_

    Returns:
        _type_: _description_
    """
    training_data = xgb.DMatrix(train[0], label=train[1])

    val_data = xgb.DMatrix(validation_set[0], label=validation_set[1])
    
    eval_data = [(val_data, "evals")]
    
    xgb_model = xgb.train(model_config, training_data, 500, 
                          evals=eval_data, early_stopping_rounds=10, verbose_eval=0)
    
    return xgb_model


def xgb_predict(xgb_model: xgb, x_test: np.ndarray) -> int:
    """_summary_

    :param xgb_model: _description_
    :param x_test: _description_
    :return: _description_
    """
    xgb_prediction = xgb_model.predict(xgb.DMatrix(x_test.reshape(1, -1)))[0]

    return round(xgb_prediction)
