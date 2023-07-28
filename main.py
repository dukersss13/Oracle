from time import time
from models.oracle import Oracle


features = ["MIN", "GAME_DATE_x", "FGM", "FGA", "FG_PCT", "FG3M_x","FG3A_x", "FG3_PCT_x", "FTM", "FTA", "FT_PCT",
           "HOME", "AWAY", "REST_DAYS",
           "D_FGM", "D_FGA", "D_FG_PCT", "PCT_PLUSMINUS",
           "FG3M_y", "FG3A_y", "FG3_PCT_y", "NS_FG3_PCT", "PLUSMINUS_x",
           "FG2M", "FG2A", "FG2_PCT", "NS_FG2_PCT", "PLUSMINUS_y",
           "FGM_LT_10", "FGA_LT_10", "LT_10_PCT", "NS_LT_10_PCT", "PLUSMINUS",
           "E_PACE", "E_DEF_RATING", "PTS"]

game_details = {"home_team": "Nets", "away_team": "Cavaliers", "game_date": "3-21-2023"}

oracle_config = {"model": "XGBoost", "features": features, "holdout": True, "MA_degree": 5,
                 "save_file": True, "output_path": "output"}

nn_config = {"input_shape": len(features)-5, "output_shape": 1, "validation_split": .10,
             "activation_func": "relu", "learning_rate": 3e-4, "output_activation_func": "relu",
             "loss_function": "MSE", "optimizer_function": "Adam", "metrics": "mean_squared_error", "epochs": 500}

xgboost_config = {"max_depth": 2, 'eta': 0.41, 'objective': "reg:squarederror",
                  "alpha": 2e-3, "lambda": 3e-3, "nthread": 5, "eval_metric": "rmse"}


if __name__ == '__main__':
    start = time()
    oracle = Oracle(game_details=game_details, oracle_config=oracle_config, model_config=xgboost_config)
    oracle.run()
    end = time()
    print(f"Total solve time E2E: {round((end-start) / 60)} minutes")