
features = ["MIN", "GAME_DATE_x", "FGM", "FGA", "FG_PCT", "FG3M_x","FG3A_x", "FG3_PCT_x", "FTM", "FTA", "FT_PCT",
            "HOME", "AWAY", "REST_DAYS",
            "D_FGM", "D_FGA", "D_FG_PCT", "PCT_PLUSMINUS",
            "FG3M_y", "FG3A_y", "FG3_PCT_y", "NS_FG3_PCT", "PLUSMINUS_x",
            "FG2M", "FG2A", "FG2_PCT", "NS_FG2_PCT", "PLUSMINUS_y",
            "FGM_LT_10", "FGA_LT_10", "LT_10_PCT", "NS_LT_10_PCT", "PLUSMINUS",
            "E_PACE", "E_DEF_RATING", "PTS"]

game_details = {"home_team": "Lakers", "away_team": "Trail Blazers", "game_date": "11-12-2023", "new_game": False}

oracle_config = {"model": "Sequential", "features": features, "holdout": False, "MA_degree": 4,
                 "scaling_method": None, "save_file": True, "output_path": "output", "fetch_new_data": True}

nn_config = {"input_shape": len(features)-5, "output_shape": 1, "validation_split": .10,
             "activation_func": "relu", "learning_rate": 1e-4, "output_activation_func": "relu", "verbose": 1,
             "loss_function": "MSE", "optimizer_function": "Adam", "metrics": "mean_squared_error", "epochs": 500}

xgboost_config = {"max_depth": 4, 'eta': 0.41, 'objective': "reg:squarederror", "gamma": 1,
                  "alpha": 3e-3, "lambda": 3e-3, "nthread": 10, "eval_metric": "rmse"}
