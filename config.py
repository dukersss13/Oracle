features = ["MIN", "GAME_DATE_player",
            "FGM", "FGA", "FG_PCT",
            "FG3M_player","FG3A_player", "FG3_PCT_player",
            "FTM", "FTA", "FT_PCT",
            "HOME", "AWAY", "REST_DAYS",
            "D_FGM", "D_FGA", "D_FG_PCT",
            "FG3M_opp_defense", "FG3A_opp_defense", "FG3_PCT_opp_defense", "NS_FG3_PCT",
            "FG2M", "FG2A", "FG2_PCT", "NS_FG2_PCT",
            "FGM_LT_10", "FGA_LT_10", "LT_10_PCT", "NS_LT_10_PCT",
            "E_PACE", "E_DEF_RATING", "PTS"]

game_details = {"home_team": "Timberwolves", "away_team": "Clippers", "game_date": "01-14-2024", "new_game": False}

oracle_config = {"model": "NN", "features": features, "holdout": False, "MA_degree": 4,
                 "scaling_method": "standard", "save_file": True, "output_path": "output", "fetch_new_data": False}

nn_config = {"type": "Normal","input_shape": len(features)-5, "output_shape": 1, "validation_split": .10,
             "activation_func": "relu", "learning_rate": 3e-3, "output_activation_func": "relu", "verbose": 1,
             "loss_function": "MSE", "optimizer_function": "Adam", "metrics": "mean_squared_error", "epochs": 500}

xgboost_config = {"max_depth": 4, 'eta': 0.41, 'objective': "reg:squarederror", "gamma": 1,
                  "alpha": 3e-3, "lambda": 3e-3, "nthread": 10, "eval_metric": "rmse"}
