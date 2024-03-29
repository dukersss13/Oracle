features = ["MIN", "GAME_DATE_player",
            "FGM", "FGA", "FG_PCT",
            "FG3M_player", "FG3A_player", "FG3_PCT_player",
            "FTM", "FTA", "FT_PCT",
            "HOME", "AWAY", "REST_DAYS",
            "D_FGM", "D_FGA", "D_FG_PCT",
            "FG3M_opp_defense", "FG3A_opp_defense", "FG3_PCT_opp_defense", "NS_FG3_PCT",
            "FG2M", "FG2A", "FG2_PCT", "NS_FG2_PCT",
            "FGM_LT_10", "FGA_LT_10", "LT_10_PCT", "NS_LT_10_PCT",
            "E_PACE", "E_DEF_RATING", "PTS"]

game_details = {"home_team": "Warriors", "away_team": "Pacers", "game_date": "03-22-2024", "new_game": False}

nn_config = {"type": "GRU", "input_shape": len(features)-5, "output_shape": 1, "validation_split": .15,
             "activation_func": "relu", "learning_rate": 2e-3, "output_activation_func": "relu", "verbose": False,
             "loss_function": "MSE", "optimizer_function": "Adam", "metrics": "mean_squared_error", "epochs": 1000, 
             "timesteps": 4, "scaling_method": "standard", "patience": 500}

oracle_config = {"model": "NN", "features": features, "holdout": True, "save_file": True, "fetch_new_data": False}

xgboost_config = {"max_depth": 4, 'eta': 0.41, 'objective': "reg:squarederror", "gamma": 1,
                  "alpha": 3e-3, "lambda": 3e-3, "nthread": 10, "eval_metric": "rmse"}
