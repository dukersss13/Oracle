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


fga_features = ["MIN", "FGA"]
fg3a_features = ["MIN", "FG3A_player"]
fta_features = ["MIN", "LT_10_PCT", "NS_LT_10_PCT", "E_DEF_RATING", "FTA"]

game_details = {"home_team": "76ers", "away_team": "Nuggets", "game_date": "01-16-2024", "new_game": False}

fga_nn = {"type": "Normal", "input_shape": 1, "output_shape": 1, "validation_split": .15,
          "activation_func": "relu", "learning_rate": 2e-4, "output_activation_func": "relu", "verbose": 1,
          "loss_function": "MSE", "optimizer_function": "Adam", "metrics": "mean_squared_error", "epochs": 200}

nn_config = {"type": "Normal", "input_shape": len(features)-5, "output_shape": 1, "validation_split": .10,
             "activation_func": "relu", "learning_rate": 2e-4, "output_activation_func": "relu", "verbose": 1,
             "loss_function": "MSE", "optimizer_function": "Adam", "metrics": "mean_squared_error", "epochs": 300}

oracle_config = {"model": "NN", "features": features, "fga_features": fga_nn, "holdout": False, "MA_degree": 5,
                 "scaling_method": "standard", "save_file": True, "output_path": "output", "fetch_new_data": False}

xgboost_config = {"max_depth": 4, 'eta': 0.41, 'objective': "reg:squarederror", "gamma": 1,
                  "alpha": 3e-3, "lambda": 3e-3, "nthread": 10, "eval_metric": "rmse"}
