from neural_networks.oracle import Oracle


if __name__ == '__main__':
    predictors = ["FG_PCT", "FGA", "FG3A", "FG3_PCT", "FTA", "FT_PCT",
                  "AST", "TOV", "PF",
                  "REST_DAYS", "HOME", "AWAY", "PTS"]

    game_details = {"home_team": "Pelicans", "away_team": "76ers", "game_date": "12-30-2022"}
    oracle_config = {"save_file": True, "output_path": "output"}

    nn_config = {"predictors": predictors, "num_seasons": 1.5, "holdout": 1, "last_x_games": 5,
                 "input_shape": len(predictors)-1, "output_shape": 1,
                 "activation_func": "relu", "learning_rate": 1e-5, "output_activation_func": "relu",
                 "loss_function": "MSE", "optimizer_function": "SGD", "metrics": "mean_squared_error", "epochs": 1}
    
    oracle = Oracle(game_details=game_details, oracle_config=oracle_config, nn_config=nn_config)
    oracle.run()
