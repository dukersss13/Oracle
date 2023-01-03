from neural_networks.oracle import Oracle


if __name__ == '__main__':
    predictors = ["FG_PCT", "FGA", "FG3_PCT", "FG3A", "FT_PCT", "FTA",
                  "AST", "TOV", "PF", "MIN",
                  "REST_DAYS", "HOME", "AWAY", "PTS"]

    game_details = {"home_team": "Pacers", "away_team": "Raptors", "game_date": "1-2-2023"}
    oracle_config = {"save_file": True, "output_path": "output"}

    nn_config = {"predictors": predictors, "num_seasons": 2, "holdout": True, "validation_set": 8,
                 "input_shape": len(predictors)-1, "output_shape": 1,
                 "activation_func": "relu", "learning_rate": 1e-5, "output_activation_func": "relu",
                 "loss_function": "MSE", "optimizer_function": "SGD", "metrics": "mean_squared_error", "epochs": 200}
    
    oracle = Oracle(game_details=game_details, oracle_config=oracle_config, nn_config=nn_config)
    oracle.run()

    # Take into account of teammates
    # Take into account of opponents
