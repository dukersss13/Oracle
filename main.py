from neural_networks.oracle import Oracle


if __name__ == '__main__':
    predictors = ["FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
                  "AST", "TOV", "PF", "MIN",
                  "REST_DAYS", "HOME", "AWAY", "PTS"]

    game_details = {"home_team": "Celtics", "away_team": "Mavericks", "game_date": "1-5-2023"}
    oracle_config = {"save_file": True, "output_path": "output"}

    nn_config = {"predictors": predictors, "num_seasons": 2, "holdout": True, "MA_degree": 8,
                 "input_shape": len(predictors)-1, "output_shape": 1, "validation_split": .10,
                 "activation_func": "relu", "learning_rate": 1e-5, "output_activation_func": "relu",
                 "loss_function": "MSE", "optimizer_function": "Adam", "metrics": "mean_squared_error", "epochs": 200}
    
    oracle = Oracle(game_details=game_details, oracle_config=oracle_config, nn_config=nn_config)
    oracle.run()

    # Take into account of opponents
    # 1. For each current opponent, search for their defense statistics + look at players' performance against 3 closest's
    # defensive teams for each category
