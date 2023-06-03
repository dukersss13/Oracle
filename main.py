from neural_networks.oracle import Oracle


predictors = ["FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
              "AST", "TOV", "PF", "MIN",
              "REST_DAYS", "HOME", "AWAY", "PTS"]

game_details = {"home_team": "Celtics", "away_team": "Mavericks", "game_date": "1-5-2023"}
oracle_config = {"save_file": True, "output_path": "output"}

nn_config = {"predictors": predictors, "num_seasons": 2, "holdout": True, "MA_degree": 8,
                "input_shape": len(predictors)-1, "output_shape": 1, "validation_split": .15,
                "activation_func": "relu", "learning_rate": 2e-4, "output_activation_func": "relu",
                "loss_function": "MSE", "optimizer_function": "Adam", "metrics": "mean_squared_error", "epochs": 500}

if __name__ == '__main__':
    oracle = Oracle(game_details=game_details, oracle_config=oracle_config, nn_config=nn_config)
    oracle.run()

    # TODO
    # Need to make function to merge opposing defense to player's game logs
    # Adjust existing NN + add XGB tree
