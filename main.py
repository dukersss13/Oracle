from neural_networks.oracle import Oracle


columns = ["MIN", "GAME_DATE_x", "FGA", "FG3A_x", "FTA", "HOME", "AWAY", "REST_DAYS",
           "D_FGM", "D_FGA", "D_FG_PCT", "PCT_PLUSMINUS",
           "FG3M_y", "FG3A_y", "FG3_PCT_y", "NS_FG3_PCT", "PLUSMINUS_x",
           "FG2M", "FG2A", "FG2_PCT", "NS_FG2_PCT", "PLUSMINUS_y",
           "FGM_LT_10", "FGA_LT_10", "LT_10_PCT", "NS_LT_10_PCT", "PLUSMINUS",
           "FG2M", "FG2A", "FG2_PCT", "NS_FG2_PCT",
           "E_PACE", "E_DEF_RATING", "PTS"]

game_details = {"home_team": "Heat", "away_team": "Mavericks", "game_date": "4-1-2023"}
oracle_config = {"save_file": True, "output_path": "output"}

nn_config = {"columns": columns, "holdout": True, "MA_degree": 8,
             "input_shape": len(columns)-1, "output_shape": 1, "validation_split": .15,
             "activation_func": "relu", "learning_rate": 2e-4, "output_activation_func": "relu",
             "loss_function": "MSE", "optimizer_function": "Adam", "metrics": "mean_squared_error", "epochs": 500}

if __name__ == '__main__':
    oracle = Oracle(game_details=game_details, oracle_config=oracle_config, nn_config=nn_config)
    oracle.run()

    # TODO
    # Need to make function to merge opposing defense to player's game logs
    # Adjust existing NN + add XGB tree
