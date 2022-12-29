from neural_networks.oracle import Oracle, Team


if __name__ == '__main__':
    predictors = ["FG_PCT", "FGA", "FG3A", "FG3_PCT", "FTA", "FT_PCT",
                  "AST", "TOV", "PF",
                  "REST_DAYS", "HOME", "AWAY", "PTS"]

    nn_config = {"predictors": predictors,  "num_seasons": 1, "input_shape": len(predictors)-1, "output_shape": 1,
                 "activation_func": "relu", "learning_rate": 1e-5, "output_activation_func": "relu",
                 "loss_function": "MSE", "optimizer_function": "SGD", "metrics": "mean_squared_error", "epochs": 300}
    
    oracle = Oracle("Nets", "Grizzlies", nn_config)
    home_team_forecast_df = oracle.get_team_forecast(Team.HOME)
    away_team_forecast_df = oracle.get_team_forecast(Team.AWAY)

    # Populate Rosters - assign minutes: int, None, DNP.
    # If DNP - remove player from active players.