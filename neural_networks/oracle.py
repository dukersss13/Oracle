import numpy as np
import pandas as pd

from data.data_fetcher import DataFetcher, Team
from neural_networks.neural_networks import SequentialNN


class Oracle:
    def __init__(self, home_team: str, away_team: str, nn_config: dict):
        """
        """
        self.data_fetcher = DataFetcher(home_team, away_team, nn_config)
        self.nn_config = nn_config
        self.home_team = home_team
        self.away_team = away_team

    def get_testing_data(self, player_game_logs: np.ndarray, holdout: int = 1, last_x_games: int = 5) -> tuple:
        """
        """
        x_test = np.mean(player_game_logs[holdout:holdout+last_x_games, :-1], axis=0)
        y_test = player_game_logs[0, -1]

        return x_test, y_test

    def prepare_training_data(self, player_game_logs: np.ndarray, holdout: int = 1) -> tuple:
        """
        """
        x_train, y_train = player_game_logs[holdout:, :-1], player_game_logs[holdout:, -1]

        return x_train, y_train

    def get_players_forecast(self, filtered_players_logs) -> int:
        """
        """
        holdout = 1
        last_x_games = 5

        x_train, y_train = self.prepare_training_data(filtered_players_logs, holdout=holdout)
        x_test, actual_points = self.get_testing_data(filtered_players_logs, holdout=holdout, last_x_games=last_x_games)
        players_trained_model = SequentialNN(self.nn_config)
        _ = players_trained_model.model.fit(x_train, y_train, epochs=self.nn_config["epochs"])
        forecasted_points = players_trained_model.model.predict(x_test.reshape(1, len(x_test)))[0][0]
        
        return forecasted_points, actual_points

    def get_team_forecast(self, team: Team):
        """
        """
        if team == Team.HOME:
            data = self.data_fetcher.home_team_data
        elif team == Team.AWAY:
            data = self.data_fetcher.away_team_data

        forecast_dict = dict(zip(["Name", "Forecasted Points", "Actual Points"], [[] for _ in range(3)]))
        for players_name in data.team_roster:
            filtered_players_logs = self.data_fetcher.get_filtered_players_logs(players_name)
            if filtered_players_logs is None:
                continue
            else:
                forecasted_points, actual_points = self.get_players_forecast(filtered_players_logs)
                forecast_dict["Name"].append(players_name)
                forecast_dict["Forecasted Points"].append(int(np.ceil(forecasted_points)))
                forecast_dict["Actual Points"].append(int(actual_points))

        forecast_df = pd.DataFrame(forecast_dict)
        total_forecasted_points = forecast_df["Forecasted Points"].sum()
        total_actual_points = forecast_df["Actual Points"].sum()
        
        print(f"{data.team_name} total forecasted points: {total_forecasted_points}")
        print(f"{data.team_name} total actual points: {total_actual_points}")

        return forecast_df
