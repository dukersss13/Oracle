import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd

from data.data_fetcher import DataFetcher, Team
from neural_networks.neural_networks import SequentialNN


class Oracle:
    def __init__(self, game_details: dict, oracle_config: dict, nn_config: dict):
        """
        """
        self.home_team = game_details["home_team"]
        self.away_team = game_details["away_team"]
        self.game_date = game_details["game_date"]
        self.nn_config = nn_config
        self.holdout = nn_config["holdout"]
        self.last_x_games = nn_config["last_x_games"]

        self.data_fetcher = DataFetcher(self.home_team, self.away_team, self.nn_config)
        self.set_oracle_config(oracle_config)

        good_to_go = self.pause_to_set_active_players()
        if good_to_go:
            self.data_fetcher.set_active_players()

    def set_oracle_config(self, oracle_config: dict):
        """
        """
        self.save_output = oracle_config["save_file"]
        self.output_path = oracle_config["output_path"]

    def pause_to_set_active_players(self):
        """
        """
        print("\nSet active players in active_players.json.")
        print("Input 0 for injured/DNP. Else, leave as null")
        good_to_go = int(input(("Enter 1 to continue: ")))
        assert good_to_go

        return good_to_go

    def get_testing_data(self, player_game_logs: np.ndarray, most_recent_game_date: pd.Timestamp, team: Team) -> tuple:
        """
        """
        #TODO: Fix rest days
        home_or_away = np.array([1., 0.]) if team == Team.HOME else np.array([0., 1.])
        x_test_statistics = np.mean(player_game_logs[self.holdout: self.holdout + self.last_x_games, :-4], axis=0)
        rest_days = (pd.Timestamp(self.game_date) - most_recent_game_date).days - 1

        x_test = np.concatenate([x_test_statistics, [rest_days], home_or_away])
        if self.holdout == 0:
            y_test = 0
        else:
            y_test = player_game_logs[self.holdout-1, -1]

        return x_test, y_test

    def prepare_training_data(self, player_game_logs: np.ndarray) -> tuple:
        """
        Prepare the training data for given player.

        :param player_game_logs: game logs of individual player.
        :return: training predictors & outputs.
        """
        x_train, y_train = player_game_logs[self.holdout:, :-1], player_game_logs[self.holdout:, -1]

        return x_train, y_train
        
    def get_players_forecast(self, filtered_players_logs: np.ndarray, most_recent_game_date: pd.Timestamp, team: Team) -> int:
        """
        """
        x_train, y_train = self.prepare_training_data(filtered_players_logs)
        x_test, actual_points = self.get_testing_data(filtered_players_logs, most_recent_game_date, team)
        players_trained_model = SequentialNN(self.nn_config)
        _ = players_trained_model.model.fit(x_train, y_train, epochs=self.nn_config["epochs"], verbose=0)
        forecasted_points = players_trained_model.model.predict(x_test.reshape(1, len(x_test)))[0][0]
        
        return forecasted_points, actual_points

    def get_team_forecast(self, team: Team):
        """
        Trigger the forecast for given team.
        
        :param team: HOME or AWAY team.
        :return forecast_df: forecast df for given team.
        """
        if team == Team.HOME:
            data = self.data_fetcher.home_team_data
        elif team == Team.AWAY:
            data = self.data_fetcher.away_team_data

        forecast_dict = dict(zip(["Name", "Forecasted Points", "Actual Points"], [[] for _ in range(3)]))
        for players_name in data.active_players:
            print(f"Fetching game logs for: {players_name}")
            filtered_players_logs, most_recent_game_date = self.data_fetcher.get_filtered_players_logs(players_name)
            if filtered_players_logs is None:
                print(f"WARNING: Cannot forecast {players_name}'s performance")
                continue
            else:
                forecasted_points, actual_points = self.get_players_forecast(filtered_players_logs, most_recent_game_date, team)
                forecast_dict["Name"].append(players_name)
                forecast_dict["Forecasted Points"].append(int(np.ceil(forecasted_points)))
                forecast_dict["Actual Points"].append(int(actual_points))
            print(f"Finished forecasting for: {players_name}\n")

        forecast_df = Oracle.form_forecast_df(forecast_dict)

        return forecast_df

    @staticmethod
    def form_forecast_df(forecast_dict: dict) -> pd.DataFrame:
        """
        Make the forecast dataframe.

        :param forecast_dict: the dict containing players' forecasts.
        :return forecast_df: the df format of the dictionary.
        """
        forecast_df = pd.DataFrame(forecast_dict)
        total_forecasted_points = forecast_df["Forecasted Points"].sum()
        total_actual_points = forecast_df["Actual Points"].sum()
        
        totals = pd.DataFrame({"Name": ["Total"], "Forecasted Points": [total_forecasted_points], 
                               "Actual Points": [total_actual_points]})
        forecast_df = pd.concat([forecast_df, totals], axis=0)

        return forecast_df

    def save_forecasts(self, home_team_forecast_df: pd.DataFrame, away_team_forecast_df: pd.DataFrame):
        """
        Save the forecasts in excel files.

        :param home_team_forecast_df: forecast df for home team.
        :param away_team_forecast_df: forecast df for away team.
        """
        output_folder_name = f"{self.away_team}_@_{self.home_team}_{self.game_date}"
        dir_path = os.path.join(self.output_path, output_folder_name)

        if not os.path.exists(dir_path):
            print(f"Making {dir_path} directory path")
            os.mkdir(dir_path)
        
        print(f"Saving forecasts under {dir_path}")
        with pd.ExcelWriter(f"{dir_path}/Forecast.xlsx") as writer:
            home_team_forecast_df.to_excel(writer, sheet_name=f"{self.home_team} Forecast", index=False)
            away_team_forecast_df.to_excel(writer, sheet_name=f"{self.away_team} Forecast", index=False)

    def run(self):
        """
        Run Oracle.
        """
        print("Running Oracle")
        home_team_forecast_df = self.get_team_forecast(Team.HOME)
        away_team_forecast_df = self.get_team_forecast(Team.AWAY)
        
        if self.save_output:
            print("Saving output files")
            self.save_forecasts(home_team_forecast_df, away_team_forecast_df)
