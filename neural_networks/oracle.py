import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd

from data.data_fetcher import DataFetcher, Team
from neural_networks.neural_networks import SequentialNN


class Oracle:
    def __init__(self, game_details: dict, oracle_config: dict, nn_config: dict):
        """
        Initialize the Oracle

        :param game_details: dict containing details of game to forecast
        :param oracle_config: _description_
        :param nn_config: _description_
        """
        self.game_date: str = game_details["game_date"]
        self.nn_config: dict = nn_config
        self.validation_set: int = nn_config["validation_set"]
        self.holdout: bool = nn_config["holdout"]

        self.data_fetcher = DataFetcher(game_details, nn_config)
        self.set_oracle_config(oracle_config)

        good_to_go = self.pause_to_set_active_players()
        if good_to_go:
            self.data_fetcher.set_active_players()
        
        if nn_config["holdout"]:
            self.box_score = self.data_fetcher.fetch_game_box_score(self.game_date)

    def set_oracle_config(self, oracle_config: dict):
        """
        Set the configuration for Oracle

        :param oracle_config: config dict for Oracle
        """
        self.save_output: bool = oracle_config["save_file"]
        self.output_path: str = oracle_config["output_path"]

    def pause_to_set_active_players(self):
        """
        Pauses the program so user can set the lineups
        """
        print("\nSet active players in active_players.json.")
        print("Input 0 for injured/DNP. Else, leave as null")
        good_to_go = int(input(("Enter 1 to continue: ")))
        assert good_to_go

        return good_to_go

    def prepare_training_data(self, player_game_logs: np.ndarray) -> tuple:
        """
        Prepare the training data for given player

        :param player_game_logs: game logs of individual player
        :return: training predictors & outputs
        """
        x_train, y_train = player_game_logs[self.validation_set:, :-1], player_game_logs[self.validation_set:, -1]

        return x_train, y_train

    def prepare_validation_data(self, player_game_logs: np.ndarray) -> tuple:
        """
        Prepare the validation set for the NN

        :param player_game_logs: game logs of individual player
        :return: validation X & Y sets
        """
        x_val, y_val = player_game_logs[:self.validation_set, :-1], player_game_logs[:self.validation_set, -1]

        return x_val, y_val

    def prepare_testing_data(self, player_game_logs: np.ndarray, most_recent_game_date: pd.Timestamp, team: Team) -> np.ndarray:
        """
        Get the input parameters for the test set

        :param player_game_logs: player's game logs matrix
        :param most_recent_game_date: date of player's most recent game
        :param team: home or away
        :return: x_test & y_test (if applicable)
        """
        home_or_away = np.array([1., 0.]) if team == Team.HOME else np.array([0., 1.])
        x_test_statistics = np.mean(player_game_logs[:self.validation_set, :-4], axis=0)
        rest_days = (pd.Timestamp(self.game_date) - most_recent_game_date).days

        x_test = np.concatenate([x_test_statistics, [rest_days], home_or_away])

        return x_test
        
    def get_players_forecast(self, players_full_name: str, filtered_players_logs: np.ndarray,
                             most_recent_game_date: pd.Timestamp, team: Team) -> int:
        """
        """
        if filtered_players_logs.shape[0] < 15:
            print(f"WARNING: {players_full_name} has only played {filtered_players_logs.shape[0]} games.")
            print(f"WARNING: Cannot run forecast for {players_full_name}. Will use player's average as forecast")
            return int(np.mean(filtered_players_logs[:, -1]))

        x_train, y_train = self.prepare_training_data(filtered_players_logs)
        x_val, y_val = self.prepare_validation_data(filtered_players_logs)
        x_test = self.prepare_testing_data(filtered_players_logs, most_recent_game_date, team)
        x_test = self.assign_player_mins(players_full_name, x_test, team)

        players_trained_model = SequentialNN(self.nn_config)
        _ = players_trained_model.model.fit(x_train, y_train, batch_size=32, epochs=self.nn_config["epochs"], 
                                            verbose=0, validation_data=(x_val, y_val),)
        forecasted_points = players_trained_model.model.predict(x_test.reshape(1, len(x_test)))[0][0]
        
        return int(forecasted_points)

    def get_team_forecast(self, team: Team):
        """
        Trigger the forecast for given team
        
        :param team: HOME or AWAY team
        :return forecast_df: forecast df for given team
        """
        if team == Team.HOME:
            data = self.data_fetcher.home_team_data
        elif team == Team.AWAY:
            data = self.data_fetcher.away_team_data

        forecast_dict = dict(zip(["PLAYER_NAME", "FORECASTED_POINTS"], [[] for _ in range(2)]))
        total_players = len(data.active_players)
        players_done = 0

        print(f"\nStarting forecast for the {data.team_name}")
        for players_name, players_id in data.active_players.iterrows():
            print(f"\nFetching game logs for: {players_name}")
            filtered_players_logs, most_recent_game_date = self.data_fetcher.get_filtered_players_logs(int(players_id))

            print(f"Starting forecast for: {players_name}")
            forecasted_points = self.get_players_forecast(players_name, filtered_players_logs, 
                                                          most_recent_game_date, team)
            forecast_dict = Oracle.append_to_forecast_dict(forecast_dict, players_name, forecasted_points)
            players_done += 1

            print(f"Finished forecasting for: {players_name}")
            print(f"\n{players_done}/{total_players} players done for the {data.team_name}")

        forecast_df = self.form_forecast_df(forecast_dict)

        return forecast_df

    def assign_player_mins(self, players_full_name: str, x_test: np.ndarray, team: Team) -> np.ndarray:
        """
        Manually assign player's minutes

        :param x_test: X test (input predictors for NN)
        :return: x_test: X Test (input predictors for NN)
        """
        players_mins_data = self.data_fetcher.home_team_data.players_mins if team == Team.HOME else \
                            self.data_fetcher.away_team_data.players_mins
        if players_mins_data[players_full_name] is not None:
            x_test[-4] = np.float(players_mins_data[players_full_name])
        
        return x_test

    @staticmethod
    def append_to_forecast_dict(forecast_dict: dict, players_name: str, forecasted_points: float) -> dict:
        """
        Append forecasted statistics to forecast dict

        :param forecast_dict: forecast dictionary
        :param players_name: player's full name
        :param forecasted_points: forecasted points (NN output)
        :return: team's forecast dictionary
        """
        forecast_dict["PLAYER_NAME"].append(players_name)
        forecast_dict["FORECASTED_POINTS"].append(forecasted_points)

        return forecast_dict  

    def form_forecast_df(self, forecast_dict: dict) -> pd.DataFrame:
        """
        Make the forecast dataframe

        :param forecast_dict: the dict containing players' forecasts
        :return forecast_df: the df format of the dictionary
        """
        forecast_df = pd.DataFrame(forecast_dict)

        if self.holdout:
            forecast_df = forecast_df.merge(self.box_score, how="left")[["PLAYER_NAME", "FORECASTED_POINTS", "PTS"]]
            forecast_df["PTS"] = forecast_df["PTS"]
            total_actual_points = forecast_df["PTS"].sum()
        else:
            forecast_df["PTS"] = np.zeros(len(forecast_df))
            total_actual_points = 0

        totals = pd.DataFrame({"PLAYER_NAME": ["Total"], "FORECASTED_POINTS": [forecast_df["FORECASTED_POINTS"].sum()],
                               "PTS": [total_actual_points]})
        forecast_df = pd.concat([forecast_df, totals], axis=0)

        return forecast_df

    def save_forecasts(self, home_team_forecast_df: pd.DataFrame, away_team_forecast_df: pd.DataFrame):
        """
        Save the forecasts in excel files

        :param home_team_forecast_df: forecast df for home team
        :param away_team_forecast_df: forecast df for away team
        """
        output_folder_name = f"{self.data_fetcher.away_team}_@_{self.data_fetcher.home_team}_{self.game_date}"
        dir_path = os.path.join(self.output_path, output_folder_name)

        if not os.path.exists(dir_path):
            print(f"Making {dir_path} directory path")
            os.mkdir(dir_path)
        
        print(f"Saving forecasts under {dir_path}")
        with pd.ExcelWriter(f"{dir_path}/Forecast.xlsx") as writer:
            home_team_forecast_df.to_excel(writer, sheet_name=f"{self.data_fetcher.home_team} Forecast", index=False)
            away_team_forecast_df.to_excel(writer, sheet_name=f"{self.data_fetcher.away_team} Forecast", index=False)

    def run(self):
        """
        Run Oracle
        """
        print("Running Oracle")
        home_team_forecast_df = self.get_team_forecast(Team.HOME)
        away_team_forecast_df = self.get_team_forecast(Team.AWAY)
        
        if self.save_output:
            print("Saving output files")
            self.save_forecasts(home_team_forecast_df, away_team_forecast_df)
