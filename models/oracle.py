import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd

from data_prep.locker_room import LockerRoom, Team
from models.neural_networks import NeuralNet
from models.ml_models import XGBoost
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class Oracle:
    def __init__(self, game_details: dict, oracle_config: dict, model_config: dict):
        """
        Initialize the Oracle

        :param game_details: dict containing details of game to forecast
        :param oracle_config: config for Oracle
        :param nn_config: config Neural Network
        """
        self.model_config: dict = model_config
        self.game_date: str = game_details["game_date"]
        self.oracle_config = oracle_config

        self.locker_room = LockerRoom(game_details, oracle_config["features"],
                                      oracle_config["fetch_new_data"], oracle_config["holdout"])
        self.setup_oracle(oracle_config)

    def setup_oracle(self, oracle_config: dict):
        """
        Set the configuration for Oracle

        :param oracle_config: config dict for Oracle
        """
        self.scaler = None
        self.save_output: bool = oracle_config["save_file"]
        self.output_path: str = oracle_config["output_path"]
        self.scaling_method: str = oracle_config["scaling_method"]
        self.holdout: bool = oracle_config["holdout"]

        model = oracle_config["model"].upper()

        if model == "NN":
            self.points_predictor = NeuralNet(self.model_config)
        elif model == "XGBOOST":
            self.points_predictor = XGBoost(self.model_config)
        else:
            raise NotImplementedError(f"{self.points_predictor} is not implemented - select a different model!")

    def scale_input(self, X: np.ndarray) -> np.ndarray:
        """
        Scale input depending on the scaling method

        :param X: input to scale
        :param scaling_method: scaling method
        :return: scaled input
        """
        scaling_method = self.oracle_config["scaling_method"]

        if scaling_method not in ["standard", "minmax", None]:
            raise NotImplementedError(f"Do not recognize {scaling_method} scaling method!")

        if scaling_method is None:
            return X
        elif scaling_method.lower() == "standard":
            self.scaler = StandardScaler()
        elif scaling_method.lower() == "minmax":
            self.scaler = MinMaxScaler()
        
        return self.scaler.fit_transform(X)
    
    def scale_test(self, X_test: np.ndarray) -> np.ndarray:
        """
        Scale the testing inputs
        """
        if self.scaler is None:
            X_test = X_test
        else:
            X_test = self.scaler.transform(X_test)

        return X_test.astype(np.float32)

    def prepare_training_data(self, player_game_logs: np.ndarray) -> tuple:
        """
        Prepare the training data for given player

        :param player_game_logs: game logs of individual player
        :return: training predictors & outputs
        """
        cols_to_drop = ["GAME_DATE_player", "FGM", "FG3M_player", "FTM"]
        x_train, y_train = player_game_logs.iloc[1:, :-1].drop(cols_to_drop, axis=1), player_game_logs.iloc[1:, -1]

        x_train = self.scale_input(x_train.values.astype(np.float32)) 

        return x_train, y_train.values.astype(np.float32)

    @staticmethod
    def init_attempts_predictor(input_shape: int) -> NeuralNet:
        """
        Init predictor for shot attempts
        """
        attempts_predictor_config = {"type": "Normal", "input_shape": None, "output_shape": 1, "validation_split": .10,
          "activation_func": "relu", "learning_rate": 2e-4, "output_activation_func": "relu", "verbose": 0,
          "loss_function": "MSE", "optimizer_function": "Adam", "metrics": "mean_squared_error", "epochs": 300}
        
        attempts_predictor_config["input_shape"] = input_shape

        return NeuralNet(attempts_predictor_config)

    def prepare_testing_data(self, players_full_name: str, player_game_logs: pd.DataFrame,
                             most_recent_game_date: pd.Timestamp, team: Team) -> np.ndarray:
        """
        Get the input parameters for the test set

        :param player_game_logs: player's game logs matrix
        :param most_recent_game_date: date of player's most recent game
        :param team: home or away
        :return: x_test & y_test (if applicable)
        """
        ma_degree: int = self.oracle_config["MA_degree"]
        home_or_away = np.array([1., 0.]) if team == Team.HOME else np.array([0., 1.])
        rest_days = (pd.Timestamp(self.game_date) - most_recent_game_date).days

        x_test_defense = self.locker_room.get_opponent_defensive_stats(team)
        data_for_fga_pred = self.locker_room.prepare_training_data(player_game_logs, "MIN", "FGA")
        # Use defensive stats to forecast fg3a & fta
        data_for_fg3a_pred = self.locker_room.prepare_training_data(player_game_logs,
                                                                    ["MIN", "FGA"], "FG3A_player")
        data_for_fta_pred = self.locker_room.prepare_training_data(player_game_logs,
                                                                   ["MIN", "LT_10_PCT", "NS_LT_10_PCT",
                                                                    "E_PACE", "E_DEF_RATING"], "FTA")
        testing_mins = self.get_player_mins(players_full_name, player_game_logs, team)

        # Use mins to forecast fga
        fga_predictor = Oracle.init_attempts_predictor(input_shape=1)
        fga = round(fga_predictor.get_forecast(data_for_fga_pred, testing_mins))
        # Use mins, fga, defense's fg3m, fg3a to forecast fg3a
        fg3a_predictor = Oracle.init_attempts_predictor(input_shape=2)
        fg3a = fg3a_predictor.get_forecast(data_for_fg3a_pred,
                                           np.concatenate([testing_mins, [fga]]).reshape(1, -1))
        # Use mins, fga and fg3a to forecast fta
        fta_predictor = Oracle.init_attempts_predictor(input_shape=5)
        fta = fta_predictor.get_forecast(data_for_fta_pred,
                                         np.concatenate([testing_mins,
                                                         x_test_defense[["LT_10_PCT", "NS_LT_10_PCT",
                                                                         "E_PACE", "E_DEF_RATING"]]]).reshape(1, -1))

        test_fg_pct = Oracle.get_pct(player_game_logs["FGM"].values[:ma_degree].sum(), 
                                     player_game_logs["FGA"].values[:ma_degree].sum())

        test_3fg_pct = Oracle.get_pct(player_game_logs["FG3M_player"].values[:ma_degree].sum(),
                                      player_game_logs["FG3A_player"].values[:ma_degree].sum())

        test_ft_pct = Oracle.get_pct(player_game_logs["FTM"].values[:ma_degree].sum(),
                                     player_game_logs["FTA"].values[:ma_degree].sum())

        # Reconstruct x_test
        x_test = np.concatenate([testing_mins, 
                                 [fga, test_fg_pct, fg3a,
                                  test_3fg_pct, fta, test_ft_pct], home_or_away, [rest_days], x_test_defense.values])

        return x_test.reshape(1, -1)

    @staticmethod
    def get_pct(x: int, y: int) -> float:
        """
        Get pct x / y
        """
        if y > 0.0:
            pct = np.float32(x / y)
        else:
            pct = 0.0
        
        return pct

    def get_players_forecast(self, players_full_name: str, filtered_players_logs: pd.DataFrame, team: Team) -> int:
        """
        Get players' forecast

        :param players_full_name: full name of player
        :param filtered_players_logs: players' game logs
        """
        if filtered_players_logs.empty or filtered_players_logs["MIN"].values[:3].mean() <= 5:
            return 0

        elif filtered_players_logs.shape[0] < 15:
            print(f"WARNING: {players_full_name} has only played {filtered_players_logs.shape[0]} games.")
            print(f"WARNING: Cannot run forecast for {players_full_name}. Will use player's average as forecast")
            return int(filtered_players_logs.iloc[:, -1].mean())

        filtered_players_logs.dropna(inplace=True)
        most_recent_game_date = self.locker_room.get_most_recent_game_date(filtered_players_logs)
        training_data = self.prepare_training_data(filtered_players_logs)
        x_test = self.prepare_testing_data(players_full_name, filtered_players_logs, most_recent_game_date, team)
        x_test = self.scale_test(x_test)

        forecasted_points = self.points_predictor.get_forecast(training_data, x_test)

        return int(round(forecasted_points))

    def get_team_forecast(self, team: Team):
        """
        Trigger the forecast for given team
        
        :param team: HOME or AWAY team
        :return forecast_df: forecast df for given team
        """
        if team == Team.HOME:
            data = self.locker_room.home_game_plan
        elif team == Team.AWAY:
            data = self.locker_room.away_game_plan

        forecast_dict = dict(zip(["PLAYER_NAME", "FORECASTED_POINTS", "ACTUAL_POINTS"], [[] for _ in range(3)]))
        total_players = len(data.active_players)
        players_done = 0

        print(f"\nStarting forecast for the {data.team_name}")

        for players_name, players_id in data.active_players.iterrows():
            print(f"\nFetching game logs for: {players_name}")
            filtered_players_logs, actual_points = self.locker_room.get_filtered_players_logs(players_id)

            print(f"Starting forecast for: {players_name}")
            forecasted_points = self.get_players_forecast(players_name, filtered_players_logs, team)
            print(f"Forecasted points: {forecasted_points}")
            forecast_dict = Oracle.append_to_forecast_dict(forecast_dict, players_name, forecasted_points, actual_points)
            players_done += 1

            print(f"Finished forecasting for: {players_name}")
            print(f"\n{players_done}/{total_players} players done for the {data.team_name}")

        forecast_df = self.form_forecast_df(forecast_dict)

        return forecast_df

    def get_player_mins(self, players_full_name: str,
                        players_game_log: pd.DataFrame, team: Team) -> np.float32:
        """
        Manually assign player's minutes

        :param players_full_name: player's full name
        :param x_test: X test (input predictors for NN)
        :return: x_test: X Test (input predictors for NN)
        """
        players_mins_data = self.locker_room.home_game_plan.players_mins if team == Team.HOME else \
                            self.locker_room.away_game_plan.players_mins

        if players_mins_data[players_full_name] is not None:
            mins = players_mins_data[players_full_name]
        else:
            mins = players_game_log["MIN"].values[:self.oracle_config["MA_degree"]].mean()
        
        return np.array([np.float32(round(mins))])

    @staticmethod
    def append_to_forecast_dict(forecast_dict: dict, players_name: str, forecasted_points: int, actual_points: int) -> dict:
        """
        Append forecasted statistics to forecast dict

        :param forecast_dict: forecast dictionary
        :param players_name: player's full name
        :param forecasted_points: forecasted points (NN output)
        :return: team's forecast dictionary
        """
        forecast_dict["PLAYER_NAME"].append(players_name)
        forecast_dict["FORECASTED_POINTS"].append(forecasted_points)
        forecast_dict["ACTUAL_POINTS"].append(actual_points)

        return forecast_dict

    def form_forecast_df(self, forecast_dict: dict) -> pd.DataFrame:
        """
        Make the forecast dataframe

        :param forecast_dict: the dict containing players' forecasts
        :return forecast_df: the df format of the dictionary
        """
        forecast_df = pd.DataFrame(forecast_dict)
        totals = pd.DataFrame({"PLAYER_NAME": ["Total"], "FORECASTED_POINTS": [forecast_df["FORECASTED_POINTS"].sum()],
                               "ACTUAL_POINTS": [forecast_df["ACTUAL_POINTS"].sum()]})
        forecast_df = pd.concat([forecast_df, totals], axis=0)

        return forecast_df

    def save_forecasts(self, home_team_forecast_df: pd.DataFrame, away_team_forecast_df: pd.DataFrame):
        """
        Save the forecasts in excel files

        :param home_team_forecast_df: forecast df for home team
        :param away_team_forecast_df: forecast df for away team
        """
        output_folder_name = f"{self.locker_room.away_team}_@_{self.locker_room.home_team}_{self.game_date}"
        output_path = os.path.join(self.output_path, output_folder_name)

        if not os.path.exists(output_path):
            print(f"Making {output_path} output path")
            os.mkdir(output_path)
        
        print(f"Saving forecasts under {output_path}")
        with pd.ExcelWriter(f"{output_path}/Forecast.xlsx") as writer:
            home_team_forecast_df.to_excel(writer, sheet_name=f"{self.locker_room.home_team} Forecast", index=False)
            away_team_forecast_df.to_excel(writer, sheet_name=f"{self.locker_room.away_team} Forecast", index=False)
        
        print("Saving config files")
        with open(f"{output_path}/oracle_config.json", "w") as json_file:
            json.dump(self.oracle_config, json_file, indent=2)

        with open(f"{output_path}/model_config.json", "w") as json_file:
            json.dump(self.model_config, json_file, indent=2)

    def run(self):
        """
        Run Oracle
        """
        print("Running Oracle")
        home_team_forecast_df = self.get_team_forecast(Team.HOME)
        away_team_forecast_df = self.get_team_forecast(Team.AWAY)
        
        print(home_team_forecast_df)
        print(away_team_forecast_df)

        if self.save_output:
            print("Saving output files")
            self.save_forecasts(home_team_forecast_df, away_team_forecast_df)
