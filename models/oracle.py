import json
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from pyhocon import ConfigFactory

from data_prep.locker_room import LockerRoom, Team
from models.neural_networks import NeuralNet
from models.ml_models import XGBoost
from models.transformer import TeamTransformer


logger = logging.getLogger(__name__)

_TRANSFORMER_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts", "models", "team_transformer")


class Oracle:
    def __init__(self, game_details: dict | None = None, oracle_config: dict | None = None,
                 model_config: dict | None = None, active_players_dict: dict | None = None,
                 progress_callback=None, non_interactive: bool = False,
                 force_refresh_cache: bool = False):
        """
        Initialize the Oracle
        """
        self.progress_callback = progress_callback
        self.active_players_dict = active_players_dict
        self.non_interactive = non_interactive
        self.force_refresh_cache = force_refresh_cache
        self._setup_config_files(game_details, oracle_config, model_config)
        self.setup_oracle()
        logger.info(
            "Oracle initialized (model=%s, holdout=%s, non_interactive=%s)",
            self.oracle_config["model"],
            self.oracle_config["holdout"],
            self.non_interactive,
        )

    def _setup_config_files(self, game_details: dict | None, oracle_config: dict | None,
                            model_config_override: dict | None):
        """
        Set up config files
        """
        if game_details is not None and oracle_config is not None:
            self.game_date = game_details["game_date"]
            self.oracle_config = oracle_config
        else:
            oracle_conf = ConfigFactory.parse_file("oracle.conf")
            game_details = oracle_conf["game_details"]
            self.game_date = game_details["game_date"]
            self.oracle_config = oracle_conf["oracle_config"]

        model_config = ConfigFactory.parse_file("model.conf")
        model_chosen = self.oracle_config["model"].upper()
        if model_config_override is not None:
            self.model_config = model_config_override
        elif model_chosen == "NN":
            self.model_config = model_config["nn_config"]
        elif model_chosen == "XGBOOST":
            self.model_config = model_config["xgboost_config"]
        elif model_chosen == "TRANSFORMER":
            self.model_config = model_config["transformer_config"]
        else:
            raise ValueError(f"{model_chosen} is not a valid selection")

        self.locker_room = LockerRoom(
            game_details,
            self.oracle_config["features"],
            self.oracle_config["fetch_new_data"],
            self.oracle_config["holdout"],
            active_players_override=self.active_players_dict,
            interactive=not self.non_interactive,
            force_refresh_cache=self.force_refresh_cache,
            cache_strategy=self.oracle_config.get("cache_strategy", os.environ.get("ORACLE_CACHE_STRATEGY", "incremental")),
            cache_ttl_hours=int(self.oracle_config.get("cache_ttl_hours", os.environ.get("ORACLE_CACHE_TTL_HOURS", "12"))),
        )

    def setup_oracle(self):
        """
        Set the configuration for Oracle

        :param oracle_config: config dict for Oracle
        """
        self.scaler = None
        self.save_output: bool = self.oracle_config["save_file"]
        self.output_path: str = "output"
        self.holdout: bool = self.oracle_config["holdout"]

        model = self.oracle_config["model"].upper()

        if model == "NN":
            self.points_predictor = None
            self.fga_predictor = None
            self.fg3a_predictor = None
        elif model == "XGBOOST":
            self.points_predictor = None
        elif model == "TRANSFORMER":
            self._transformer = None
            self._enriched_logs = None
        else:
            raise NotImplementedError(f"{self.points_predictor} is not implemented - select a different model!")
        logger.info("Oracle setup complete (model=%s, save_output=%s)", model, self.save_output)

    def _init_predictors(self):
        fga_predictor = Oracle.init_attempts_predictor(input_shape=1)
        fg3a_predictor = Oracle.init_attempts_predictor(input_shape=2)
        model = self.oracle_config["model"].upper()
        if model == "NN":
            points_predictor = NeuralNet(self.model_config)
            return points_predictor, fga_predictor, fg3a_predictor
        if model == "XGBOOST":
            return XGBoost(self.model_config), fga_predictor, fg3a_predictor
        raise NotImplementedError(f"{model} is not implemented - select a different model!")

    def _timesteps(self) -> int:
        return int(self.model_config.get("timesteps", 4))

    def _model_type(self) -> str:
        return str(self.model_config.get("type", self.oracle_config.get("model", "NN"))).upper()

    def prepare_training_data(self, player_game_logs: np.ndarray) -> tuple:
        """
        Prepare the training data for given player

        :param player_game_logs: game logs of individual player
        :return: training predictors & outputs
        """
        cols_to_drop = ["GAME_DATE_player", "FGM", "FG3M_player", "FTM"]
        x_train, y_train = player_game_logs.iloc[1:, :-1].drop(cols_to_drop, axis=1), player_game_logs.iloc[1:, -1]

        return x_train.values, y_train.values

    @staticmethod
    def init_attempts_predictor(input_shape: int) -> NeuralNet:
        """
        Init predictor for shot attempts
        """
        attempts_predictor_config = {"type": "Normal", "input_shape": input_shape, "output_shape": 1, "validation_split": .10,
          "activation_func": "relu", "learning_rate": 1e-3, "output_activation_func": "relu", "verbose": False,
          "loss_function": "MSE", "optimizer_function": "Adam", "metrics": "mean_squared_error", "epochs": 500,
          "timesteps": 0, "scaling_method": "standard", "patience": 300}

        return NeuralNet(attempts_predictor_config)

    def prepare_testing_data(self, players_full_name: str, player_game_logs: pd.DataFrame,
                             most_recent_game_date: pd.Timestamp, team: Team,
                             fga_predictor, fg3a_predictor) -> np.ndarray:
        """
        Get the input parameters for the test set

        :param player_game_logs: player's game logs matrix
        :param most_recent_game_date: date of player's most recent game
        :param team: home or away
        :return: x_test & y_test (if applicable)
        """
        timesteps: int = self._timesteps()
        home_or_away = np.array([1., 0.]) if team == Team.HOME else np.array([0., 1.])
        rest_days = (pd.Timestamp(self.game_date) - most_recent_game_date).days

        x_test_defense = self.locker_room.get_opponent_defensive_stats(team)

        # Use mins, fga, defense's fg3m, fg3a to forecast fg3a
        data_for_fga_pred = self.locker_room.prepare_training_data(player_game_logs, "MIN", "FGA")
        testing_mins = self.get_player_mins(players_full_name, player_game_logs, team)

        # Use mins to forecast fga
        
        fga = round(fga_predictor.get_forecast(data_for_fga_pred, testing_mins))
        
        # Use defensive stats to forecast fg3a & fta
        if player_game_logs["FG3A_player"].mean() <= 5:
            fg3a = player_game_logs["FG3A_player"].values[:timesteps].mean()
        else:
            data_for_fg3a_pred = self.locker_room.prepare_training_data(player_game_logs,
                                                                        ["MIN", "FGA"], "FG3A_player")
            fg3a = fg3a_predictor.get_forecast(data_for_fg3a_pred,
                                               np.concatenate([testing_mins, [fga]]))

        fta = round(player_game_logs["FTA"].values[:timesteps].mean())

        test_fg_pct = Oracle.get_pct(player_game_logs["FGM"].values[:timesteps].sum(), 
                                     player_game_logs["FGA"].values[:timesteps].sum())

        test_3fg_pct = Oracle.get_pct(player_game_logs["FG3M_player"].values[:timesteps].sum(),
                                      player_game_logs["FG3A_player"].values[:timesteps].sum())

        test_ft_pct = Oracle.get_pct(player_game_logs["FTM"].values[:timesteps].sum(),
                                     player_game_logs["FTA"].values[:timesteps].sum())

        # Reconstruct x_test
        starting_idx = int(self.holdout)
        end_idx = timesteps - 1 if not self.holdout else timesteps
        x_test_previous = player_game_logs.iloc[starting_idx:end_idx, :][self.oracle_config["features"]].\
                   drop(columns=["GAME_DATE_player", "FGM", "FG3M_player", "FTM", "PTS"]).values
        x_test = np.concatenate([testing_mins, 
                                [fga, test_fg_pct, fg3a,
                                 test_3fg_pct, fta, test_ft_pct],
                                 home_or_away, [rest_days], x_test_defense.values])

        return np.concatenate([x_test.reshape(1, -1), x_test_previous])

    @staticmethod
    def get_pct(x: int, y: int) -> float:
        """
        Get pct x / y
        """
        if y > 0.0:
            pct = (x / y)
        else:
            pct = 0.0
        
        return pct

    def get_players_forecast(self, players_full_name: str, filtered_players_logs: pd.DataFrame, team: Team,
                             points_predictor, fga_predictor, fg3a_predictor) -> int:
        """
        Get players' forecast

        :param players_full_name: full name of player
        :param filtered_players_logs: players' game logs
        """
        empty_logs = filtered_players_logs.empty or "MIN" not in filtered_players_logs.columns
        timesteps = self._timesteps()
        if empty_logs:
            logger.warning("No game logs found for %s; returning 0 points", players_full_name)
            return 0

        doesnt_play = filtered_players_logs["MIN"].values[:timesteps].mean() < 10.0
        game_plan = self.locker_room.home_game_plan if team == Team.HOME else self.locker_room.away_game_plan
        todays_mins = game_plan.players_mins[players_full_name]

        if self._model_type() == "GRU":
            min_games = timesteps * 8
        else:
            min_games = 20

        if (empty_logs or doesnt_play) and (todays_mins in [None, 0]):
            return 0

        elif filtered_players_logs.shape[0] < min_games:
            logger.warning(
                "%s has only played %s games; using average points as fallback",
                players_full_name,
                filtered_players_logs.shape[0],
            )
            return int(filtered_players_logs.iloc[:, -1].mean())

        most_recent_game_date = self.locker_room.get_most_recent_game_date(filtered_players_logs)
        training_data = self.prepare_training_data(filtered_players_logs)
        x_test = self.prepare_testing_data(
            players_full_name,
            filtered_players_logs,
            most_recent_game_date,
            team,
            fga_predictor,
            fg3a_predictor,
        )

        forecasted_points = points_predictor.get_forecast(training_data, x_test)

        return forecasted_points

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
        workers = int(os.environ.get("ORACLE_PARALLEL_WORKERS", "2"))

        logger.info("Starting team forecast (team=%s, players=%s, workers=%s)", data.team_name, total_players, workers)

        player_rows = list(data.active_players.iterrows())
        if workers <= 1 or len(player_rows) <= 1:
            for idx, (players_name, players_id) in enumerate(player_rows, start=1):
                _, forecasted_points, actual_points = self._run_player_forecast(team, players_name, players_id)
                forecast_dict = Oracle.append_to_forecast_dict(forecast_dict, players_name, forecasted_points, actual_points)
                logger.info("Team %s progress: %s/%s", data.team_name, idx, total_players)
        else:
            future_map = {}
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for players_name, players_id in player_rows:
                    future = executor.submit(self._run_player_forecast, team, players_name, players_id)
                    future_map[future] = players_name

                finished = 0
                results = {}
                for future in as_completed(future_map):
                    players_name, forecasted_points, actual_points = future.result()
                    results[players_name] = (forecasted_points, actual_points)
                    finished += 1
                    logger.info("Team %s progress: %s/%s", data.team_name, finished, total_players)

            for players_name, _ in player_rows:
                forecasted_points, actual_points = results[players_name]
                forecast_dict = Oracle.append_to_forecast_dict(forecast_dict, players_name, forecasted_points, actual_points)

        forecast_df = self.form_forecast_df(forecast_dict)

        return forecast_df

    def _run_player_forecast(self, team: Team, players_name: str, players_id):
        logger.info("Fetching game logs for player=%s", players_name)
        filtered_players_logs, actual_points = self.locker_room.get_filtered_players_logs(players_id)

        points_predictor, fga_predictor, fg3a_predictor = self._init_predictors()
        logger.info("Starting forecast for player=%s", players_name)
        forecasted_points = self.get_players_forecast(
            players_name,
            filtered_players_logs,
            team,
            points_predictor,
            fga_predictor,
            fg3a_predictor,
        )
        logger.info("Forecast completed for player=%s points=%s", players_name, forecasted_points)
        if self.progress_callback:
            self.progress_callback({"type": "player_complete", "player": players_name})
        return players_name, forecasted_points, actual_points

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
            mins = players_game_log["MIN"].values[:self._timesteps()].mean()
        
        return np.array([(round(mins))])

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
            logger.info("Creating output directory: %s", output_path)
            os.mkdir(output_path)
        
        logger.info("Saving forecasts under %s", output_path)
        with pd.ExcelWriter(f"{output_path}/Forecast.xlsx") as writer:
            home_team_forecast_df.to_excel(writer, sheet_name=f"{self.locker_room.home_team} Forecast", index=False)
            away_team_forecast_df.to_excel(writer, sheet_name=f"{self.locker_room.away_team} Forecast", index=False)
        
        logger.info("Saving model/oracle config snapshots")
        with open(f"{output_path}/oracle_config.json", "w") as json_file:
            json.dump(self.oracle_config, json_file, indent=2)

        with open(f"{output_path}/model_config.json", "w") as json_file:
            json.dump(self.model_config, json_file, indent=2)

    def _load_transformer(self) -> TeamTransformer:
        """Load the pre-trained TeamTransformer model from disk."""
        model_path = os.path.join(_TRANSFORMER_MODEL_DIR, "model.keras")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Pre-trained Transformer model not found at {model_path}. "
                "Run 'python -m scripts.train_transformer' first."
            )
        logger.info("Loading pre-trained TeamTransformer from %s", model_path)
        return TeamTransformer.load(model_path, self.model_config)

    def _get_team_forecast_transformer(self, team: Team) -> pd.DataFrame:
        """Run Transformer-based team-total-points forecast for one side."""
        from data_prep.team_features import (
            build_enriched_team_logs,
            build_inference_inputs,
            compute_roster_aggregates,
        )
        from data_prep.locker_room import collected_seasons
        from pathlib import Path

        if self._transformer is None:
            self._transformer = self._load_transformer()

        if self._enriched_logs is None:
            all_logs = self.locker_room.all_logs
            self._enriched_logs = build_enriched_team_logs(
                collected_seasons, all_logs_df=all_logs, cache_dir=Path("artifacts/cache")
            )

        game_plan = self.locker_room.home_game_plan if team == Team.HOME else self.locker_room.away_game_plan
        opp_plan = self.locker_room.away_game_plan if team == Team.HOME else self.locker_room.home_game_plan
        is_home = team == Team.HOME

        # Get team abbreviation
        from data_prep.gamelogs import nba_teams_info
        team_row = nba_teams_info[nba_teams_info["nickname"] == game_plan.team_name]
        team_abbr = team_row["abbreviation"].values[0] if not team_row.empty else game_plan.team_name[:3].upper()

        # Opponent defensive context
        opp_defense = self.locker_room.get_opponent_defensive_stats(team)
        opp_def_rating = float(opp_defense.get("E_DEF_RATING", 110.0))
        opp_pace = float(opp_defense.get("E_PACE", 100.0))

        # Opponent rolling stats (PPG allowed, FG% allowed, FG3% allowed)
        opp_row = nba_teams_info[nba_teams_info["nickname"] == opp_plan.team_name]
        opp_abbr = opp_row["abbreviation"].values[0] if not opp_row.empty else opp_plan.team_name[:3].upper()
        opp_games = self._enriched_logs[
            (self._enriched_logs["TEAM_ABBREVIATION"] == opp_abbr)
            & (self._enriched_logs["GAME_DATE"] < pd.Timestamp(self.game_date))
        ].sort_values("GAME_DATE").tail(10)

        opp_ppg_allowed = float(opp_games["PTS"].mean()) if not opp_games.empty else 110.0
        opp_fg_pct_allowed = float(opp_games["FG_PCT"].mean()) if not opp_games.empty else 0.46
        opp_fg3_pct_allowed = float(opp_games["FG3_PCT"].mean()) if not opp_games.empty else 0.36

        # Roster aggregates from active players
        player_logs = {}
        if game_plan.active_players is not None:
            for player_name, row in game_plan.active_players.iterrows():
                pid = int(row["PLAYER_ID"]) if "PLAYER_ID" in row.index else int(row.name) if isinstance(row.name, (int, float)) else 0
                if pid == 0:
                    continue
                try:
                    logs_frames = []
                    for season in collected_seasons:
                        try:
                            sl = self.locker_room.fetch_players_game_logs_df(pid, season)
                            if sl is not None and not sl.empty:
                                logs_frames.append(sl)
                        except Exception:
                            pass
                    if logs_frames:
                        player_logs[pid] = pd.concat(logs_frames, ignore_index=True)
                except Exception:
                    logger.warning("Could not fetch logs for player %s (id=%s)", player_name, pid)

        active_ids = list(player_logs.keys())
        roster_aggs = compute_roster_aggregates(player_logs, active_ids, self.game_date)

        seq_len = int(self.model_config.get("seq_len", 10))
        X_seq, X_static = build_inference_inputs(
            enriched_logs=self._enriched_logs,
            team_abbr=team_abbr,
            game_date=self.game_date,
            is_home=is_home,
            opp_def_rating=opp_def_rating,
            opp_pace=opp_pace,
            opp_ppg_allowed=opp_ppg_allowed,
            opp_fg_pct_allowed=opp_fg_pct_allowed,
            opp_fg3_pct_allowed=opp_fg3_pct_allowed,
            roster_aggs=roster_aggs,
            seq_len=seq_len,
        )

        forecasted_pts = self._transformer.get_forecast(X_seq, X_static)

        # Get actual points if holdout
        actual_pts = 0
        if self.holdout:
            try:
                game_date_ts = pd.Timestamp(self.game_date)
                game_row = self._enriched_logs[
                    (self._enriched_logs["TEAM_ABBREVIATION"] == team_abbr)
                    & (self._enriched_logs["GAME_DATE"] == game_date_ts)
                ]
                if not game_row.empty:
                    actual_pts = int(game_row["PTS"].iloc[0])
            except Exception:
                pass

        if self.progress_callback:
            self.progress_callback({"type": "step", "message": f"{game_plan.team_name} forecast complete"})

        forecast_df = pd.DataFrame({
            "PLAYER_NAME": [f"{game_plan.team_name} (Transformer)", "Total"],
            "FORECASTED_POINTS": [forecasted_pts, forecasted_pts],
            "ACTUAL_POINTS": [actual_pts, actual_pts],
        })
        logger.info(
            "Transformer forecast: team=%s, predicted=%d, actual=%d",
            game_plan.team_name, forecasted_pts, actual_pts,
        )
        return forecast_df

    def run(self):
        """
        Run Oracle
        """
        logger.info("Running Oracle forecast pipeline")

        if self.oracle_config["model"].upper() == "TRANSFORMER":
            if self.progress_callback:
                self.progress_callback({"type": "step", "message": "Loading Transformer model", "progress": 10})
            home_team_forecast_df = self._get_team_forecast_transformer(Team.HOME)
            if self.progress_callback:
                self.progress_callback({"type": "step", "message": "Home team forecast complete", "progress": 55})
            away_team_forecast_df = self._get_team_forecast_transformer(Team.AWAY)
            if self.progress_callback:
                self.progress_callback({"type": "step", "message": "Away team forecast complete", "progress": 95})
        else:
            home_team_forecast_df = self.get_team_forecast(Team.HOME)
            away_team_forecast_df = self.get_team_forecast(Team.AWAY)

        logger.info(
            "Team forecasts completed (home_total=%s, away_total=%s)",
            home_team_forecast_df["FORECASTED_POINTS"].iloc[-1],
            away_team_forecast_df["FORECASTED_POINTS"].iloc[-1],
        )

        if self.save_output:
            logger.info("Persisting forecast output files")
            self.save_forecasts(home_team_forecast_df, away_team_forecast_df)

        return home_team_forecast_df, away_team_forecast_df
