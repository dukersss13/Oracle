import os
import time
import logging
from datetime import datetime

import json
import pandas as pd
import numpy as np
from enum import Enum
from tensorflow import one_hot
from dataclasses import dataclass

from data_prep.gamelogs import update_data, consolidate_all_game_logs, nba_teams_info
from data_prep.injury_report import injury_report
from data_prep.db import OracleCacheDB
from nba_api.stats.endpoints import playergamelog, commonteamroster
from nba_api.stats.static import  players


logger = logging.getLogger(__name__)

pd.set_option('mode.chained_assignment', None)
pd.set_option('display.max_columns', None)


def _current_nba_season(current_dt: datetime | None = None) -> str:
    if current_dt is None:
        current_dt = datetime.now()
    season_start_year = current_dt.year if current_dt.month >= 10 else current_dt.year - 1
    return f"{season_start_year}-{(season_start_year + 1) % 100:02d}"


def _recent_nba_seasons(n: int = 3, current_dt: datetime | None = None) -> list[str]:
    if current_dt is None:
        current_dt = datetime.now()
    current = _current_nba_season(current_dt)
    start_year = int(current.split("-")[0])
    return [f"{y}-{(y + 1) % 100:02d}" for y in range(start_year, start_year - n, -1)]


collected_seasons = _recent_nba_seasons(3)
current_season = [collected_seasons[0]]

class Team(Enum):
    HOME = 0
    AWAY = 1


@dataclass
class GamePlan:
    def __init__(self, team_name: str):
        self.team_name: str = team_name
        self.team_id: int = None
        self.team_game_logs: pd.DataFrame = None
        self.team_roster: pd.DataFrame = None
        self.active_players: pd.DataFrame = None
        self.players_mins: dict = None


class LockerRoom:
    def __init__(self, game_details: dict, features: list,
                 fetch_new_data: bool, holdout: bool,
                 active_players_override: dict | None = None,
                 interactive: bool = True,
                 force_refresh_cache: bool = False,
                 cache_strategy: str = "incremental",
                 cache_ttl_hours: int = 12):
        """
        Initialize the Locker Room

        In sports, the locker room is where both teams get ready for the game.
        Similarly here, the LockerRoom class prepares the data for both teams
        needed for forecasting.

        :param game_details: details of the game to be forecasted
        :param nn_config: _description_, defaults to None
        :param season: _description_, defaults to "2022-23"
        """
        self.overwrite = game_details["new_game"]
        self.holdout = holdout

        self.home_team = game_details["home_team"]
        self.away_team = game_details["away_team"]
        self.game_date = game_details["game_date"]

        self.predictors_plus_label = features
        self.nba_teams_info = nba_teams_info
        self.interactive = interactive
        self.active_players_override = active_players_override
        self.force_refresh_cache = force_refresh_cache
        self.cache_strategy = cache_strategy.lower()
        self.cache_ttl_hours = int(cache_ttl_hours)
        self.db_cache = OracleCacheDB()
        logger.info(
            "LockerRoom initialized (home=%s, away=%s, holdout=%s, cache_strategy=%s)",
            self.home_team,
            self.away_team,
            self.holdout,
            self.cache_strategy,
        )

        self._fetch_teams_data(fetch_new_data)

    def _fetch_teams_data(self, fetch_new_data: bool):
        """
        Fetch the data needed for each team & create/update active players json
        """
        home_lookup_values = ["nickname", self.home_team]
        away_lookup_values = ["nickname", self.away_team]

        self.home_game_plan = GamePlan(self.home_team)
        self.away_game_plan = GamePlan(self.away_team)

        self.home_game_plan.team_roster = self.fetch_roster(home_lookup_values)
        self.away_game_plan.team_roster = self.fetch_roster(away_lookup_values)

        self.home_game_plan.team_id = self.fetch_teams_id(home_lookup_values)
        self.away_game_plan.team_id = self.fetch_teams_id(away_lookup_values)

        self.home_away_dict = {Team.HOME: self.home_team, Team.AWAY: self.away_team}

        self._set_game_plan()
        self._fetch_all_logs(fetch_new_data)

    def _set_game_plan(self):
        """
        Set the game plan such as active players & matchups
        """
        self._update_game_plan()
        if self.active_players_override:
            self.set_active_players_from_dict(self.active_players_override)
            return

        if not self.interactive:
            self.set_active_players()
            return

        set_active_players = LockerRoom._pause_for_configurations()

        if set_active_players == 1:
            self.set_active_players()
        else:
            raise ValueError("Aborting program!")         
    
    def _fetch_all_logs(self, fetch_new_data: bool):
        """
        Grab the logs for all games
        """
        dataset_key = f"all_logs_{'_'.join(collected_seasons)}"

        if fetch_new_data:
            logger.info("Refreshing all_logs dataset from NBA APIs")
            update_data(current_season)
            consolidate_all_game_logs(collected_seasons, current_season)
            csv_logs = pd.read_csv("data/all_logs.csv", low_memory=False)
            csv_logs = csv_logs[csv_logs["SEASON_YEAR"].isin(collected_seasons)]
            self.db_cache.upsert_dataset_df(dataset_key, csv_logs)
            self.all_logs = csv_logs
            return

        db_logs = self.db_cache.get_dataset_df(dataset_key, ttl_hours=None)
        if db_logs is not None and not db_logs.empty:
            logger.info("Loaded all_logs dataset from cache (rows=%s)", len(db_logs))
            self.all_logs = db_logs
            return

        # Bootstrap DB from disk once, then continue DB-first.
        csv_logs = pd.read_csv("data/all_logs.csv", low_memory=False)
        csv_logs = csv_logs[csv_logs["SEASON_YEAR"].isin(collected_seasons)]
        self.db_cache.upsert_dataset_df(dataset_key, csv_logs)
        self.all_logs = csv_logs
        logger.info("Bootstrapped all_logs dataset from disk (rows=%s)", len(csv_logs))

    @staticmethod
    def _pause_for_configurations() -> int:
        """
        Pauses the program so user can set the lineups
        """
        print("\nSet active players in active_players.json.")
        print("Input 0 for injured/DNP. Else, leave as null")

        good_to_go = int(input(("Enter 1 to continue: ")))

        return good_to_go

    @staticmethod
    def _init_months_dict() -> dict:
        """
        Create months dictionary
        """
        months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
        months_dict = dict(zip(months, range(1, 13)))

        return months_dict

    def fetch_roster(self, team_lookup_tuple: list) -> pd.DataFrame:
        """
        Fetch the roster of given team

        :param team_name: name of NBA team
        :return: df of team roster & players' IDs
        """
        if " " in team_lookup_tuple[1]:
            team_lookup_tuple[1] = team_lookup_tuple[1].title()
        else:
            team_lookup_tuple[1] = team_lookup_tuple[1].capitalize()

        team_id = int(self.fetch_teams_id(team_lookup_tuple))
        season = _current_nba_season()
        season_start_year = int(season.split("-")[0])
        prev_season = f"{season_start_year - 1}-{season_start_year % 100:02d}"
        seasons_to_try = [season, prev_season]

        # Check fresh cache first (any season)
        for try_season in seasons_to_try:
            db_roster = self.db_cache.get_roster(team_id, try_season, ttl_hours=self.cache_ttl_hours)
            if db_roster is not None and not db_roster.empty:
                if self.cache_strategy == "cache-only" or not self.force_refresh_cache:
                    return db_roster

        if self.cache_strategy == "cache-only":
            return pd.DataFrame(columns=["PLAYER", "PLAYER_ID"])

        retries = int(os.environ.get("ORACLE_ROSTER_RETRIES", "3"))
        backoff_seconds = float(os.environ.get("ORACLE_ROSTER_BACKOFF", "1.5"))
        request_timeout = int(os.environ.get("ORACLE_NBA_API_TIMEOUT", "20"))

        # Try fetching from API: current season first, then previous
        for try_season in seasons_to_try:
            for attempt in range(1, retries + 1):
                try:
                    team_roster = commonteamroster.CommonTeamRoster(
                        team_id=team_id,
                        season=try_season,
                        timeout=request_timeout,
                    ).get_data_frames()[0][["PLAYER", "PLAYER_ID"]]
                    self.db_cache.upsert_roster(team_id, try_season, team_roster)
                    if try_season != season:
                        logger.info("Roster fetched from fallback season %s for team_id=%s", try_season, team_id)
                    return team_roster
                except Exception as exc:
                    if attempt < retries:
                        logger.warning(
                            "Roster fetch failed for team_id=%s season=%s (attempt %s/%s); retrying",
                            team_id, try_season, attempt, retries,
                        )
                        time.sleep(backoff_seconds * attempt)

        # If all live fetches fail, fall back to stale DB cache (any season)
        for try_season in seasons_to_try:
            stale_db_roster = self.db_cache.get_roster(team_id, try_season, ttl_hours=None)
            if stale_db_roster is not None and not stale_db_roster.empty:
                logger.warning("Using stale cached roster (season=%s) for team_id=%s after API failures", try_season, team_id)
                return stale_db_roster

        logger.warning(
            "Unable to fetch roster for team_id=%s; returning empty roster",
            team_id,
        )
        return pd.DataFrame(columns=["PLAYER", "PLAYER_ID"])

    def get_most_recent_game_date(self, players_game_logs_df: pd.DataFrame) -> pd.Timestamp:
        """
        Get the date of the most recent game by given player

        :param players_game_logs_df: player's game logs df
        :return: the date of their most recent game
        """
        most_recent_game_date = players_game_logs_df["GAME_DATE_player"].values[0]

        return most_recent_game_date

    def set_active_players(self):
        """
        Set active players & allocate their minutes if need be
        """
        with open(self.active_players_path) as f:
            active_players_json = json.load(f)

        for team in active_players_json:
            team_data = self.home_game_plan if team == self.home_team else self.away_game_plan
            full_team_name = self.nba_teams_info[self.nba_teams_info["nickname"]==team_data.team_name]["full_name"].values[0]
            team_injury_report = injury_report[injury_report["team"]==full_team_name]
            active_players_df = pd.DataFrame(active_players_json[team], index=["Mins"]).T
            injured_players = team_injury_report["name"].values
            active_players = active_players_df[(~np.isin(active_players_df.index.values, injured_players) & \
                                                          (active_players_df["Mins"]!=0).values)]
            team_data.active_players = team_data.team_roster[np.isin(team_data.team_roster["PLAYER"], active_players.index)].set_index("PLAYER")
            team_data.players_mins = active_players.to_dict()["Mins"]

    def set_active_players_from_dict(self, active_players_json: dict):
        """Set active players from an in-memory dict, used by API workflows."""
        for team in active_players_json:
            if team not in [self.home_team, self.away_team]:
                continue

            team_data = self.home_game_plan if team == self.home_team else self.away_game_plan
            active_players_df = pd.DataFrame(active_players_json[team], index=["Mins"]).T
            active_players = active_players_df[active_players_df["Mins"] != 0]
            team_data.active_players = team_data.team_roster[
                np.isin(team_data.team_roster["PLAYER"], active_players.index)
            ].set_index("PLAYER")
            team_data.players_mins = active_players.to_dict()["Mins"]

    def _update_game_plan(self):
        """
        Update the active players json to set active players or manually assign minutes

        :param json_type: whether it's active players or matchups json
        """
        self.active_players_path = f"{os.getcwd()}/artifacts/active_players.json"

        path = self.active_players_path

        self._check_requisite_jsons(path)
        with open(path) as f:
            prereq_json = json.load(f)

        if self.overwrite:
            # Refreshes the file & overwrite
            for team_name in prereq_json:
                del team_name
            prereq_json = self._init_rerequisite_jsons()
            with open(path, 'w') as f:
                json.dump(prereq_json, f, indent=1)

    def _check_requisite_jsons(self, json_path: str):
        """
        Check if active players/matchus json exists. If not, create one.
        """
        if not os.path.exists(json_path):
            prereq_json = self._init_rerequisite_jsons()
            with open(json_path, 'w') as f:
                json.dump(prereq_json, f, indent=1)

    def _init_rerequisite_jsons(self):
        """
        Initialize the active players json
        """
        home_roster = self.home_game_plan.team_roster
        away_roster = self.away_game_plan.team_roster

        json =  {self.home_team: dict(zip(home_roster["PLAYER"].values, [None for _ in range(len(home_roster))])),
                 self.away_team: dict(zip(away_roster["PLAYER"].values, [None for _ in range(len(away_roster))]))}
        
        return json

    def fetch_players_game_logs_df(self, players_id: str, season: str) -> pd.DataFrame:
        """
        Access the PlayerGameLog module to fetch the game logs df of given player

        :param players_id: player ID
        :return: the given player's game logs in df format
        """
        normalized_player_id = self._normalize_player_id(players_id)
        db_cached_df = self.db_cache.get_player_logs(normalized_player_id, season, ttl_hours=self.cache_ttl_hours)
        stale_db_cached_df = self.db_cache.get_player_logs(normalized_player_id, season, ttl_hours=None)

        if db_cached_df is not None and not db_cached_df.empty:
            cached_df = db_cached_df
        else:
            cached_df = stale_db_cached_df if stale_db_cached_df is not None else pd.DataFrame()

        if self.cache_strategy == "cache-only":
            return cached_df

        # For past games, start inference from DB-cached training data.
        if self.holdout and not cached_df.empty and not self.force_refresh_cache:
            return cached_df

        if self.cache_strategy == "full" or self.force_refresh_cache:
            fresh_df = self._fetch_player_gamelog_with_retry(normalized_player_id, season)
            self.db_cache.upsert_player_logs(normalized_player_id, season, fresh_df)
            return fresh_df

        if self.cache_strategy == "incremental" and not cached_df.empty and db_cached_df is not None:
            return cached_df

        # Incremental mode: try pulling latest data from nba_api and merge with cache.
        try:
            fresh_df = self._fetch_player_gamelog_with_retry(normalized_player_id, season)
            merged_df = self._merge_game_logs(cached_df, fresh_df)
            self.db_cache.upsert_player_logs(normalized_player_id, season, merged_df)
            return merged_df
        except Exception:
            if not cached_df.empty:
                logger.warning(
                    "Using cached game logs for player_id=%s season=%s after API failure",
                    normalized_player_id,
                    season,
                )
                return cached_df
            raise

    @staticmethod
    def _merge_game_logs(cached_df: pd.DataFrame, fresh_df: pd.DataFrame) -> pd.DataFrame:
        if cached_df.empty:
            return fresh_df
        if fresh_df.empty:
            return cached_df

        combined = pd.concat([fresh_df, cached_df], axis=0, ignore_index=True)
        game_id_col = "Game_ID" if "Game_ID" in combined.columns else "GAME_ID" if "GAME_ID" in combined.columns else None

        if game_id_col:
            combined = combined.drop_duplicates(subset=[game_id_col], keep="first")
        else:
            combined = combined.drop_duplicates(keep="first")

        if "GAME_DATE" in combined.columns:
            # Preserve latest-first ordering chronologically even with string dates.
            combined["_GAME_DATE_SORT"] = pd.to_datetime(combined["GAME_DATE"], errors="coerce")
            combined = combined.sort_values(by="_GAME_DATE_SORT", ascending=False)
            combined = combined.drop(columns=["_GAME_DATE_SORT"])

        return combined.reset_index(drop=True)

    @staticmethod
    def _fetch_player_gamelog_with_retry(player_id: int, season: str) -> pd.DataFrame:
        retries = int(os.environ.get("ORACLE_PLAYERLOG_RETRIES", "3"))
        backoff_seconds = float(os.environ.get("ORACLE_PLAYERLOG_BACKOFF", "1.0"))
        request_timeout = int(os.environ.get("ORACLE_NBA_API_TIMEOUT", "20"))
        last_error = None

        for attempt in range(1, retries + 1):
            try:
                return playergamelog.PlayerGameLog(
                    player_id=player_id,
                    season=season,
                    season_type_all_star="Regular Season",
                    timeout=request_timeout,
                ).get_data_frames()[0]
            except Exception as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(backoff_seconds * attempt)

        raise RuntimeError(
            f"Failed to fetch game logs for player_id={player_id}, season={season} after {retries} attempts"
        ) from last_error

    @staticmethod
    def _normalize_player_id(players_id: str | int) -> int:
        if hasattr(players_id, "values"):
            return int(players_id.values[0])
        return int(players_id)

    def get_filtered_players_logs(self, players_id: int) -> tuple[pd.DataFrame, int]:
        """
        Retrieve the filtered game logs for given player

        :param players_full_name: full name of the player
        :return filtered_log.values: an array of player's game logs filtered by specific columns
        """
        all_logs = []
        for season in collected_seasons:
            try:
                players_game_logs_df = self.fetch_players_game_logs_df(players_id, season)
                if not players_game_logs_df.empty:
                    all_logs.append(players_game_logs_df)
            except Exception:
                logger.exception(
                    "Player logs could not be fetched (player_id=%s, season=%s)",
                    self._normalize_player_id(players_id),
                    season,
                )

        if not all_logs:
            return pd.DataFrame(), 0

        all_logs = pd.concat(all_logs, axis=0)
        actual_points = 0   
        if not all_logs.empty:
            all_logs: pd.DataFrame = self._add_predictors_to_players_log(all_logs)
            if self.holdout:
                try:
                    actual_points = all_logs[all_logs["GAME_DATE_player"] == pd.Timestamp(self.game_date)]["PTS"].values[0]
                except IndexError:
                    actual_points = 0
            all_logs = all_logs[all_logs["GAME_DATE_player"] < self.game_date]

        return all_logs, actual_points
    
    def get_opponent_defensive_stats(self, team: Team) -> pd.Series:
        """
        Retrieve the opponent's defensive stats
        """
        cols = ["D_FGM", "D_FGA", "D_FG_PCT",
                "FG3M", "FG3A", "FG3_PCT", "NS_FG3_PCT",
                "FG2M", "FG2A", "FG2_PCT", "NS_FG2_PCT",
                "FGM_LT_10", "FGA_LT_10", "LT_10_PCT", "NS_LT_10_PCT",
                "E_PACE", "E_DEF_RATING"]
        opponent_name = self.home_away_dict[Team.AWAY if team == Team.HOME else Team.HOME]
        opponent_id = self.fetch_teams_id(("nickname", opponent_name))
        season_team_slice = self.all_logs[
            (self.all_logs["SEASON_YEAR"] == current_season[0]) &
            (self.all_logs["TEAM_ID"] == opponent_id)
        ]

        if season_team_slice.empty:
            season_team_slice = self.all_logs[self.all_logs["TEAM_ID"] == opponent_id]

        if not season_team_slice.empty:
            return season_team_slice[cols].iloc[0, :]

        # Final fallback: return dataset-level means to keep forecasts running.
        if self.all_logs.empty:
            logger.warning("all_logs dataset is empty; using zero-vector defensive fallback")
            return pd.Series(np.zeros(len(cols)), index=cols)

        return self.all_logs[cols].mean(numeric_only=True).reindex(cols, fill_value=0.0)

    @staticmethod
    def prepare_training_data(players_game_log: pd.DataFrame, input_cols: list[str],
                              label_col: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract specified input and output columns for training
        """
        input_cols = players_game_log[input_cols].values.astype(np.float32)
        label_col = players_game_log[label_col].values.astype(np.float32)
        
        return input_cols, label_col 

    def _add_predictors_to_players_log(self, players_game_logs_df: pd.DataFrame) -> pd.DataFrame:
        """_summary_

        :param players_game_log: _description_
        :return: _description_
        """
        players_log = self._add_rest_days_and_opp_id(players_game_logs_df)
        players_log = LockerRoom._add_home_away_columns(players_log)
        complete_log = self._merge_defensive_stats_to_players_log(players_log)
        filtered_log = LockerRoom.filter_stats(complete_log, self.predictors_plus_label)

        return filtered_log

    def _add_rest_days_and_opp_id(self, players_game_logs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add rest days column

        :param players_game_logs_df: player's game logs df
        :return: player's game logs df w/ rest days
        """
        players_game_logs_df["GAME_DATE"] = players_game_logs_df["GAME_DATE"].apply(LockerRoom.convert_to_timestamp)

        # Drop rows with unparseable dates instead of failing the whole forecast run.
        players_game_logs_df = players_game_logs_df.dropna(subset=["GAME_DATE"])

        players_game_logs_df = players_game_logs_df[players_game_logs_df["GAME_DATE"] <= self.game_date]
        players_game_logs_df["REST_DAYS"] = players_game_logs_df["GAME_DATE"].diff(periods=-1)
        players_game_logs_df = players_game_logs_df.iloc[:-1, :]
        players_game_logs_df["REST_DAYS"] = players_game_logs_df["REST_DAYS"].dt.days
        players_game_logs_df.loc[players_game_logs_df["REST_DAYS"] > 7, "REST_DAYS"] = 1
        players_game_logs_df["TEAM_ID"] = players_game_logs_df["MATCHUP"].apply(self.get_opp_id)

        return players_game_logs_df

    def _merge_defensive_stats_to_players_log(self, players_game_log: pd.DataFrame) -> pd.DataFrame:
        """
        Merge the opposing defense stats
        to the player's log
        """
        self.all_logs.dropna(inplace=True)
        players_game_log = players_game_log.rename(columns={"Game_ID": "GAME_ID"})
        players_game_log["GAME_ID"] = players_game_log["GAME_ID"].astype(int)
        log_with_defensive_stats = players_game_log.merge(self.all_logs, how="left", 
                                                          on=["TEAM_ID"],
                                                          suffixes=["_player",
                                                                    "_opp_defense"]).drop_duplicates(subset=["GAME_ID_player"])

        return log_with_defensive_stats

    @staticmethod
    def _add_home_away_columns(players_game_logs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add one_hot encoding home or away bool columns
        
        :param players_game_logs_df: player's game logs df
        :return: player's game logs df w/ home & away columns
        """
        players_game_logs_df.loc[:, ("HOME", "AWAY")] = one_hot(players_game_logs_df["MATCHUP"].\
                                                        apply(LockerRoom.detect_home_or_away_games), 2)

        return players_game_logs_df

    @staticmethod
    def detect_home_or_away_games(game: str):
        """
        Detect if a game is on home court or away
        """
        return 1 if "@" in game else 0

    @staticmethod
    def convert_to_timestamp(date_string: str) -> pd.Timestamp:
        """
        Convert a date in string format to pd.Timestamp format
        """
        # nba_api date values may be strings ("Apr 3, 2024") or tokenized lists.
        if isinstance(date_string, list):
            raw_date = " ".join(str(token) for token in date_string)
        else:
            raw_date = str(date_string)

        parsed = pd.to_datetime(raw_date, errors="coerce")
        return pd.Timestamp(parsed) if not pd.isna(parsed) else pd.NaT

    def get_opp_id(self, matchup: str):
        """
        Fetch opponent's ID when looking at the matchup
        """
        matchup_split = matchup.split(" ")
        opp_abb = matchup_split[2]
        team_id = self.nba_teams_info[self.nba_teams_info["abbreviation"]==opp_abb]["id"].values[0]
        
        return team_id

    @staticmethod
    def filter_stats(game_logs_df: pd.DataFrame, columns_wanted: list) -> pd.DataFrame:
        """
        Filter the game logs df with just wanted columns

        :param game_logs_df: game logs df of given player
        :param columns_wanted: columns wanted

        :return: game logs df with just columns wanted
        """
        if columns_wanted is None:
            return game_logs_df

        return game_logs_df[columns_wanted]

    @staticmethod
    def fetch_players_id(players_full_name: str) -> int:
        """
        Get players ID given full name

        :param: players_full_name: player's full name
        :return: player's ID
        """
        try:
            players_id = players.find_players_by_full_name(players_full_name)[0]["id"]
        except IndexError:
            logger.warning("Player does not have an NBA API player ID: %s", players_full_name)
            players_id = None
        
        return players_id

    def fetch_teams_id(self, lookup_values: list) -> int:
        """
        Fetch the team's ID

        :param lookup_values: name_type + name of the team
        :return: team ID
        """
        try:
            name_type, name = lookup_values
            teams_id = self.nba_teams_info[self.nba_teams_info[name_type]==name]["id"]
        except Exception:
            logger.exception("Team lookup failed for values=%s", lookup_values)
            teams_id = None
        
        return teams_id.values[0]

    def fetch_team_game_logs(self, team_name: str) -> pd.DataFrame:
        """
        Fetch all the game logs for given team

        :param team_name: name of the team (i.e. Mavericks, Lakers)
        :return: the team's game logs
        """
        col, name = team_name
        team_abbreviation = self.nba_teams_info[self.nba_teams_info[col]==name]["abbreviation"].values[0]
        path = "data/seasonal_data"
        game_logs_by_year = []

        for season in collected_seasons:
            season_year = f"20{season[-2:]}"
            game_log = pd.read_csv(f"{path}/{season_year}/team_logs/{team_abbreviation}.csv", index_col=0)
            game_logs_by_year.append(game_log)

        # Filter out the wanted columns
        return pd.concat(game_logs_by_year, axis=0)
