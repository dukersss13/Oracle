import os

import json
import pandas as pd
import numpy as np
from enum import Enum
from tensorflow import one_hot
from dataclasses import dataclass

from data_prep.gamelogs import update_data, consolidate_all_game_logs, nba_teams_info
from nba_api.stats.endpoints import (playergamelog, boxscoretraditionalv2,
                                     commonteamroster)
from nba_api.stats.static import  players

pd.set_option('mode.chained_assignment', None)
pd.set_option('display.max_columns', None)


current_season = ["2023-24"]
collected_seasons = ["2023-24", "2022-23", "2021-22", "2020-21"]

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
                 fetch_new_data: bool, holdout: bool):
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
        set_active_players = LockerRoom._pause_for_configurations()

        if set_active_players == 1:
            self.set_active_players()
        else:
            raise ValueError("Aborting program!")         
    
    def _fetch_all_logs(self, fetch_new_data: bool):
        """_summary_

        :return: _description_
        """
        if fetch_new_data:
            update_data(current_season)
            consolidate_all_game_logs(collected_seasons, current_season)

        self.all_logs = pd.read_csv("data/all_logs.csv", index_col=0, low_memory=False)

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

        team_id = self.fetch_teams_id(team_lookup_tuple)
        team_roster = commonteamroster.CommonTeamRoster(team_id=team_id,
                                                        season=current_season).get_data_frames()[0][["PLAYER", "PLAYER_ID"]]
        
        return team_roster

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
            active_players_df = pd.DataFrame(active_players_json[team], index=["Mins"]).T
            active_players = active_players_df[active_players_df["Mins"] != 0]
            team_data.active_players = team_data.team_roster[np.isin(team_data.team_roster["PLAYER"],
                                                             active_players.index)].set_index("PLAYER")
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
        players_game_log = playergamelog.PlayerGameLog(player_id=players_id, season=season,
                                                       season_type_all_star="Regular Season").get_data_frames()[0]

        return players_game_log

    def get_filtered_players_logs(self, players_id: int) -> tuple[pd.DataFrame, int]:
        """
        Retrieve the filtered game logs for given player

        :param players_full_name: full name of the player
        :return filtered_log.values: an array of player's game logs filtered by specific columns
        """
        all_logs = pd.DataFrame([])
        for season in collected_seasons:
            try:
                players_game_logs_df = self.fetch_players_game_logs_df(players_id, season)
                all_logs = pd.concat([all_logs, players_game_logs_df])
            except:
                print(f"Logs for playerID: {players_id.values[0]} for {season} cannot be fetched.")

        if not all_logs.empty:
            all_logs = self._add_predictors_to_players_log(all_logs)
            if self.holdout:
                actual_points = all_logs[all_logs["GAME_DATE"]==self.game_date]
            else:
                actual_points = 0

        return all_logs, actual_points
    
    def get_opponent_defensive_stats(self, team: Team) -> pd.Series:
        """_summary_

        :param team: _description_
        :return: _description_
        """
        cols = ["D_FGM", "D_FGA", "D_FG_PCT",
                "FG3M", "FG3A", "FG3_PCT", "NS_FG3_PCT",
                "FG2M", "FG2A", "FG2_PCT", "NS_FG2_PCT",
                "FGM_LT_10", "FGA_LT_10", "LT_10_PCT", "NS_LT_10_PCT",
                "E_PACE", "E_DEF_RATING"]
        opponent_name = self.home_away_dict[Team.AWAY if team == Team.HOME else Team.HOME]
        opponent_id = self.fetch_teams_id(("nickname", opponent_name))
        opponent_defensive_stats = self.all_logs[(self.all_logs["SEASON_YEAR"]==current_season[0]) & \
                                                 (self.all_logs["TEAM_ID"]==opponent_id)][cols].iloc[0, :]

        return opponent_defensive_stats

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
        players_game_logs_df["GAME_DATE"] = players_game_logs_df["GAME_DATE"].apply(lambda x: x.split(" "))
        players_game_logs_df["GAME_DATE"] = players_game_logs_df["GAME_DATE"].apply(LockerRoom.convert_to_timestamp)

        players_game_logs_df = players_game_logs_df[players_game_logs_df["GAME_DATE"] <= self.game_date]
        players_game_logs_df["REST_DAYS"] = players_game_logs_df["GAME_DATE"].diff(periods=-1)
        players_game_logs_df = players_game_logs_df.iloc[:-1, :]
        players_game_logs_df.loc[:, "REST_DAYS"] = players_game_logs_df["REST_DAYS"].dt.days
        players_game_logs_df["TEAM_ID"] = players_game_logs_df["MATCHUP"].apply(self.get_opp_id)

        return players_game_logs_df

    def _merge_defensive_stats_to_players_log(self, players_game_log: pd.DataFrame) -> pd.DataFrame:
        """
        Merge the opposing defense stats
        to the player's log
        """
        players_game_log = players_game_log.rename(columns={"Game_ID": "GAME_ID"})
        players_game_log["GAME_ID"] = players_game_log["GAME_ID"].astype(np.int64)
        log_with_defensive_stats = players_game_log.merge(self.all_logs, how="left", 
                                                          on=["GAME_ID", "TEAM_ID"],
                                                          suffixes=["_player", "_opp_defense"])

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
        months_dict = LockerRoom._init_months_dict()
        date = pd.Timestamp(f"{date_string[2]}-{months_dict[date_string[0]]}-{date_string[1][:-1]}")

        return date

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
            print(f"WARNING: {players_full_name} does not have a player ID!")
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
        except:
            print(f"WARNING: {lookup_values}'s ID cannot be found!")
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
