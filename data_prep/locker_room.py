import os

from typing import Tuple
import json
import pandas as pd
import numpy as np
from enum import Enum

from nba_api.stats.endpoints import (playergamelog, leagueseasonmatchups, boxscoretraditionalv2,
                                     commonteamroster, teamgamelogs, leaguedashptteamdefend)
from nba_api.stats.library.parameters import SeasonAll
from nba_api.stats.static import  players
from tensorflow import one_hot
from dataclasses import dataclass

pd.set_option('mode.chained_assignment', None)
pd.set_option('display.max_columns', None)


class Team(Enum):
    HOME = 0
    AWAY = 1

class JsonType(Enum):
    ACTIVE_PLAYERS = 1

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
    def __init__(self, game_details: dict, nn_config: dict, season="2022-23"):
        """
        Initialize the Locker Room

        In sports, the locker room is where both teams get ready for the game.
        Similarly here, the LockerRoom class prepares the data for both teams
        needed for forecasting.

        :param game_details: details of the game to be forecasted
        :param nn_config: _description_, defaults to None
        :param season: _description_, defaults to "2022-23"
        """
        self.seasons = ["2022-23", "2021-22", "2020-21"]
        self.current_season = season

        self.home_team = game_details["home_team"]
        self.away_team = game_details["away_team"]
        self.game_date = game_details["game_date"]

        self.nn_config = nn_config
        self.predictors = nn_config["predictors"]
        self.num_seasons = nn_config["num_seasons"]
        self.nba_teams_info = pd.read_csv("data/static_data/static_team_info.csv")

        self.fetch_teams_data()

    def fetch_teams_data(self):
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

        self.home_away_dict = {self.home_team: Team.HOME, self.away_team: Team.AWAY}

        self.set_game_plan()

        if self.nn_config["holdout"]:
            self.home_game_plan.team_game_logs = self.fetch_team_game_logs(home_lookup_values)

    def set_game_plan(self):
        """
        Set the game plan such as active players & matchups
        """
        self.update_game_plan()
        set_active_players = LockerRoom.pause_for_configurations()
        if set_active_players:
            self.set_active_players()
    
    @staticmethod
    def pause_for_configurations():
        """
        Pauses the program so user can set the lineups
        """
        print("\nSet active players in active_players.json.")
        print("Input 0 for injured/DNP. Else, leave as null")

        good_to_go = input(("Enter 1 to continue: "))

        return good_to_go

    @staticmethod
    def init_months_dict() -> dict:
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
                                                        season=self.current_season).get_data_frames()[0][["PLAYER", "PLAYER_ID"]]
        
        return team_roster

    def get_most_recent_game_date(self, players_game_logs_df: pd.DataFrame) -> pd.Timestamp:
        """
        Get the date of the most recent game by given player

        :param players_game_logs_df: player's game logs df
        :return: the date of their most recent game
        """
        if self.nn_config["holdout"]:
            most_recent_game_date = players_game_logs_df[players_game_logs_df["GAME_DATE"] < self.game_date]["GAME_DATE"].values[0]
        else:
            most_recent_game_date = players_game_logs_df["GAME_DATE"][0]

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

    def update_game_plan(self):
        """
        Update the active players json to set active players or manually assign minutes

        :param json_type: whether it's active players or matchups json
        """
        self.active_players_path = f"{os.getcwd()}/artifacts/active_players.json"

        path = self.active_players_path

        self.check_requisite_jsons(path)
        with open(path) as f:
            prereq_json = json.load(f)

        # Refreshes the file & overwrite
        for team_name in prereq_json:
            del team_name

        prereq_json = self.init_rerequisite_jsons()
        with open(path, 'w') as f:
            json.dump(prereq_json, f, indent=1)

    def check_requisite_jsons(self, json_path: str):
        """
        Check if active players/matchus json exists. If not, create one.
        """
        if not os.path.exists(json_path):
            prereq_json = self.init_rerequisite_jsons()
            with open(json_path, 'w') as f:
                json.dump(prereq_json, f, indent=1)

    def init_rerequisite_jsons(self):
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

    def get_filtered_players_logs(self, players_id: int) -> Tuple[np.ndarray, pd.Timestamp]:
        """
        Retrieve the filtered game logs for given player

        :param players_full_name: full name of the player
        :return filtered_log.values: an array of player's game logs filtered by specific columns
        """
        # TODO iterate through seasons HERE
        all_logs = pd.DataFrame([])
        for season in self.seasons:
            try:
                players_game_logs_df = self.fetch_players_game_logs_df(players_id, season)
                filtered_logs = self.add_predictors_to_players_log(players_game_logs_df, season)
                all_logs = pd.concat([all_logs, filtered_logs])
            except:
                print(f"Logs for playerID: {players_id} for {season} cannot be fetched.")

        return all_logs

    def add_predictors_to_players_log(self, players_game_logs_df: pd.DataFrame, season: str) -> pd.DataFrame:
        """_summary_

        :param players_game_log: _description_
        :return: _description_
        """
        players_log = self.add_rest_days_and_opp_id(players_game_logs_df)
        #most_recent_game_date = self.get_most_recent_game_date(players_game_logs_df)
        players_log = LockerRoom.add_home_away_columns(players_log)
        complete_log = self.merge_defensive_stats_to_players_log(players_log, season)
        filtered_log = LockerRoom.filter_stats(complete_log, self.predictors)

        return filtered_log

    def add_rest_days_and_opp_id(self, players_game_logs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add rest days column

        :param players_game_logs_df: player's game logs df
        :return: player's game logs df w/ rest days
        """
        players_game_logs_df["GAME_DATE"] = players_game_logs_df["GAME_DATE"].apply(lambda x: x.split(" "))
        players_game_logs_df["GAME_DATE"] = players_game_logs_df["GAME_DATE"].apply(LockerRoom.convert_to_timestamp)
        players_game_logs_df["REST_DAYS"] = players_game_logs_df["GAME_DATE"].diff(periods=-1)
        players_game_logs_df = players_game_logs_df.iloc[:-1, :]
        players_game_logs_df.loc[:, "REST_DAYS"] = players_game_logs_df["REST_DAYS"].dt.days
        players_game_logs_df["TEAM_ID"] = players_game_logs_df["MATCHUP"].apply(self.get_opp_id)

        return players_game_logs_df

    def merge_defensive_stats_to_players_log(self, players_game_log: pd.DataFrame, season: str):
        """_summary_

        :param players_game_log: _description_
        :param season: _description_
        :return: _description_
        """
        year = f"20{season[-2:]}"
        all_star_date = f"{year}-02-14"
        defensive_data_path = f"data/seasonal_data/{year}/defensive_data/"

        for filename in os.listdir(defensive_data_path):
            data_path = os.path.join(defensive_data_path, f"{filename}")
            defensive_data = pd.read_csv(data_path, index_col=0)

            x = 3



    @staticmethod
    def add_home_away_columns(players_game_logs_df: pd.DataFrame) -> pd.DataFrame:
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
        months_dict = LockerRoom.init_months_dict()
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
        
        return int(teams_id)

    def fetch_team_game_logs(self, team_name: str):
        """
        Fetch all the game logs for given team

        :param team_name: name of the team (i.e. Mavericks, Lakers)
        :return: the team's game logs
        """
        col, name = team_name
        team_id = self.nba_teams_info[self.nba_teams_info[col]==name]["id"]
        team_game_logs = teamgamelogs.TeamGameLogs(team_id_nullable=team_id,
                                                   season_nullable=self.current_season).get_data_frames()[0]
        team_game_logs["GAME_DATE"] = [pd.Timestamp(game_date) for game_date in team_game_logs["GAME_DATE"]]

        return team_game_logs

    def fetch_game_box_score(self, game_date: str) -> pd.DataFrame:
        """_summary_

        :param team_game_logs: _description_
        :param game_date: _description_
        :return: _description_
        """
        game_date = pd.Timestamp(game_date)
        team_game_logs = self.home_game_plan.team_game_logs
        game_id = team_game_logs[team_game_logs["GAME_DATE"] == game_date]["GAME_ID"].values[0]
        box_score = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id).get_data_frames()[0]

        return box_score

    def fetch_matchup_stats(self, off_player_id: int, def_player_id: int, season: str = "2022-23"):
        """
        :param off_player:
        :param def_player:
        :param season:
        :return:
        """
        matchup_data = leagueseasonmatchups.LeagueSeasonMatchups(off_player_id_nullable=off_player_id,
                                                                 def_player_id_nullable=def_player_id,
                                                                 season=season).get_data_frames()[0]
        return matchup_data
