import os

from typing import Tuple
import json
import pandas as pd
import numpy as np
from enum import Enum

from nba_api.stats.endpoints import playergamelog, leagueseasonmatchups, commonteamroster, teamgamelogs
from nba_api.stats.library.parameters import SeasonAll
from nba_api.stats.static import teams, players
import tensorflow as tf
from dataclasses import dataclass

pd.set_option('mode.chained_assignment', None)

class Team(Enum):
    HOME = 0
    AWAY = 1


@dataclass
class TeamData:
    team_name: str = None
    team_game_logs: pd.DataFrame = None
    team_roster: np.ndarray = None
    active_players: np.ndarray = None


class DataFetcher:
    def __init__(self, home_team: str = None, away_team: str = None, nn_config: dict = None, season="2022-23"):
        self.home_team = home_team
        self.away_team = away_team
        self.season = season
        self.nn_config = nn_config
        self.predictors = nn_config["predictors"]
        self.num_seasons = nn_config["num_seasons"]

        self.target_path = f"{os.getcwd()}/artifacts/active_players.json"
        self.fetch_teams_data()

    @staticmethod
    def fetch_team_dict(team_name: str) -> dict:
        """
        Fetch the dictionary data for given team. The dictionary contains team_id, team nick name, etc...
        :param team_name: name of NBA team

        :return: team_data: data of the team
        """
        nba_teams = teams.get_teams()
        team_dict = [team for team in nba_teams if team["nickname"] == team_name][0]

        return team_dict

    @staticmethod
    def init_months_dict() -> dict:
        """
        Create months dictionary.
        """
        months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
        months_dict = dict(zip(months, range(1, len(months)+1)))

        return months_dict

    def fetch_roster(self, team_name: str) -> np.ndarray:
        """
        Fetch the roster of given team.

        :param team_name: name of NBA team.
        :return: array of team roster
        """
        if " " in team_name:
            team_name = team_name.title()
        else:
            team_name = team_name.capitalize()

        team_dict = DataFetcher.fetch_team_dict(team_name)
        team_roster = commonteamroster.CommonTeamRoster(team_id=team_dict["id"],
                                                        season=self.season).get_data_frames()[0]["PLAYER"].values
        
        return team_roster
    
    def fetch_teams_data(self):
        """
        Fetch the data needed for each team & create/update active players json.
        """
        self.home_team_data = TeamData()
        self.away_team_data = TeamData()

        self.home_team_data.team_name = self.home_team
        self.away_team_data.team_name = self.away_team

        self.home_team_data.team_roster = self.fetch_roster(self.home_team)
        self.away_team_data.team_roster = self.fetch_roster(self.away_team)

        self.home_away_dict = {self.home_team: Team.HOME, self.away_team: Team.AWAY}

        self.update_active_players_json()

    def fetch_players_game_logs_df(self, players_id: str) -> pd.DataFrame:
        """
        Access the PlayerGameLog module to fetch the game logs df of given player.

        :param players_id: player ID.
        :return: the given player's game logs in df format.
        """
        last_games = int(82 * self.num_seasons)
        players_game_log = playergamelog.PlayerGameLog(player_id=players_id, season=SeasonAll.all,
                                                       season_type_all_star="Regular Season")
        players_game_logs_df = players_game_log.get_data_frames()[0]
        try:
            players_game_logs_df =  players_game_logs_df.iloc[:last_games+1, :]
        except:
            players_game_logs_df = players_game_logs_df

        return players_game_logs_df

    def get_filtered_players_logs(self, players_full_name: str) -> Tuple[np.ndarray, pd.Timestamp]:
        """
        Retrieve the filtered game logs for given player.

        :param players_full_name: full name of the player.
        :return filtered_log.values: an array of player's game logs filtered by specific columns.
        """
        players_id = self.fetch_players_id(players_full_name)

        if players_id is None:
            print(f"WARNING: Cannot retrieve logs for {players_full_name}")
            return None, None

        players_game_logs_df = self.fetch_players_game_logs_df(players_id)
        players_game_logs_with_rest_df = DataFetcher.add_rest_days(players_game_logs_df)
        most_recent_game_date = self.get_most_recent_game_date(players_game_logs_df)
        complete_players_game_logs = DataFetcher.add_home_away_columns(players_game_logs_with_rest_df)
        filtered_log = DataFetcher.filter_stats(complete_players_game_logs, self.predictors)

        return filtered_log.values, most_recent_game_date

    def get_most_recent_game_date(self, players_game_logs_df: pd.DataFrame) -> pd.Timestamp:
        """_summary_

        :param players_game_logs_df: _description_
        :return: _description_
        """
        if self.nn_config["holdout"] in [0, 1]:
            return players_game_logs_df["GAME_DATE"].values[self.nn_config["holdout"]]
        else:
            raise ValueError("Oracle currently can only handle at most 1 holdout game!")

    def set_active_players(self):
        """
        """
        with open(self.target_path) as f:
            active_players_json = json.load(f)
        
        for team in active_players_json:
            team_data = self.home_team_data if self.home_away_dict[team] == Team.HOME else self.away_team_data
            active_players_df = pd.DataFrame(active_players_json[team], index=["Mins"]).T
            active_players = active_players_df[active_players_df["Mins"] != 0]
            team_data.active_players = active_players.index.values

    def update_active_players_json(self):
        """
        Update the active players json to set active players or manually assign minutes.
        """
        # Check if the json file exists in the first place
        # If not, create one
        self.check_active_players_json_exists()

        with open(self.target_path) as f:
            active_players_json = json.load(f)
        
        for team_name in active_players_json:
            del team_name
        
        active_players_json = self.init_active_players_json()
        with open(self.target_path, 'w') as f:
            json.dump(active_players_json, f, indent=1)

    def check_active_players_json_exists(self):
        """
        Check if active players json exists. If not, create one.
        """
        if not os.path.exists(self.target_path):
            active_players_json = self.init_active_players_json()
            with open(self.target_path, 'w') as f:
                json.dumps(active_players_json, f, indent=1)

    def init_active_players_json(self):
        """
        """
        home_roster = self.home_team_data.team_roster
        away_roster = self.away_team_data.team_roster

        json =  {self.home_team: dict(zip(home_roster, [None for _ in range(len(home_roster))])),
                 self.away_team: dict(zip(away_roster, [None for _ in range(len(away_roster))]))}
        
        return json

    @staticmethod
    def add_rest_days(players_game_logs_df: pd.DataFrame) -> pd.DataFrame:
        """
        """
        players_game_logs_df["GAME_DATE"] = players_game_logs_df["GAME_DATE"].apply(lambda x: x.split(" "))
        players_game_logs_df["GAME_DATE"] = players_game_logs_df["GAME_DATE"].apply(DataFetcher.convert_to_timestamp)
        players_game_logs_df["REST_DAYS"] = players_game_logs_df["GAME_DATE"].diff(periods=-1) - pd.Timedelta(1, "days")
        players_game_logs_df = players_game_logs_df.iloc[:-1, :]
        players_game_logs_df.loc[:, "REST_DAYS"] = players_game_logs_df["REST_DAYS"].dt.days

        return players_game_logs_df

    @staticmethod
    def add_home_away_columns(players_game_logs_df: pd.DataFrame) -> pd.DataFrame:
        """
        """
        players_game_logs_df.loc[:, ("HOME", "AWAY")] = tf.one_hot(players_game_logs_df["MATCHUP"].\
                                                        apply(DataFetcher.detect_home_or_away_games), 2)

        return players_game_logs_df

    @staticmethod
    def detect_home_or_away_games(value):
        """
        """
        return 1 if "@" in value else 0

    @staticmethod
    def convert_to_timestamp(value):
        """
        """
        months_dict = DataFetcher.init_months_dict()
        value = pd.Timestamp(f"{value[2]}-{months_dict[value[0]]}-{value[1][:-1]}")

        return value

    @staticmethod
    def filter_stats(game_logs_df: pd.DataFrame, columns_wanted: list) -> pd.DataFrame:
        """
        :param game_logs_df:
        :param columns_wanted:
        :return:
        """
        if columns_wanted is None:
            return game_logs_df

        return game_logs_df[columns_wanted]

    @staticmethod
    def fetch_players_id(players_full_name):
        """
        :param: players_full_name:
        :return:
        """
        try:
            players_id = players.find_players_by_full_name(players_full_name)[0]["id"]
        except IndexError:
            print(f"WARNING: {players_full_name} does not have a player ID!")
            players_id = None
        
        return players_id

    ### EXTRA
    def fetch_matchup_stats(self, off_player: str, def_player: str, season: str = "2021-22"):
        """
        :param off_player:
        :param def_player:
        :param season:
        :return:
        """
        off_player_id = self.fetch_players_id(off_player)
        def_player_id = self.fetch_players_id(def_player)
        matchup_data = leagueseasonmatchups.LeagueSeasonMatchups(off_player_id_nullable=off_player_id,
                                                                 def_player_id_nullable=def_player_id,
                                                                 season=season).get_data_frames()[0]
        return matchup_data

    def fetch_team_game_logs(self, team_name: str):
        """
        """
        team_dict = DataFetcher.fetch_team_dict(team_name.capitalize())
        team_game_logs = teamgamelogs.TeamGameLogs(team_id_nullable=team_dict["id"],
                                                   season_nullable=self.season).get_data_frames()[0]

        return team_game_logs