import pandas as pd
import numpy as np
from enum import Enum

from nba_api.stats.endpoints import playergamelog, leagueseasonmatchups, commonteamroster, teamgamelogs
from nba_api.stats.library.parameters import SeasonAll
from nba_api.stats.static import teams, players
from dataclasses import dataclass


class Team(Enum):
    HOME = 0
    AWAY = 1


@dataclass
class TeamData:
    team_name: str = None
    team_game_logs: pd.DataFrame = None
    team_roster: np.ndarray = None
    individual_players_game_logs: dict = None
    game_forecast: pd.DataFrame = None


class DataFetcher:
    def __init__(self, home_team: str = None, away_team: str = None, nn_config: dict = None, season="2022-23"):
        self.home_team = home_team
        self.away_team = away_team
        self.season = season
        self.predictors = nn_config["predictors"]
        self.num_seasons = nn_config["num_seasons"]
        self.fetch_teams_data()

    @staticmethod
    def fetch_teams_dict(team_name: str):
        """

        :param team_name:
        :return:
        """
        nba_teams = teams.get_teams()
        team_data = [team for team in nba_teams if team["nickname"] == team_name][0]

        return team_data

    @staticmethod
    def make_dict(keys, values):
        return {key: value for key, value in zip(keys, values)}

    def fetch_rosters(self, team_name: str):
        """
        :return:
        """
        team_dict = DataFetcher.fetch_teams_dict(team_name.capitalize())
        team_roster = commonteamroster.CommonTeamRoster(team_id=team_dict["id"],
                                                        season=self.season).get_data_frames()[0]["PLAYER"].values
        
        return team_roster
    
    def fetch_teams_data(self):
        """
        """
        self.home_team_data = TeamData()
        self.away_team_data = TeamData()

        self.home_team_data.team_name = self.home_team
        self.away_team_data.team_name = self.away_team

        self.home_team_data.team_roster = self.fetch_rosters(self.home_team)
        self.away_team_data.team_roster = self.fetch_rosters(self.away_team)

    def fetch_players_game_logs_df(self, players_id: str) -> pd.DataFrame:
        """
        :param players_full_name:
        :param last_games:
        :return:
        """
        last_games = 82 * self.num_seasons

        players_game_log = playergamelog.PlayerGameLog(player_id=players_id, season=SeasonAll.all,
                                                       season_type_all_star="Regular Season")
        players_game_logs_df = players_game_log.get_data_frames()[0]
        try:
            players_game_logs_df =  players_game_logs_df.iloc[:last_games, :]
        except:
            players_game_logs_df = players_game_logs_df

        return players_game_logs_df

    def get_filtered_players_logs(self, players_full_name: str) -> np.ndarray:
        """
        """
        players_id = self.fetch_players_id(players_full_name)

        if players_id is None:
            print(f"WARNING: Cannot retrieve logs for {players_full_name}")
            return None

        players_game_logs_df = self.fetch_players_game_logs_df(players_id)
        filtered_log = DataFetcher.filter_stats(players_game_logs_df, self.predictors)

        return filtered_log.values

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
        team_dict = DataFetcher.fetch_teams_dict(team_name.capitalize())
        team_game_logs = teamgamelogs.TeamGameLogs(team_id_nullable=team_dict["id"],
                                                   season_nullable=self.season).get_data_frames()[0]

        return team_game_logs