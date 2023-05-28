import pandas as pd
import numpy as np

from nba_api.stats.endpoints import leaguedashptteamdefend, teamgamelogs


nba_teams_info = pd.read_excel("static_data/nba_teams.xlsx")
seasons = ["2021-22", "2022-23"]


def fetch_team_game_logs(team_name: str, season: str):
    """
    Fetch all the game logs for given team

    :param team_name: name of the team (i.e. Mavericks, Lakers)
    :return: the team's game logs
    """
    team_id = nba_teams_info[nba_teams_info["nickname"]==team_name]["id"][0]
    team_game_logs = teamgamelogs.TeamGameLogs(team_id_nullable=team_id,
                                               season_nullable=season).get_data_frames()[0]
    team_game_logs["GAME_DATE"] = [pd.Timestamp(game_date) for game_date in team_game_logs["GAME_DATE"]]

    return team_game_logs


# for season in seasons:
#     for team_name in nba_teams_info["nickname"]:
#         team_game_logs = fetch_team_game_logs(team_name, season)

