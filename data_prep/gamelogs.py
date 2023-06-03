import pandas as pd
import numpy as np

from nba_api.stats.endpoints import leaguedashptteamdefend, teamgamelogs


nba_teams_info = pd.read_csv("data/static_data/static_team_info.csv")
seasons = ["2022-23"]


def fetch_team_game_logs(team_id: str, season: str):
    """
    Fetch all the game logs for given team

    :param team_name: name of the team (i.e. Mavericks, Lakers)
    :return: the team's game logs
    """
    team_game_logs = teamgamelogs.TeamGameLogs(team_id_nullable=team_id,
                                               season_nullable=season).get_data_frames()[0]
    team_game_logs["GAME_DATE"] = [pd.Timestamp(game_date) for game_date in team_game_logs["GAME_DATE"]]

    return team_game_logs


for season in seasons:
    for team_id in nba_teams_info["id"]:
        team_game_logs = fetch_team_game_logs(team_id, season)
        team_game_logs.to_csv(f"data/seasonal_data/20{season[-2:]}/team_logs/{team_game_logs['TEAM_ABBREVIATION'].values[0]}.csv")
