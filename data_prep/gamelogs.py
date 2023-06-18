import os
import pandas as pd

from nba_api.stats.endpoints import leaguedashptteamdefend, teamgamelogs


nba_teams_info = pd.read_csv("data/static_data/static_team_info.csv")
seasons = ["2020-21", "2021-22", "2022-23"]


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

def save_teams_logs_per_season(seasons: list):
    """
    Save the game logs of all teams for specified season(s)

    :param seasons: list of season(s) of the games to save
    """
    for season in seasons:
        for team_id in nba_teams_info["id"]:
            team_game_logs = fetch_team_game_logs(team_id, season)
            team_game_logs.to_csv(f"data/seasonal_data/20{season[-2:]}/team_logs/{team_game_logs['TEAM_ABBREVIATION'].values[0]}.csv")

def get_opp_id(matchup: str):
    """
    Fetch opponent's ID when looking at the matchup
    """
    matchup_split = matchup.split(" ")
    opp_abb = matchup_split[2]
    team_id = nba_teams_info[nba_teams_info["abbreviation"]==opp_abb]["id"].values[0]
    
    return team_id

def merge_defensive_stats(season: str, game_log: pd.DataFrame, pre_asb_stats: list,
                          post_asb_stats: list, metrics: pd.DataFrame=None) -> pd.DataFrame:
    """
    Merge the necessary defensive stats to current team game logs
    The stats are separated by the All Star Break (ASB) date

    It's important to distinguish the 2 periods because historically
    teams perform differently after this break point, due to many changes
    like trades, signings, waivers, etc...

    :param season: season string, ex: "2022-23" is the 2023 NBA season
    :param game_log: game log of the NBA team
    :param pre_asb_stats: list of all pre-ASB stats
    :param post_asb_stats: list of all post-ASB stats
    :param metrics: this team's defensive metrics for the entire season

    :return: a complete game log with all stats merged as new columns
    """
    all_star_date = f"20{season}-02-14"
    game_log = game_log[["SEASON_YEAR", "TEAM_ABBREVIATION", "TEAM_NAME", "GAME_ID", "GAME_DATE", "MATCHUP"]]
    game_log["TEAM_ID"] = game_log["MATCHUP"].apply(get_opp_id)
    
    pre_asb = game_log.copy()
    for defensive_stats in pre_asb_stats:
        pre_asb = pre_asb[pre_asb["GAME_DATE"] < all_star_date].merge(defensive_stats, on="TEAM_ID")
    
    post_asb = game_log.copy()
    for defensive_stats in post_asb_stats:
        post_asb = post_asb[post_asb["GAME_DATE"] >= all_star_date].merge(defensive_stats, on="TEAM_ID")
    
    complete_log = pd.concat([pre_asb, post_asb]).merge(metrics, on="TEAM_ID")
    complete_log = complete_log.sort_values(by=["GAME_DATE"], ascending=False)

    return complete_log


def merge_defensive_stats_to_game_logs(seasons: list):
    """
    Function to actually save & update the game logs

    :param seasons: _description_
    """
    for season in seasons:
        dir = f"data/seasonal_data/20{season[-2:]}"

        for team_abb in nba_teams_info["abbreviation"]:
            defense_data_dir = f"{dir}/defensive_data"
            team_logs_dir = f"{dir}/team_logs"
            team_logs_path = f"{team_logs_dir}/{team_abb}.csv"
            team_logs_data = pd.read_csv(team_logs_path, index_col=0)

            pre_asb_data = []
            post_asb_data = []
            for filename in os.listdir(defense_data_dir):
                defense_file_path = os.path.join(defense_data_dir, filename)
                if os.path.isfile(defense_file_path):
                    defense_data = pd.read_csv(defense_file_path, index_col=0)
                if "pre" in filename:
                    pre_asb_data.append(defense_data)
                elif "post" in filename:
                    post_asb_data.append(defense_data)
                else:
                    metrics = defense_data

            complete_log = merge_defensive_stats(season, team_logs_data, pre_asb_data, post_asb_data, metrics)
            complete_log.to_csv(f"{team_logs_path}")

save_teams_logs_per_season(seasons)
merge_defensive_stats_to_game_logs(seasons)
