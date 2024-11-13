import os
import pandas as pd

from nba_api.stats.endpoints import teamgamelogs, leaguedashptteamdefend, teamestimatedmetrics

nba_teams_info = pd.read_csv("data/static_data/static_team_info.csv", index_col=0)


def fetch_defensive_stats(seasons: list[str], season_segment: str=None):
    """_summary_

    :param team_id: _description_
    :param season: _description_
    """
    defense_categories = ["Overall", "3 Pointers", "2 Pointers", "Less Than 10Ft"]
    for season in seasons:
        data_path = f"data/seasonal_data/20{season[-2:]}/defensive_data"
        for category in defense_categories:
            team_defensive_stats: pd.DataFrame = leaguedashptteamdefend.LeagueDashPtTeamDefend(season=season,
                                                                                               defense_category=category,
                                                                                 season_segment_nullable=season_segment).get_data_frames()[0]
            if not os.path.exists(data_path):
                os.makedirs(data_path)

            if season_segment == "Post All-Star":
                file_name = f"post_asb_{category}.csv"
            else:
                file_name = f"pre_asb_{category}.csv"
            
            team_defensive_stats.to_csv(f"{data_path}/{file_name}")
        
        if not os.path.exists(f"{data_path}/overall_defensive_metrics_{season[-2:]}.csv"):
            team_overall_defensive_metrics: pd.DataFrame = teamestimatedmetrics.TeamEstimatedMetrics(season=season).get_data_frames()[0]
            team_overall_defensive_metrics = team_overall_defensive_metrics[["TEAM_ID", "E_PACE", "E_DEF_RATING"]]
            team_overall_defensive_metrics.to_csv(f"{data_path}/overall_defensive_metrics_{season[-2:]}.csv")


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
    print(f"Fetching new game logs for {seasons}")
    for season in seasons:
        path = f"data/seasonal_data/20{season[-2:]}/team_logs"
        for team_id in nba_teams_info["id"]:
            team_game_logs = fetch_team_game_logs(team_id, season)
            team_game_logs: pd.DataFrame = team_game_logs[["SEASON_YEAR", "TEAM_ABBREVIATION", "TEAM_NAME",
                                                           "GAME_ID", "GAME_DATE", "MATCHUP"]]

            if not os.path.exists(path):
                os.makedirs(path)

            team_game_logs.to_csv(f"{path}/{team_game_logs['TEAM_ABBREVIATION'].values[0]}.csv")
                
def get_opp_id(matchup: str):
    """
    Fetch opponent's ID when looking at the matchup
    """
    matchup_split = matchup.split(" ")
    opp_abb = matchup_split[2]
    team_id = nba_teams_info[nba_teams_info["abbreviation"]==opp_abb]["id"].values[0]
    
    return team_id

def merge_defensive_stats(season: str, game_log: pd.DataFrame, pre_asb_stats: list = [],
                          post_asb_stats: list = [], metrics: pd.DataFrame=None) -> pd.DataFrame:
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
    all_star_date = f"20{season[-2:]}-02-14"
    game_log["TEAM_ID"] = game_log["MATCHUP"].apply(get_opp_id)
    
    pre_asb = game_log.copy()
    for defensive_stats in pre_asb_stats:
        col_to_drop = [col for col in defensive_stats.columns if "PLUSMINUS" in col]
        defensive_stats: pd.DataFrame = defensive_stats.drop(columns=["TEAM_NAME", "TEAM_ABBREVIATION", "FREQ",
                                                                      "GP", "G"] + col_to_drop)
        pre_asb = pre_asb[pre_asb["GAME_DATE"] < all_star_date].merge(defensive_stats, on="TEAM_ID")
    
    if len(post_asb_stats):
        post_asb = game_log.copy()
        for defensive_stats in post_asb_stats:
            defensive_stats: pd.DataFrame = defensive_stats.drop(columns=["TEAM_NAME", "TEAM_ABBREVIATION", "FREQ",
                                                                          "GP", "G"])
            post_asb = post_asb[post_asb["GAME_DATE"] >= all_star_date].merge(defensive_stats, on="TEAM_ID")
    else:
        post_asb = pd.DataFrame([])
    
    complete_log = pd.concat([pre_asb, post_asb]).merge(metrics, on="TEAM_ID")
    complete_log = complete_log.sort_values(by=["GAME_DATE"], ascending=False)

    return complete_log

def merge_defensive_stats_to_game_logs(seasons: list):
    """
    Function to actually save & update the game logs

    :param seasons: _description_
    """
    print("Merging defensive data to game logs")
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


def consolidate_all_game_logs(seasons: list, season_to_update: list[str]):
    """
    Consolidate all the logs into 1
    across ALL collected seasons
    """
    print(f"Adding new game logs from {seasons} to all_logs.csv")
    season_all_logs = []
    for season in season_to_update:
        season_team_logs = []
        dir = f"data/seasonal_data/20{season[-2:]}/team_logs"
        for filename in os.listdir(dir):
            team_log = pd.read_csv(f"{dir}/{filename}", index_col=0)
            season_team_logs.append(team_log)
        season_team_logs: pd.DataFrame = pd.concat(season_team_logs, axis=0)
        season_team_logs.to_csv(f"{dir}/all_logs.csv")

    all_logs = []
    for season in seasons:
        dir = f"data/seasonal_data/20{season[-2:]}/team_logs"
        season_all_logs = pd.read_csv(f"{dir}/all_logs.csv")
        # if season == "2023-24":
        #     cols_wanted = season_all_logs.columns
        # season_all_logs = season_all_logs[cols_wanted]
        all_logs.append(season_all_logs)
    
    all_logs = pd.concat(all_logs, axis=0)
    all_logs.reset_index(inplace=True)
    all_logs.to_csv(f"data/all_logs.csv")
        


def update_data(seasons: list):
    save_teams_logs_per_season(seasons)
    fetch_defensive_stats(seasons, season_segment="Pre All-Star")
    merge_defensive_stats_to_game_logs(seasons)
