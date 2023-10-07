# Set up experiments pipeline
# Pick 333 random unique games to forecast per each model (999 runs total)
# 666 scores to evaluate RMSE's (2 scores for each playing team)

import pandas as pd
import random
from config import oracle_config, nn_config, xgboost_config
from models.oracle import Oracle


def create_game_details_dict(df_row) -> dict:
    """_summary_

    :param df_row: _description_
    """
    game_dict = {}
    matchup = df_row.MATCHUP.split(" ")
    if matchup[1] == "@":
        home_team = teams_info[teams_info["abbreviation"]==matchup[2]]["nickname"]
        away_team = teams_info[teams_info["abbreviation"]==matchup[0]]["nickname"]
    else:
        home_team = teams_info[teams_info["abbreviation"]==matchup[0]]["nickname"]
        away_team = teams_info[teams_info["abbreviation"]==matchup[2]]["nickname"]
    
    game_dict["home_team"] = home_team.values[0]
    game_dict["away_team"] = away_team.values[0]
    game_dict["game_date"] = df_row.GAME_DATE

    return game_dict

teams_info = pd.read_csv("data/static_data/static_team_info.csv", index_col=0)
all_logs = pd.read_csv("data/all_logs.csv", index_col=0)
latest_season = "2022-23"

latest_season_games = all_logs[all_logs["SEASON_YEAR"]==latest_season]
unique_games = list(latest_season_games["GAME_ID"].unique())

random.seed(41)
sampled_game_ids = random.sample(unique_games, 333)
games_selected = all_logs[all_logs["GAME_ID"].isin(sampled_game_ids)].drop_duplicates(subset="GAME_ID")


for _, df_row in games_selected.iterrows():
    game_dict = create_game_details_dict(df_row)
    oracle = Oracle(game_dict, )



