# Set up experiments pipeline
# Pick 333 random unique games to forecast per each model (999 runs total)
# 666 scores to evaluate RMSE's (2 scores for each playing team)

import pandas as pd
from models.oracle import Oracle
from config import oracle_config, nn_config, xgboost_config


all_logs = pd.read_csv("data/all_logs.csv")
latest_season = "2022-23"

latest_season_games = all_logs[all_logs["SEASON_YEAR"]==latest_season]
unique_games = latest_season_games["GAME_ID"].unique()
experiment_game_ids = []

