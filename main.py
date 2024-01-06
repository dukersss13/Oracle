from time import time
from models.oracle import Oracle
from config import game_details, oracle_config, xgboost_config, nn_config


if __name__ == '__main__':
    start = time()
    oracle = Oracle(game_details=game_details, oracle_config=oracle_config, model_config=nn_config)
    oracle.run()
    end = time()
    print(f"Total solve time E2E: {round((end-start) / 60)} minutes")


# TODO
# If player's avg mins =< 5, don't play them
# Fix Neural Network 
# Forecast Player's Attempts using Minutes
# attempts = f(minutes) & % = f(opp_defense) & pts = f(attempts, %)
# Build GRU/LSTM
