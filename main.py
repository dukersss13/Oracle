from time import time
from models.oracle import Oracle
from config import game_details, oracle_config, nn_config, xgboost_config


if __name__ == '__main__':
    start = time()
    oracle = Oracle(game_details=game_details, oracle_config=oracle_config, model_config=nn_config)
    oracle.run()x
    end = time()
    print(f"Total solve time E2E: {round((end-start) / 60)} minutes")