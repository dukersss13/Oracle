from time import time
from models.oracle import Oracle

if __name__ == "__main__":
    start = time()
    oracle = Oracle()
    oracle.run()
    end = time()
    print(f"Total solve time E2E: {round((end-start) / 60)} minutes")


### INSTRUCTIONS
# Set your desired game to forecast in oracle.conf under "game_details"
# If this is your first time running Oracle, you will need to set
# the "fetch_new_data" flag to true

# Once ready, run main.py
