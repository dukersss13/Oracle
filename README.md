# Oracle: NBA Forecasting Model
 
### Overview
Oracle is an NBA forecasting model that leverages data from [nba_api](https://github.com/swar/nba_api/tree/master). With the latest NBA games data (individual and teams), Oracle uses a Gated Recurrent Unit (GRU) Neural Network in its core to forecast the performance of every player before aggregating up to a team's level. The forecasting process for the individual player requires heavy feature engineering work, uses features such as `MINS, FG%, FGA, FT%, etc...` & has `PTS` as the response variable. 


### Architecture Overview

1. Provide the **Oracle** with the game and model config files.
2. Game config: contains information related to the game. Model config: information related to the model chosen for forecast.
3. Oracle will trigger the LockerRoom class which will fetch data from NBA API, performs data preprocessing steps to prepare the necessary training and testing data.
4. Forecasts will be done for each individual player, as each player will have a custom trained model on their game logs. Team forecast is the aggregation of all individual player's forecast.

![image](https://github.com/user-attachments/assets/26810daf-f3f9-4b86-805c-9ff2da9846cd)

