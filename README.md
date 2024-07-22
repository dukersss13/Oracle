# Oracle: NBA Forecasting Model
 
### Overview
Oracle is an NBA forecasting model that leverages data from [nba_api](https://github.com/swar/nba_api/tree/master). With the latest NBA games data (individual and teams), Oracle uses a Gated Recurrent Unit (GRU) Neural Network in its core to forecast the performance of every player before aggregating up to a team's level. The forecasting process for the individual player requires heavy feature engineering work, uses features such as `MINS, FG%, FGA, FT%, etc...` & has `PTS` as the response variable. 


### Architecture Overview

