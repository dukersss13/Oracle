# Oracle: NBA Forecasting Model
 
### Overview
Oracle is an NBA forecasting model that leverages data from [nba_api](https://github.com/swar/nba_api/tree/master). With the latest NBA games data (individual and teams), Oracle uses a Gated Recurrent Unit (GRU) Neural Network or XGB (your choice) in its core to forecast the performance of every player before aggregating up to a team's level. The forecasting process for the individual player requires heavy feature engineering work, uses features such as `MINS, FG%, FGA, FT%, etc...` & has `PTS` as the target variable. 

Oracle can fetch the latest data thanks to NBA API library. It only retrieves the latest 3 seasons for training purposes to avoid noise caused by players' ascension or descension. 


### Architecture Overview

1. Provide the **Oracle** with the game and model config files.
2. **Game config**: contains information related to the game. **Model config**: information related to the model chosen for forecast.
3. Oracle will trigger the LockerRoom class which will fetch data from NBA API, performs data preprocessing steps to prepare the necessary training and testing data.
4. Data Preprocessing extracts offensive features of the players (mentioned above) along with the opposing teams' defensive metrics to take into account of the opponents played as well as non-basketball statistics such as `REST_DAYS (int)`, `HOME_OR_AWAY (bool)`.
5. Forecasts will be done for each individual player, as each player will have a custom trained model on their game logs. Team forecast is the aggregation of all individual player's forecast.

![image](https://github.com/user-attachments/assets/9e6b200e-f585-4cb0-a56a-9a64b7cbf41c)




### Oracle in Action

#### Game 1: Hawks @ Lakers. March 18th, 2024. 
[Box Score](https://www.google.com/search?q=hawks+vs+lakers+march+2024&sca_esv=02ef9b5328b381af&rlz=1C5GCEM_enUS1102US1102&sxsrf=ADLYWIIKnHspuFZj6FNYxYIJnNA0o62CcA%3A1731016424803&ei=6DYtZ7TdMPbBkPIP_8He4Q8&ved=0ahUKEwi08ebYmsuJAxX2IEQIHf-gN_wQ4dUDCA8&uact=5&oq=hawks+vs+lakers+march+2024&gs_lp=Egxnd3Mtd2l6LXNlcnAiGmhhd2tzIHZzIGxha2VycyBtYXJjaCAyMDI0MgUQIRigATIFECEYoAEyBRAhGKABMgUQIRigATIFECEYoAFI_EZQkCxYwUFwAXgBkAEAmAF5oAGwDaoBBDI1LjG4AQPIAQD4AQGYAhugAoAOwgIKEAAYsAMY1gQYR8ICBBAjGCfCAgoQIxiABBgnGIoFwgILEAAYgAQYkQIYigXCAgoQABiABBhDGIoFwgIOEC4YgAQYsQMY0QMYxwHCAhQQLhiABBixAxjRAxiDARjHARiKBcICDhAAGIAEGLEDGIMBGIoFwgIQEC4YgAQYxwEYJxiKBRivAcICERAuGIAEGJECGLEDGIMBGIoFwgIREAAYgAQYkQIYsQMYgwEYigXCAhQQLhiABBiRAhixAxiDARjJAxiKBcICCxAAGIAEGJIDGIoFwgIQEAAYgAQYsQMYQxiDARiKBcICCxAAGIAEGLEDGIMBwgIIEAAYgAQYsQPCAgQQABgDwgIIEC4YgAQYsQPCAgoQABiABBgUGIcCwgIFEAAYgATCAgYQABgWGB7CAggQABiABBiiBMICCBAAGKIEGIkFwgIFECEYqwLCAgUQIRifBZgDAIgGAZAGCJIHBDI2LjGgB4KuAQ&sclient=gws-wiz-serp#sie=m;/g/11kt8qf25m;3;/m/05jvx;tb1;fp;1;;;)

Forecast: Hawks 108 - 131 Lakers.
Actual: Hawks 105 - 136 Lakers.


**Oracle Forecast vs. Real Box Score**

![image](https://github.com/user-attachments/assets/9fc36df0-ce0c-4b34-a09f-bc228f5fd7ef)







#### Game 2: Nuggets @ Spurs. March 15th, 2024.
[Box Score](https://www.google.com/search?q=spurs+vs+nuggets+2024&sca_esv=02ef9b5328b381af&rlz=1C5GCEM_enUS1102US1102&sxsrf=ADLYWIJT3x2ykkHGo5dtogK_nk1UDZXXGA%3A1731016471558&ei=FzctZ-HpIY3KkPIPmrWiqQE&ved=0ahUKEwih0YzvmsuJAxUNJUQIHZqaKBUQ4dUDCA8&uact=5&oq=spurs+vs+nuggets+2024&gs_lp=Egxnd3Mtd2l6LXNlcnAiFXNwdXJzIHZzIG51Z2dldHMgMjAyNDIFEAAYgAQyCxAAGIAEGIYDGIoFMgsQABiABBiGAxiKBTIIEAAYogQYiQUyCBAAGKIEGIkFMggQABiiBBiJBUjVIFCMBVjaH3AFeAGQAQCYAVygAdELqgECMjW4AQPIAQD4AQGYAh6gAq8MwgIKEAAYsAMY1gQYR8ICERAuGIAEGJECGMcBGIoFGK8BwgIREC4YgAQYkQIY0QMYxwEYigXCAgoQABiABBhDGIoFwgIKEC4YgAQYQxiKBcICIBAuGIAEGJECGMcBGIoFGK8BGJcFGNwEGN4EGOAE2AEBwgIKECMYgAQYJxiKBcICChAuGIAEGCcYigXCAgsQABiABBiRAhiKBcICERAuGIAEGJECGLEDGIMBGIoFwgIQEC4YgAQYxwEYJxiKBRivAcICEBAAGIAEGLEDGEMYgwEYigXCAh0QLhiABBjHARiKBRivARiXBRjcBBjeBBjgBNgBAcICEBAuGIAEGLEDGEMYgwEYigXCAgsQLhiABBixAxiDAcICCxAAGIAEGLEDGIMBwgIOEAAYgAQYsQMYgwEYigXCAgoQABiABBgUGIcCwgIGEAAYFhgewgIFECEYoAGYAwCIBgGQBgi6BgYIARABGBSSBwIzMKAHgbgB&sclient=gws-wiz-serp#sie=m;/g/11vchf1k3q;3;/m/05jvx;dt;fp;1;;;)

Forecast: Spurs 106 - 121 Nuggets.
Actual: Spurs 106 - 117 Nuggets.


**Oracle Forecast vs. Real Box Score**

![image](https://github.com/user-attachments/assets/9d4721a1-8c67-4f3f-84db-282636c4f340)





### Final Remarks
Oracle performs best when the reality is in line with the empirical data, just like any other machine learning model. What this means is it cannot accurately predict/foresee injuries (undisclosed prior to the game or during the game), cannot predict the game plan (if a starter, historically, gets traded to a new team and is now a role player), and it specifically struggles with how the minutes and shots distribution will be if star players don't play. 

An example would be if Lebron James does not play for the Lakers for a game, who is taking up his minutes or his shots? This can only be an educated guess as we do not have the information of the game plan.
