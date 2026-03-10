import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GRU, Dense
from xgboost import XGBRegressor

from nba_api.stats.endpoints import leaguegamelog

SEASON = "2024-25"
CACHE_PATH = Path("artifacts/cache/leaguegamelog_2024_25.csv")


def _fetch_league_team_logs_with_retry(retries: int = 3, timeout: int = 60) -> pd.DataFrame:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return leaguegamelog.LeagueGameLog(
                season=SEASON,
                player_or_team_abbreviation="T",
                timeout=timeout,
            ).get_data_frames()[0]
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(attempt * 2)

    raise RuntimeError("Failed to fetch LeagueGameLog data") from last_err


def _load_base_data() -> pd.DataFrame:
    if CACHE_PATH.exists():
        df = pd.read_csv(CACHE_PATH)
    else:
        df = _fetch_league_team_logs_with_retry()
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CACHE_PATH, index=False)

    required = {"TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "MATCHUP", "PTS"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns in league logs: {sorted(missing)}")

    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df["PTS"] = pd.to_numeric(df["PTS"], errors="coerce")
    df = df.dropna(subset=["TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "MATCHUP", "PTS"])
    df["HOME"] = (~df["MATCHUP"].astype(str).str.contains("@", regex=False)).astype(int)
    return df.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ABBREVIATION"]).reset_index(drop=True)


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    grp = out.groupby("TEAM_ABBREVIATION", group_keys=False)

    for lag in range(1, 6):
        out[f"PTS_LAG_{lag}"] = grp["PTS"].shift(lag)

    out["PTS_ROLL_MEAN_5"] = grp["PTS"].shift(1).rolling(5).mean().reset_index(level=0, drop=True)
    out["PTS_ROLL_STD_5"] = grp["PTS"].shift(1).rolling(5).std().reset_index(level=0, drop=True)

    out["PREV_GAME_DATE"] = grp["GAME_DATE"].shift(1)
    out["REST_DAYS"] = (out["GAME_DATE"] - out["PREV_GAME_DATE"]).dt.days

    return out.dropna().reset_index(drop=True)


def _split_train_test_by_date(df: pd.DataFrame, train_frac: float = 0.75):
    unique_dates = np.array(sorted(df["GAME_DATE"].unique()))
    if len(unique_dates) < 2:
        raise RuntimeError("Not enough dates to split train/test")

    split_idx = max(1, int(len(unique_dates) * train_frac))
    split_idx = min(split_idx, len(unique_dates) - 1)
    split_date = unique_dates[split_idx]

    train_df = df[df["GAME_DATE"] < split_date].copy()
    test_df = df[df["GAME_DATE"] >= split_date].copy()
    if train_df.empty or test_df.empty:
        raise RuntimeError("Train/test split failed; empty partition")

    return train_df, test_df


def _winner_accuracy(pred_df: pd.DataFrame) -> float:
    paired = pred_df.groupby("GAME_ID").filter(lambda g: len(g) == 2)
    if paired.empty:
        return float("nan")

    outcomes = []
    for _, group in paired.groupby("GAME_ID"):
        pred_team = group.loc[group["pred_points"].idxmax(), "TEAM_ABBREVIATION"]
        actual_team = group.loc[group["actual_points"].idxmax(), "TEAM_ABBREVIATION"]
        outcomes.append(int(pred_team == actual_team))

    return float(np.mean(outcomes)) if outcomes else float("nan")


def run_xgboost_experiment(df_feat: pd.DataFrame) -> dict:
    feature_cols = [
        "HOME",
        "REST_DAYS",
        "PTS_LAG_1",
        "PTS_LAG_2",
        "PTS_LAG_3",
        "PTS_LAG_4",
        "PTS_LAG_5",
        "PTS_ROLL_MEAN_5",
        "PTS_ROLL_STD_5",
    ]

    train_df, test_df = _split_train_test_by_date(df_feat)

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=350,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=4,
    )
    model.fit(train_df[feature_cols], train_df["PTS"])

    preds = model.predict(test_df[feature_cols])
    rmse = float(np.sqrt(mean_squared_error(test_df["PTS"], preds)))

    pred_df = test_df[["GAME_ID", "TEAM_ABBREVIATION", "PTS"]].copy()
    pred_df["pred_points"] = preds
    pred_df = pred_df.rename(columns={"PTS": "actual_points"})

    paired_games = pred_df.groupby("GAME_ID").filter(lambda g: len(g) == 2)["GAME_ID"].nunique()
    return {
        "model": "XGBoost",
        "team_rmse": rmse,
        "winner_accuracy": _winner_accuracy(pred_df),
        "num_team_rows_eval": int(len(pred_df)),
        "num_games_eval": int(paired_games),
    }


def _build_gru_dataset(df: pd.DataFrame, timesteps: int = 5):
    rows = []
    for team, team_df in df.sort_values("GAME_DATE").groupby("TEAM_ABBREVIATION"):
        team_df = team_df.sort_values("GAME_DATE").reset_index(drop=True)
        pts = team_df["PTS"].values.astype(np.float32)
        homes = team_df["HOME"].values.astype(np.float32)

        for i in range(timesteps, len(team_df)):
            seq_pts = pts[i - timesteps:i]
            seq_home = homes[i - timesteps:i]
            x_seq = np.stack([seq_pts, seq_home], axis=1)
            rows.append(
                {
                    "GAME_ID": team_df.loc[i, "GAME_ID"],
                    "TEAM_ABBREVIATION": team,
                    "GAME_DATE": team_df.loc[i, "GAME_DATE"],
                    "y": pts[i],
                    "x": x_seq,
                }
            )

    if not rows:
        raise RuntimeError("No GRU samples created")

    return pd.DataFrame(rows)


def run_gru_experiment(df: pd.DataFrame) -> dict:
    seq_df = _build_gru_dataset(df, timesteps=5)
    train_df, test_df = _split_train_test_by_date(seq_df)

    x_train = np.stack(train_df["x"].values)
    x_test = np.stack(test_df["x"].values)
    y_train = train_df["y"].values.astype(np.float32)
    y_test = test_df["y"].values.astype(np.float32)

    scaler = StandardScaler()
    x_train_2d = x_train.reshape(-1, x_train.shape[-1])
    x_test_2d = x_test.reshape(-1, x_test.shape[-1])
    x_train_scaled = scaler.fit_transform(x_train_2d).reshape(x_train.shape)
    x_test_scaled = scaler.transform(x_test_2d).reshape(x_test.shape)

    model = Sequential([
        GRU(32, input_shape=(x_train_scaled.shape[1], x_train_scaled.shape[2])),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        x_train_scaled,
        y_train,
        validation_split=0.1,
        epochs=30,
        batch_size=32,
        verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
    )

    preds = model.predict(x_test_scaled, verbose=0).reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

    pred_df = test_df[["GAME_ID", "TEAM_ABBREVIATION"]].copy()
    pred_df["actual_points"] = y_test
    pred_df["pred_points"] = preds

    paired_games = pred_df.groupby("GAME_ID").filter(lambda g: len(g) == 2)["GAME_ID"].nunique()
    return {
        "model": "GRU",
        "team_rmse": rmse,
        "winner_accuracy": _winner_accuracy(pred_df),
        "num_team_rows_eval": int(len(pred_df)),
        "num_games_eval": int(paired_games),
    }


def main() -> None:
    base_df = _load_base_data()
    feat_df = _add_features(base_df)

    xgb_result = run_xgboost_experiment(feat_df)
    gru_result = run_gru_experiment(base_df)

    summary = {
        "season": SEASON,
        "games_in_dataset": int(base_df["GAME_ID"].nunique()),
        "results": [xgb_result, gru_result],
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
