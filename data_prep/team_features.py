"""
Feature engineering for team-level Transformer model.

Builds sequence features (team's last N games) and static features
(roster quality + opponent matchup context) for the TeamTransformer.
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

from data_prep.db import OracleCacheDB

logger = logging.getLogger(__name__)

# ── Column sets ─────────────────────────────────────────────────────────────

TEAM_BOX_COLS = [
    "SEASON_ID", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME",
    "GAME_ID", "GAME_DATE", "MATCHUP", "WL", "MIN",
    "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB",
    "AST", "STL", "BLK", "TOV", "PF", "PTS", "PLUS_MINUS",
]

SEQUENCE_FEATURES = [
    "PTS", "REB", "AST", "TOV", "STL", "BLK",
    "FG_PCT", "FG3_PCT", "FT_PCT", "FGA",
    "HOME", "REST_DAYS",
    "OPP_DEF_RATING", "OPP_PACE", "PLUS_MINUS",
]

STATIC_FEATURES = [
    "HOME", "REST_DAYS",
    "OPP_DEF_RATING", "OPP_PACE",
    "OPP_PPG_ALLOWED", "OPP_FG_PCT_ALLOWED", "OPP_FG3_PCT_ALLOWED",
    "ROSTER_PPG", "ROSTER_RPG", "ROSTER_APG",
    "ROSTER_TOPG", "ROSTER_TS_PCT",
]

NUM_SEQ_FEATURES = len(SEQUENCE_FEATURES)    # 15
NUM_STATIC_FEATURES = len(STATIC_FEATURES)   # 12

# ── Helpers ─────────────────────────────────────────────────────────────────


def _fetch_league_team_logs(season: str, cache_dir: Path | None = None) -> pd.DataFrame:
    """Fetch team-level box scores for a season from nba_api, with CSV cache."""
    season_tag = season.replace("-", "_")
    if cache_dir is not None:
        cache_path = cache_dir / f"leaguegamelog_{season_tag}.csv"
        if cache_path.exists():
            logger.info("Loading cached league team logs for %s", season)
            return pd.read_csv(cache_path)

    retries = int(os.environ.get("ORACLE_GAMELOG_RETRIES", "3"))
    backoff = float(os.environ.get("ORACLE_GAMELOG_BACKOFF", "2.0"))
    timeout = int(os.environ.get("ORACLE_NBA_API_TIMEOUT", "60"))

    logger.info("Fetching league team game logs from NBA API for %s", season)
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            df = leaguegamelog.LeagueGameLog(
                season=season,
                player_or_team_abbreviation="T",
                timeout=timeout,
            ).get_data_frames()[0]

            if cache_dir is not None:
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_path = cache_dir / f"leaguegamelog_{season_tag}.csv"
                df.to_csv(cache_path, index=False)

            return df
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                import time as _time
                logger.warning(
                    "League game log fetch failed for %s (attempt %s/%s); retrying in %.0fs",
                    season, attempt, retries, backoff * attempt,
                )
                _time.sleep(backoff * attempt)

    raise RuntimeError(f"Failed to fetch league game logs for {season}") from last_error


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add HOME, REST_DAYS, and opponent identifiers to team box scores."""
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df["HOME"] = (~df["MATCHUP"].astype(str).str.contains("@", regex=False)).astype(int)

    # Extract opponent abbreviation from matchup string
    df["OPP_ABBR"] = df["MATCHUP"].str.extract(r"(?:vs\.|@)\s*(\w+)", expand=False)

    df = df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)

    # REST_DAYS per team
    df["PREV_GAME_DATE"] = df.groupby("TEAM_ABBREVIATION")["GAME_DATE"].shift(1)
    df["REST_DAYS"] = (df["GAME_DATE"] - df["PREV_GAME_DATE"]).dt.days.fillna(3).clip(upper=7)

    return df


def _merge_opponent_context(df: pd.DataFrame, all_logs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Merge opponent E_DEF_RATING and E_PACE from all_logs into the team box score df.

    ``all_logs`` is the consolidated defensive-stats dataset already used by
    Oracle (``data/all_logs.csv``).  If it is not supplied or does not contain
    the required columns we fall back to league-wide constants.
    """
    df = df.copy()

    if all_logs is not None and {"E_DEF_RATING", "E_PACE", "TEAM_ID", "SEASON_YEAR"}.issubset(all_logs.columns):
        # Build a per-team-season lookup (one row per team, most recent value)
        opp_ctx = (
            all_logs.sort_values("GAME_DATE", ascending=False)
            .drop_duplicates(subset=["TEAM_ID", "SEASON_YEAR"], keep="first")
            [["TEAM_ID", "SEASON_YEAR", "E_DEF_RATING", "E_PACE"]]
            .rename(columns={"TEAM_ID": "OPP_TEAM_ID", "E_DEF_RATING": "OPP_DEF_RATING", "E_PACE": "OPP_PACE"})
        )
    else:
        opp_ctx = None

    # We need OPP_TEAM_ID to merge — build from abbreviation mapping
    abbr_to_id = df.drop_duplicates("TEAM_ABBREVIATION").set_index("TEAM_ABBREVIATION")["TEAM_ID"].to_dict()
    df["OPP_TEAM_ID"] = df["OPP_ABBR"].map(abbr_to_id)

    # Build SEASON_YEAR from SEASON_ID (e.g. "22024" -> "2024-25")
    if "SEASON_YEAR" not in df.columns and "SEASON_ID" in df.columns:
        def _season_id_to_year(sid):
            s = str(sid)
            if len(s) >= 5:
                start_year = int(s[1:])
                return f"{start_year}-{(start_year + 1) % 100:02d}"
            return None
        df["SEASON_YEAR"] = df["SEASON_ID"].apply(_season_id_to_year)

    if opp_ctx is not None:
        df = df.merge(opp_ctx, on=["OPP_TEAM_ID", "SEASON_YEAR"], how="left")
    else:
        df["OPP_DEF_RATING"] = 110.0  # league average fallback
        df["OPP_PACE"] = 100.0

    df["OPP_DEF_RATING"] = df["OPP_DEF_RATING"].fillna(110.0)
    df["OPP_PACE"] = df["OPP_PACE"].fillna(100.0)

    return df


def _compute_opponent_rolling_defense(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Compute rolling defensive stats for opponents: PPG allowed, FG% allowed, FG3% allowed.

    These are used as static features for the *upcoming* game.
    We compute them on the opponent's past games (not the team's own games).
    """
    df = df.copy()

    # Build per-team rolling stats (PPG allowed = opponent's PTS)
    # We need the *opponent's* stats from their own games
    # First, compute per-team rolling averages
    team_rolling = (
        df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])
        .groupby("TEAM_ABBREVIATION")
        .rolling(window, min_periods=1, on="GAME_DATE")
        .agg({"PTS": "mean", "FG_PCT": "mean", "FG3_PCT": "mean"})
        .reset_index()
        .rename(columns={"PTS": "_TEAM_ROLL_PPG", "FG_PCT": "_TEAM_ROLL_FG_PCT",
                          "FG3_PCT": "_TEAM_ROLL_FG3_PCT"})
    )
    team_rolling = team_rolling[["TEAM_ABBREVIATION", "GAME_DATE",
                                  "_TEAM_ROLL_PPG", "_TEAM_ROLL_FG_PCT", "_TEAM_ROLL_FG3_PCT"]]

    # Merge opponent's rolling stats as "what the opponent gives up"
    # OPP_PPG_ALLOWED = opponent team's average PTS scored against them
    # Approximation: use opponent's own PPG as a proxy for pace/context,
    # and OPP_DEF_RATING for actual defensive quality
    # For "PPG allowed" we look at the opponent's *defensive* perspective:
    # = average PTS scored by teams *playing against* the opponent
    # Simplification: use the opponent team's own rolling PPG as a feature
    # (captures offensive pace context; DEF_RATING handles actual defense)

    df = df.merge(
        team_rolling.rename(columns={
            "TEAM_ABBREVIATION": "OPP_ABBR",
            "_TEAM_ROLL_PPG": "OPP_PPG_ALLOWED",
            "_TEAM_ROLL_FG_PCT": "OPP_FG_PCT_ALLOWED",
            "_TEAM_ROLL_FG3_PCT": "OPP_FG3_PCT_ALLOWED",
        }),
        on=["OPP_ABBR", "GAME_DATE"],
        how="left",
    )

    # Fill missing (first few games of season) with league averages
    df["OPP_PPG_ALLOWED"] = df["OPP_PPG_ALLOWED"].fillna(df["PTS"].mean())
    df["OPP_FG_PCT_ALLOWED"] = df["OPP_FG_PCT_ALLOWED"].fillna(df["FG_PCT"].mean())
    df["OPP_FG3_PCT_ALLOWED"] = df["OPP_FG3_PCT_ALLOWED"].fillna(df["FG3_PCT"].mean())

    return df


# ── Roster aggregate computation ───────────────────────────────────────────


def compute_roster_aggregates(
    player_logs: dict[int, pd.DataFrame],
    active_player_ids: list[int],
    game_date: str | pd.Timestamp,
    window: int = 10,
) -> dict[str, float]:
    """Compute roster-level rolling aggregates for active players.

    Parameters
    ----------
    player_logs : dict[int, pd.DataFrame]
        Mapping of player_id -> their game logs DataFrame.
        Each must have columns: GAME_DATE, PTS, REB, AST, TOV, FGA, FTA, MIN.
    active_player_ids : list[int]
        Player IDs in tonight's lineup.
    game_date : str or Timestamp
        The date of the upcoming game (only use games *before* this date).
    window : int
        Number of recent games per player to average over.

    Returns
    -------
    dict with keys: ROSTER_PPG, ROSTER_RPG, ROSTER_APG, ROSTER_TOPG, ROSTER_TS_PCT
    """
    game_date = pd.Timestamp(game_date)
    totals = {"ppg": 0.0, "rpg": 0.0, "apg": 0.0, "topg": 0.0}
    ts_numer = 0.0
    ts_denom = 0.0
    active_count = 0

    for pid in active_player_ids:
        logs = player_logs.get(pid)
        if logs is None or logs.empty:
            continue

        logs = logs.copy()
        logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"], errors="coerce")
        logs = logs.dropna(subset=["GAME_DATE"])
        logs = logs[logs["GAME_DATE"] < game_date].sort_values("GAME_DATE", ascending=False)

        if logs.empty:
            continue

        recent = logs.head(window)
        active_count += 1

        totals["ppg"] += recent["PTS"].mean() if "PTS" in recent.columns else 0.0
        totals["rpg"] += recent["REB"].mean() if "REB" in recent.columns else 0.0
        totals["apg"] += recent["AST"].mean() if "AST" in recent.columns else 0.0
        totals["topg"] += recent["TOV"].mean() if "TOV" in recent.columns else 0.0

        # True Shooting %: PTS / (2 * (FGA + 0.44 * FTA))
        pts_sum = recent["PTS"].sum() if "PTS" in recent.columns else 0.0
        fga_sum = recent["FGA"].sum() if "FGA" in recent.columns else 0.0
        fta_sum = recent["FTA"].sum() if "FTA" in recent.columns else 0.0
        min_sum = recent["MIN"].sum() if "MIN" in recent.columns else 1.0
        tsa = 2 * (fga_sum + 0.44 * fta_sum)

        ts_numer += pts_sum * max(min_sum, 1.0)
        ts_denom += tsa * max(min_sum, 1.0) if tsa > 0 else max(min_sum, 1.0)

    roster_ts = (ts_numer / ts_denom) if ts_denom > 0 else 0.55  # league average fallback

    return {
        "ROSTER_PPG": totals["ppg"],
        "ROSTER_RPG": totals["rpg"],
        "ROSTER_APG": totals["apg"],
        "ROSTER_TOPG": totals["topg"],
        "ROSTER_TS_PCT": roster_ts,
    }


# ── Training dataset builder ───────────────────────────────────────────────


def build_enriched_team_logs(
    seasons: list[str],
    all_logs_df: pd.DataFrame | None = None,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch + enrich team-level box scores for all specified seasons.

    Returns a DataFrame with all SEQUENCE_FEATURES + opponent context columns,
    sorted by (TEAM_ABBREVIATION, GAME_DATE).
    """
    if cache_dir is None:
        cache_dir = Path("artifacts/cache")

    frames = []
    for season in seasons:
        df = _fetch_league_team_logs(season, cache_dir)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = _add_derived_columns(combined)
    combined = _merge_opponent_context(combined, all_logs_df)
    combined = _compute_opponent_rolling_defense(combined)
    combined = combined.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)

    return combined


def build_training_dataset(
    enriched_logs: pd.DataFrame,
    seq_len: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X_seq, X_static, y) arrays from enriched team logs.

    For each game, X_seq is the team's previous ``seq_len`` games,
    X_static is the context for that game, and y is the team's PTS.

    Parameters
    ----------
    enriched_logs : pd.DataFrame
        Output of ``build_enriched_team_logs()``.
    seq_len : int
        Number of previous games to include in the sequence.

    Returns
    -------
    X_seq : ndarray of shape (N, seq_len, NUM_SEQ_FEATURES)
    X_static : ndarray of shape (N, NUM_STATIC_FEATURES)
    y : ndarray of shape (N,)
    """
    X_seq_list = []
    X_static_list = []
    y_list = []

    for team, group in enriched_logs.groupby("TEAM_ABBREVIATION"):
        group = group.sort_values("GAME_DATE").reset_index(drop=True)

        # Ensure all sequence feature columns exist
        for col in SEQUENCE_FEATURES:
            if col not in group.columns:
                group[col] = 0.0

        seq_data = group[SEQUENCE_FEATURES].values.astype(np.float32)
        pts = group["PTS"].values.astype(np.float32)

        # Static features for each game
        static_cols_present = [c for c in STATIC_FEATURES if c in group.columns]
        static_cols_missing = [c for c in STATIC_FEATURES if c not in group.columns]
        static_data = group[static_cols_present].values.astype(np.float32)
        if static_cols_missing:
            zeros = np.zeros((len(group), len(static_cols_missing)), dtype=np.float32)
            static_data = np.hstack([static_data, zeros])

        for i in range(seq_len, len(group)):
            x_seq = seq_data[i - seq_len:i]
            x_static = static_data[i]
            target = pts[i]

            # Skip samples with NaN
            if np.isnan(x_seq).any() or np.isnan(x_static).any() or np.isnan(target):
                continue

            X_seq_list.append(x_seq)
            X_static_list.append(x_static)
            y_list.append(target)

    X_seq = np.array(X_seq_list, dtype=np.float32)
    X_static = np.array(X_static_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    logger.info("Built training dataset: X_seq=%s, X_static=%s, y=%s", X_seq.shape, X_static.shape, y.shape)
    return X_seq, X_static, y


def build_inference_inputs(
    enriched_logs: pd.DataFrame,
    team_abbr: str,
    game_date: str | pd.Timestamp,
    is_home: bool,
    opp_def_rating: float,
    opp_pace: float,
    opp_ppg_allowed: float,
    opp_fg_pct_allowed: float,
    opp_fg3_pct_allowed: float,
    roster_aggs: dict[str, float],
    seq_len: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Build inference inputs for a single upcoming game.

    Returns
    -------
    X_seq : ndarray of shape (1, seq_len, NUM_SEQ_FEATURES)
    X_static : ndarray of shape (1, NUM_STATIC_FEATURES)
    """
    game_date = pd.Timestamp(game_date)
    team_games = enriched_logs[enriched_logs["TEAM_ABBREVIATION"] == team_abbr].copy()
    team_games = team_games[team_games["GAME_DATE"] < game_date].sort_values("GAME_DATE")

    if len(team_games) < seq_len:
        # Pad with repeated earliest game
        pad_count = seq_len - len(team_games)
        if team_games.empty:
            # Absolute fallback: zeros
            x_seq = np.zeros((seq_len, NUM_SEQ_FEATURES), dtype=np.float32)
        else:
            padding = pd.concat([team_games.iloc[:1]] * pad_count, ignore_index=True)
            team_games = pd.concat([padding, team_games], ignore_index=True)
            x_seq = team_games[SEQUENCE_FEATURES].values[-seq_len:].astype(np.float32)
    else:
        x_seq = team_games[SEQUENCE_FEATURES].values[-seq_len:].astype(np.float32)

    # Compute REST_DAYS for upcoming game
    if not team_games.empty:
        last_game_date = team_games["GAME_DATE"].iloc[-1]
        rest_days = min((game_date - last_game_date).days, 7)
    else:
        rest_days = 3

    x_static = np.array([
        float(is_home),
        float(rest_days),
        opp_def_rating,
        opp_pace,
        opp_ppg_allowed,
        opp_fg_pct_allowed,
        opp_fg3_pct_allowed,
        roster_aggs.get("ROSTER_PPG", 0.0),
        roster_aggs.get("ROSTER_RPG", 0.0),
        roster_aggs.get("ROSTER_APG", 0.0),
        roster_aggs.get("ROSTER_TOPG", 0.0),
        roster_aggs.get("ROSTER_TS_PCT", 0.55),
    ], dtype=np.float32)

    np.nan_to_num(x_seq, copy=False)
    np.nan_to_num(x_static, copy=False)

    return x_seq[np.newaxis, :, :], x_static[np.newaxis, :]
