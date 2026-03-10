import os
import time

import pandas as pd
from nba_api.stats.endpoints import commonteamroster, playergamelog

from data_prep.db import OracleCacheDB
from data_prep.gamelogs import consolidate_all_game_logs, nba_teams_info, update_data

SEASONS = ["2024-25", "2023-24", "2022-23"]


def _retry_fetch(fn, retries: int, backoff_seconds: float, label: str):
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                print(f"WARN: {label} failed (attempt {attempt}/{retries}), retrying...")
                time.sleep(backoff_seconds * attempt)
    raise RuntimeError(f"Failed: {label}") from last_error


def preload_all_logs_dataset(db: OracleCacheDB) -> None:
    """Ensure consolidated training data for last 3 seasons is persisted in DB."""
    fetch_new_data = os.environ.get("ORACLE_PRELOAD_REFRESH_ALL_LOGS", "1") == "1"
    if fetch_new_data:
        print("Refreshing seasonal source files from nba_api...")
        try:
            update_data(SEASONS)
            consolidate_all_game_logs(SEASONS, SEASONS)
        except Exception as exc:
            print(f"WARN: seasonal refresh failed, using existing local all_logs.csv ({exc})")

    all_logs = pd.read_csv("data/all_logs.csv", low_memory=False)
    all_logs = all_logs[all_logs["SEASON_YEAR"].isin(SEASONS)]
    db.upsert_dataset_df(f"all_logs_{'_'.join(SEASONS)}", all_logs)
    db.upsert_dataset_df("static_team_info", nba_teams_info.reset_index(drop=True))
    print(f"Cached dataset rows: all_logs={len(all_logs)}, static_team_info={len(nba_teams_info)}")


def preload_rosters_and_player_logs(db: OracleCacheDB) -> None:
    """Fetch rosters + player logs for all teams and cache into SQLite."""
    retries = int(os.environ.get("ORACLE_PRELOAD_RETRIES", "3"))
    backoff = float(os.environ.get("ORACLE_PRELOAD_BACKOFF", "1.0"))
    timeout = int(os.environ.get("ORACLE_NBA_API_TIMEOUT", "20"))

    processed_players = set()
    total_rosters = 0
    total_player_logs = 0

    for _, team in nba_teams_info.iterrows():
        team_id = int(team["id"])
        season = SEASONS[0]

        def _get_roster():
            return commonteamroster.CommonTeamRoster(
                team_id=team_id,
                season=season,
                timeout=timeout,
            ).get_data_frames()[0][["PLAYER", "PLAYER_ID"]]

        try:
            roster_df = _retry_fetch(_get_roster, retries, backoff, f"roster team_id={team_id}")
        except Exception as exc:
            print(f"WARN: skipping roster team_id={team_id}: {exc}")
            continue

        db.upsert_roster(team_id, season, roster_df)
        total_rosters += len(roster_df)

        for _, row in roster_df.iterrows():
            player_id = int(row["PLAYER_ID"])
            if player_id in processed_players:
                continue
            processed_players.add(player_id)

            for player_season in SEASONS:
                def _get_logs():
                    return playergamelog.PlayerGameLog(
                        player_id=player_id,
                        season=player_season,
                        season_type_all_star="Regular Season",
                        timeout=timeout,
                    ).get_data_frames()[0]

                try:
                    logs_df = _retry_fetch(
                        _get_logs,
                        retries,
                        backoff,
                        f"player_logs player_id={player_id} season={player_season}",
                    )
                    db.upsert_player_logs(player_id, player_season, logs_df)
                    total_player_logs += len(logs_df)
                except Exception as exc:
                    print(f"WARN: skipping player_id={player_id} season={player_season}: {exc}")

    print(
        "Preload complete: "
        f"teams={len(nba_teams_info)}, roster_rows={total_rosters}, "
        f"players={len(processed_players)}, player_log_rows={total_player_logs}"
    )


def main() -> None:
    db = OracleCacheDB()
    preload_all_logs_dataset(db)
    preload_rosters_and_player_logs(db)
    print("DB preload finished.")


if __name__ == "__main__":
    main()
