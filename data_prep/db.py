import json
import logging
import sqlite3
import time
from io import StringIO
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)


class OracleCacheDB:
    """SQLite-backed cache for roster and player log data."""

    def __init__(self, db_path: Path | str | None = None):
        if db_path is None:
            db_path = Path.cwd() / "artifacts" / "cache" / "oracle_cache.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug("Initializing OracleCacheDB at %s", self.db_path)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS roster_cache (
                    team_id INTEGER NOT NULL,
                    season TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (team_id, season)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS player_logs_cache (
                    player_id INTEGER NOT NULL,
                    season TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (player_id, season)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dataset_cache (
                    dataset_key TEXT NOT NULL PRIMARY KEY,
                    updated_at REAL NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _is_fresh(updated_at: float, ttl_hours: int | None) -> bool:
        if ttl_hours is None or ttl_hours <= 0:
            return True
        return (time.time() - float(updated_at)) <= ttl_hours * 3600

    @staticmethod
    def _df_to_json(df: pd.DataFrame) -> str:
        return df.to_json(orient="records", date_format="iso")

    @staticmethod
    def _json_to_df(payload_json: str) -> pd.DataFrame:
        if not payload_json:
            return pd.DataFrame()
        return pd.read_json(StringIO(payload_json), orient="records")

    def get_roster(self, team_id: int, season: str, ttl_hours: int | None = None) -> pd.DataFrame | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT updated_at, payload_json FROM roster_cache WHERE team_id=? AND season=?",
                (int(team_id), season),
            ).fetchone()
        if not row:
            logger.debug("Roster cache miss (team_id=%s, season=%s)", team_id, season)
            return None
        updated_at, payload_json = row
        if not self._is_fresh(updated_at, ttl_hours):
            logger.debug("Roster cache stale (team_id=%s, season=%s)", team_id, season)
            return None
        return self._json_to_df(payload_json)

    def upsert_roster(self, team_id: int, season: str, roster_df: pd.DataFrame) -> None:
        payload_json = self._df_to_json(roster_df)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO roster_cache (team_id, season, updated_at, payload_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(team_id, season)
                DO UPDATE SET
                    updated_at=excluded.updated_at,
                    payload_json=excluded.payload_json
                """,
                (int(team_id), season, time.time(), payload_json),
            )

    def get_player_logs(self, player_id: int, season: str, ttl_hours: int | None = None) -> pd.DataFrame | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT updated_at, payload_json FROM player_logs_cache WHERE player_id=? AND season=?",
                (int(player_id), season),
            ).fetchone()
        if not row:
            logger.debug("Player logs cache miss (player_id=%s, season=%s)", player_id, season)
            return None
        updated_at, payload_json = row
        if not self._is_fresh(updated_at, ttl_hours):
            logger.debug("Player logs cache stale (player_id=%s, season=%s)", player_id, season)
            return None
        return self._json_to_df(payload_json)

    def upsert_player_logs(self, player_id: int, season: str, player_logs_df: pd.DataFrame) -> None:
        payload_json = self._df_to_json(player_logs_df)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO player_logs_cache (player_id, season, updated_at, payload_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(player_id, season)
                DO UPDATE SET
                    updated_at=excluded.updated_at,
                    payload_json=excluded.payload_json
                """,
                (int(player_id), season, time.time(), payload_json),
            )

    def get_dataset_df(self, dataset_key: str, ttl_hours: int | None = None) -> pd.DataFrame | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT updated_at, payload_json FROM dataset_cache WHERE dataset_key=?",
                (dataset_key,),
            ).fetchone()
        if not row:
            logger.debug("Dataset cache miss (dataset_key=%s)", dataset_key)
            return None
        updated_at, payload_json = row
        if not self._is_fresh(updated_at, ttl_hours):
            logger.debug("Dataset cache stale (dataset_key=%s)", dataset_key)
            return None
        return self._json_to_df(payload_json)

    def upsert_dataset_df(self, dataset_key: str, dataset_df: pd.DataFrame) -> None:
        payload_json = self._df_to_json(dataset_df)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO dataset_cache (dataset_key, updated_at, payload_json)
                VALUES (?, ?, ?)
                ON CONFLICT(dataset_key)
                DO UPDATE SET
                    updated_at=excluded.updated_at,
                    payload_json=excluded.payload_json
                """,
                (dataset_key, time.time(), payload_json),
            )
