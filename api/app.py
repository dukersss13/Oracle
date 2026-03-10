import asyncio
import json
import logging
import os
import time
import uuid
from threading import Thread

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from nba_api.stats.endpoints import commonteamroster, scoreboardv2
from sse_starlette.sse import EventSourceResponse

from api.schemas import (
    ForecastRequest,
    ForecastResponse,
    PlayerForecast,
    RosterPlayer,
    TeamForecast,
    TeamInfo,
    TodayGameOption,
)
from data_prep.db import OracleCacheDB
from data_prep.gamelogs import nba_teams_info
from models.oracle import Oracle

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Oracle NBA Forecaster")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_forecasts: dict[str, dict] = {}
_db_cache = OracleCacheDB()


@app.get("/api/teams", response_model=list[TeamInfo])
def list_teams():
    df = nba_teams_info.copy()
    return [
        TeamInfo(
            id=int(row["id"]),
            full_name=row["full_name"],
            abbreviation=row["abbreviation"],
            nickname=row["nickname"],
            city=row["city"],
        )
        for _, row in df.iterrows()
    ]


@app.get("/api/roster/{nickname}", response_model=list[RosterPlayer])
def get_roster(nickname: str):
    team_row = nba_teams_info[nba_teams_info["nickname"] == nickname]
    if team_row.empty:
        raise HTTPException(404, f"Team '{nickname}' not found")
    team_id = int(team_row["id"].values[0])

    season = "2024-25"
    cache_ttl_hours = int(os.environ.get("ORACLE_CACHE_TTL_HOURS", "12"))
    cached_db_roster = _db_cache.get_roster(team_id, season, ttl_hours=cache_ttl_hours)
    if cached_db_roster is not None and not cached_db_roster.empty and os.environ.get("ORACLE_REFRESH_CACHE", "0") != "1":
        roster_df = cached_db_roster
        return [
            RosterPlayer(player_name=row["PLAYER"], player_id=int(row["PLAYER_ID"]))
            for _, row in roster_df.iterrows()
        ]

    retries = int(os.environ.get("ORACLE_ROSTER_RETRIES", "3"))
    backoff_seconds = float(os.environ.get("ORACLE_ROSTER_BACKOFF", "1.5"))
    request_timeout = int(os.environ.get("ORACLE_NBA_API_TIMEOUT", "20"))
    last_error = None
    roster_df = None

    for attempt in range(1, retries + 1):
        try:
            roster_df = commonteamroster.CommonTeamRoster(
                team_id=team_id,
                season=season,
                timeout=request_timeout,
            ).get_data_frames()[0][["PLAYER", "PLAYER_ID"]]
            _db_cache.upsert_roster(team_id, season, roster_df)
            break
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(backoff_seconds * attempt)

    if roster_df is None:
        stale_roster = _db_cache.get_roster(team_id, season, ttl_hours=None)
        if stale_roster is not None and not stale_roster.empty:
            roster_df = stale_roster
        else:
            logger.warning("Roster unavailable for team_id=%s; returning empty roster", team_id)
            roster_df = pd.DataFrame(columns=["PLAYER", "PLAYER_ID"])

    return [
        RosterPlayer(player_name=row["PLAYER"], player_id=int(row["PLAYER_ID"]))
        for _, row in roster_df.iterrows()
    ]


@app.get("/api/games/today", response_model=list[TodayGameOption])
def get_todays_games():
    timeout = int(os.environ.get("ORACLE_NBA_API_TIMEOUT", "20"))
    today = pd.Timestamp.now().strftime("%m/%d/%Y")
    try:
        board = scoreboardv2.ScoreboardV2(game_date=today, timeout=timeout)
        header_df = board.game_header.get_data_frame()
    except Exception:
        return []

    if header_df.empty:
        return []

    team_id_to_nickname = {
        int(row["id"]): row["nickname"]
        for _, row in nba_teams_info.iterrows()
    }

    options: list[TodayGameOption] = []
    for _, row in header_df.drop_duplicates(subset=["GAME_ID"]).iterrows():
        game_id = str(row["GAME_ID"])
        home_team_id = int(row["HOME_TEAM_ID"])
        away_team_id = int(row["VISITOR_TEAM_ID"])
        home_team = team_id_to_nickname.get(home_team_id, str(home_team_id))
        away_team = team_id_to_nickname.get(away_team_id, str(away_team_id))
        home_abb = nba_teams_info[nba_teams_info["id"] == home_team_id]["abbreviation"].values[0]
        away_abb = nba_teams_info[nba_teams_info["id"] == away_team_id]["abbreviation"].values[0]
        game_date = pd.to_datetime(row["GAME_DATE_EST"]).strftime("%m-%d-%Y")
        options.append(
            TodayGameOption(
                game_id=game_id,
                game_date=game_date,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                home_abbreviation=home_abb,
                away_abbreviation=away_abb,
                home_team=home_team,
                away_team=away_team,
                label=f"{away_team} @ {home_team}",
            )
        )

    return sorted(options, key=lambda g: g.label)


def _default_features() -> list[str]:
    return [
        "MIN",
        "GAME_DATE_player",
        "FGM",
        "FGA",
        "FG_PCT",
        "FG3M_player",
        "FG3A_player",
        "FG3_PCT_player",
        "FTM",
        "FTA",
        "FT_PCT",
        "HOME",
        "AWAY",
        "REST_DAYS",
        "D_FGM",
        "D_FGA",
        "D_FG_PCT",
        "FG3M_opp_defense",
        "FG3A_opp_defense",
        "FG3_PCT_opp_defense",
        "NS_FG3_PCT",
        "FG2M",
        "FG2A",
        "FG2_PCT",
        "NS_FG2_PCT",
        "FGM_LT_10",
        "FGA_LT_10",
        "LT_10_PCT",
        "NS_LT_10_PCT",
        "E_PACE",
        "E_DEF_RATING",
        "PTS",
    ]


def _df_to_team_forecast(df: pd.DataFrame, label: str) -> TeamForecast:
    players = []
    total_f = total_a = 0
    for _, row in df.iterrows():
        if row["PLAYER_NAME"] == "Total":
            total_f = int(row["FORECASTED_POINTS"])
            total_a = int(row["ACTUAL_POINTS"])
            continue
        players.append(
            PlayerForecast(
                player_name=row["PLAYER_NAME"],
                forecasted_points=int(row["FORECASTED_POINTS"]),
                actual_points=int(row["ACTUAL_POINTS"]),
            )
        )
    return TeamForecast(
        team_name=label,
        players=players,
        total_forecasted=total_f,
        total_actual=total_a,
    )


def _run_forecast_in_thread(forecast_id: str, req: ForecastRequest):
    state = _forecasts[forecast_id]
    loop: asyncio.AbstractEventLoop = state["_loop"]

    def progress_callback(msg: dict):
        asyncio.run_coroutine_threadsafe(state["_queue"].put(msg), loop)

    try:
        cache_strategy = os.environ.get("ORACLE_CACHE_STRATEGY", "cache-only").lower()
        fetch_new_data = os.environ.get("ORACLE_FETCH_NEW_DATA", "0") == "1"
        game_details = {
            "home_team": req.home_team,
            "away_team": req.away_team,
            "game_date": req.game_date,
            "new_game": True,
        }
        oracle_config = {
            "model": req.model,
            "features": _default_features(),
            "holdout": req.holdout,
            "fetch_new_data": fetch_new_data,
            "cache_strategy": cache_strategy,
            "cache_ttl_hours": int(os.environ.get("ORACLE_CACHE_TTL_HOURS", "12")),
            "save_file": False,
        }

        oracle = Oracle(
            game_details=game_details,
            oracle_config=oracle_config,
            active_players_dict=req.lineup,
            progress_callback=progress_callback,
            non_interactive=True,
            force_refresh_cache=os.environ.get("ORACLE_REFRESH_CACHE", "0") == "1",
        )

        home_df, away_df = oracle.run()
        state["home_df"] = home_df
        state["away_df"] = away_df
        state["status"] = "completed"
        asyncio.run_coroutine_threadsafe(state["_queue"].put({"type": "completed"}), loop)
    except Exception as exc:
        logger.exception("Forecast failed")
        state["status"] = "error"
        state["error"] = str(exc)
        asyncio.run_coroutine_threadsafe(state["_queue"].put({"type": "error", "message": str(exc)}), loop)


@app.post("/api/forecast")
async def create_forecast(req: ForecastRequest):
    forecast_id = uuid.uuid4().hex[:12]
    loop = asyncio.get_event_loop()
    _forecasts[forecast_id] = {
        "status": "running",
        "_queue": asyncio.Queue(),
        "_loop": loop,
        "home_df": None,
        "away_df": None,
    }
    thread = Thread(target=_run_forecast_in_thread, args=(forecast_id, req), daemon=True)
    thread.start()
    return {"forecast_id": forecast_id}


@app.get("/api/forecast/{forecast_id}/stream")
async def stream_forecast(forecast_id: str):
    state = _forecasts.get(forecast_id)
    if state is None:
        raise HTTPException(404, "Forecast not found")

    async def event_gen():
        queue: asyncio.Queue = state["_queue"]
        while True:
            msg = await queue.get()
            yield {"data": json.dumps(msg)}
            if msg.get("type") in ("completed", "error"):
                break

    return EventSourceResponse(event_gen())


@app.get("/api/forecast/{forecast_id}", response_model=ForecastResponse)
def get_forecast(forecast_id: str):
    state = _forecasts.get(forecast_id)
    if state is None:
        raise HTTPException(404, "Forecast not found")

    resp = ForecastResponse(forecast_id=forecast_id, status=state["status"])
    if state["status"] == "completed" and state.get("home_df") is not None:
        resp.home_forecast = _df_to_team_forecast(state["home_df"], "Home")
        resp.away_forecast = _df_to_team_forecast(state["away_df"], "Away")
    return resp


_ui_dist = os.path.join(os.path.dirname(__file__), "..", "ui", "dist")
if os.path.isdir(_ui_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(_ui_dist, "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = os.path.join(_ui_dist, full_path)
        if full_path and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(_ui_dist, "index.html"))
