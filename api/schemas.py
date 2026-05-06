from typing import Optional

from pydantic import BaseModel


class TeamInfo(BaseModel):
    id: int
    full_name: str
    abbreviation: str
    nickname: str
    city: str


class RosterPlayer(BaseModel):
    player_name: str
    player_id: int


class TodayGameOption(BaseModel):
    game_id: str
    game_date: str
    home_team_id: int
    away_team_id: int
    home_abbreviation: str
    away_abbreviation: str
    home_team: str
    away_team: str
    label: str


class ForecastRequest(BaseModel):
    home_team: str
    away_team: str
    game_date: str
    model: str = "TRANSFORMER"  # NN, XGBOOST, or TRANSFORMER
    holdout: bool = False
    lineup: Optional[dict[str, dict[str, Optional[float]]]] = None


class PlayerForecast(BaseModel):
    player_name: str
    forecasted_points: int
    actual_points: int


class TeamForecast(BaseModel):
    team_name: str
    players: list[PlayerForecast]
    total_forecasted: int
    total_actual: int


class ForecastResponse(BaseModel):
    forecast_id: str
    status: str
    home_forecast: Optional[TeamForecast] = None
    away_forecast: Optional[TeamForecast] = None


class InjuryEntry(BaseModel):
    name: str
    team: str
    date: str
    description: str
