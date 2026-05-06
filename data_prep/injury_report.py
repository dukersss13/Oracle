import logging
import os
from datetime import datetime

import requests
import pandas as pd
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

url = "https://www.basketball-reference.com/friv/injuries.cgi"


def _fetch_injury_report_dict() -> dict[str, list[str]]:
    injury_data = {"name": [], "team": [], "date": [], "injury_description": []}
    if os.environ.get("ORACLE_DISABLE_INJURY_REPORT", "0") == "1":
        logger.info("Injury report fetch disabled via ORACLE_DISABLE_INJURY_REPORT=1")
        return injury_data

    timeout_seconds = int(os.environ.get("ORACLE_INJURY_TIMEOUT", "15"))

    try:
        response = requests.get(url, timeout=timeout_seconds)
    except requests.RequestException:
        logger.warning("Failed to fetch injury report from %s", url, exc_info=True)
        return injury_data

    if response.status_code != 200:
        logger.warning("Injury report request returned status=%s", response.status_code)
        return injury_data

    soup = BeautifulSoup(response.content, "html.parser")
    injury_table = soup.find("table", {"id": "injuries"})
    if not injury_table:
        logger.warning("Injury report table not found in response")
        return injury_data

    rows = injury_table.find_all("tr")
    for row in rows[1:]:
        columns = row.find_all("td")
        if len(columns) < 3:
            continue
        player_header = row.find("th")
        if player_header is None:
            continue

        injury_data["name"].append(player_header.text.strip())
        injury_data["team"].append(columns[0].text.strip())
        injury_data["date"].append(columns[1].text.strip())
        injury_data["injury_description"].append(columns[2].text.strip())

    logger.info("Fetched injury report entries=%s", len(injury_data["name"]))
    return injury_data

def adjust_datetime_format(date_string: str) -> str:
    try:
        date_object = datetime.strptime(date_string, "%a, %b %d, %Y")
        return date_object.strftime("%m-%d-%Y")
    except Exception:
        logger.debug("Unable to parse injury date: %s", date_string)
        return date_string

def get_injury_status(injury_report: pd.DataFrame) -> pd.DataFrame:
    """
    Filter and return players who are ruled out
    """
    injury_report["Out"] = ["Out" in description for description in injury_report["injury_description"]]
    injury_report = injury_report[injury_report["Out"]==True]

    return injury_report


def fetch_current_injuries() -> list[dict]:
    """Fetch a fresh injury report and return a list of dicts.

    Each dict has keys: name, team, date, description, out.
    Only players marked 'Out' are included.
    """
    raw = _fetch_injury_report_dict()
    df = pd.DataFrame(raw)
    if df.empty:
        return []
    if "date" in df.columns:
        df["date"] = df["date"].apply(adjust_datetime_format)
    df["out"] = df["injury_description"].str.contains("Out", case=False, na=False)
    out_df = df[df["out"]].copy()
    return [
        {
            "name": row["name"],
            "team": row["team"],
            "date": row["date"],
            "description": row["injury_description"],
        }
        for _, row in out_df.iterrows()
    ]


injury_report = _fetch_injury_report_dict()
injury_report = pd.DataFrame(injury_report)
if not injury_report.empty and "date" in injury_report.columns:
    injury_report["date"] = injury_report["date"].apply(adjust_datetime_format)
injury_report = get_injury_status(injury_report)
