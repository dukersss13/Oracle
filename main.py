import atexit
import logging
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import webbrowser


_UI_DEV_PROCESS: subprocess.Popen | None = None
logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=os.environ.get("ORACLE_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def run_cli():
    from models.oracle import Oracle

    start = time.time()
    logger.info("Starting Oracle in CLI mode")
    oracle = Oracle()
    oracle.run()
    end = time.time()
    logger.info("CLI run completed in %.2f minutes", (end - start) / 60)


def _refresh_training_data() -> None:
    """Ensure rosters, game logs, and the Transformer model are up to date."""
    from scripts.preload_training_cache import fetch_latest_rosters_to_current_date
    from scripts.train_transformer import is_model_stale, train_model

    # 1. Refresh current-season rosters
    try:
        logger.info("Refreshing current-season rosters")
        refresh_summary = fetch_latest_rosters_to_current_date()
        logger.info(
            "Roster refresh finished (season=%s, teams=%s, failures=%s)",
            refresh_summary["season"],
            refresh_summary["teams_processed"],
            refresh_summary["failures"],
        )
    except Exception:
        logger.exception("Roster refresh failed; continuing with existing cache")

    # 2. Refresh league game logs (team box scores for Transformer features)
    try:
        logger.info("Refreshing league team game logs for Transformer features")
        from data_prep.team_features import _fetch_league_team_logs
        from pathlib import Path
        from data_prep.locker_room import _current_nba_season

        cache_dir = Path("artifacts/cache")
        season = _current_nba_season()
        season_tag = season.replace("-", "_")
        cache_path = cache_dir / f"leaguegamelog_{season_tag}.csv"

        # Only remove cached file if it's older than the configured max age
        max_cache_hours = int(os.environ.get("ORACLE_CACHE_TTL_HOURS", "12"))
        if cache_path.exists():
            import datetime as _dt
            age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
            if age_hours > max_cache_hours:
                logger.info("League game log cache is %.1fh old (> %dh); refreshing", age_hours, max_cache_hours)
                cache_path.unlink()
            else:
                logger.info("League game log cache is fresh (%.1fh old); skipping refresh", age_hours)

        _fetch_league_team_logs(season, cache_dir)
        logger.info("League team game logs refreshed for %s", season)
    except Exception:
        logger.exception("League game log refresh failed; continuing with existing cache")

    # 3. Train Transformer if model is missing or stale (>24h)
    max_age = int(os.environ.get("ORACLE_MODEL_MAX_AGE_HOURS", "24"))
    if is_model_stale(max_age_hours=max_age):
        try:
            logger.info("Transformer model is stale or missing; training now")
            metrics = train_model(verbose=False)
            logger.info(
                "Transformer training complete (RMSE=%.2f, MAE=%.2f, samples=%d)",
                metrics["test_rmse"],
                metrics["test_mae"],
                metrics["train_samples"],
            )
        except Exception:
            logger.exception("Transformer training failed; model may not be available")
    else:
        logger.info("Transformer model is up to date; skipping training")


def run_server():
    import uvicorn
    from scripts.preload_training_cache import fetch_latest_rosters_to_current_date

    _refresh_training_data()

    browser_port = 8000
    if not _ui_dist_exists() and _start_ui_dev_server():
        browser_port = 5173
        logger.info("UI dist not found; using Vite dev server on port %s", browser_port)
    else:
        logger.info("Serving built UI assets via FastAPI on port %s", browser_port)

    _launch_browser_async(host="127.0.0.1", port=browser_port, path="/")
    logger.info("Starting FastAPI server on 0.0.0.0:8000")
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=False)


def _ui_dist_exists() -> bool:
    root_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.isdir(os.path.join(root_dir, "ui", "dist"))


def _start_ui_dev_server() -> bool:
    global _UI_DEV_PROCESS
    if _UI_DEV_PROCESS and _UI_DEV_PROCESS.poll() is None:
        return True

    npm = shutil.which("npm")
    if not npm:
        return False

    root_dir = os.path.dirname(os.path.abspath(__file__))
    ui_dir = os.path.join(root_dir, "ui")
    if not os.path.isdir(ui_dir):
        return False

    try:
        logger.info("Starting UI dev server with npm")
        _UI_DEV_PROCESS = subprocess.Popen(
            [npm, "run", "dev", "--", "--host", "127.0.0.1", "--port", "5173", "--strictPort"],
            cwd=ui_dir,
        )
    except Exception:
        logger.exception("Failed to start UI dev server")
        _UI_DEV_PROCESS = None
        return False

    def _cleanup_ui_process() -> None:
        if _UI_DEV_PROCESS and _UI_DEV_PROCESS.poll() is None:
            _UI_DEV_PROCESS.terminate()

    atexit.register(_cleanup_ui_process)
    return True


def _launch_browser_async(host: str, port: int, path: str = "/") -> None:
    if os.environ.get("ORACLE_OPEN_BROWSER", "1") in {"0", "false", "False"}:
        logger.info("Browser auto-open is disabled")
        return

    url = f"http://{host}:{port}{path}"

    def _open_url_with_fallback(target_url: str) -> bool:
        try:
            if webbrowser.open(target_url):
                return True
        except Exception:
            logger.warning("webbrowser.open failed for %s", target_url, exc_info=True)

        try:
            if sys.platform == "darwin":
                return subprocess.call(["open", target_url]) == 0
            if os.name == "nt":
                return os.system(f'start "" "{target_url}"') == 0
            return subprocess.call(["xdg-open", target_url]) == 0
        except Exception:
            logger.warning("OS fallback browser launch failed for %s", target_url, exc_info=True)
            return False

    def _wait_and_open() -> None:
        for _ in range(120):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.1)
                if sock.connect_ex((host, port)) == 0:
                    opened = _open_url_with_fallback(url)
                    if opened:
                        logger.info("Opened browser at %s", url)
                    else:
                        logger.warning("Could not auto-open browser. Open this URL manually: %s", url)
                    return
            time.sleep(0.1)

        logger.warning("Server did not become reachable in time. Open manually: %s", url)

    threading.Thread(target=_wait_and_open, daemon=True).start()


if __name__ == "__main__":
    _configure_logging()
    if "--cli" in sys.argv:
        run_cli()
    else:
        run_server()
