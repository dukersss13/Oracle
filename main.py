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


def run_server():
    import uvicorn

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
