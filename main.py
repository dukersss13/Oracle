import atexit
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import webbrowser


_UI_DEV_PROCESS: subprocess.Popen | None = None


def run_cli():
    from models.oracle import Oracle

    start = time.time()
    oracle = Oracle()
    oracle.run()
    end = time.time()
    print(f"Total solve time E2E: {round((end - start) / 60)} minutes")


def run_server():
    import uvicorn

    browser_port = 8000
    if not _ui_dist_exists() and _start_ui_dev_server():
        browser_port = 5173

    _launch_browser_async(host="127.0.0.1", port=browser_port, path="/")
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
        _UI_DEV_PROCESS = subprocess.Popen(
            [npm, "run", "dev", "--", "--host", "127.0.0.1", "--port", "5173", "--strictPort"],
            cwd=ui_dir,
        )
    except Exception:
        _UI_DEV_PROCESS = None
        return False

    def _cleanup_ui_process() -> None:
        if _UI_DEV_PROCESS and _UI_DEV_PROCESS.poll() is None:
            _UI_DEV_PROCESS.terminate()

    atexit.register(_cleanup_ui_process)
    return True


def _launch_browser_async(host: str, port: int, path: str = "/") -> None:
    if os.environ.get("ORACLE_OPEN_BROWSER", "1") in {"0", "false", "False"}:
        return

    def _wait_and_open() -> None:
        for _ in range(120):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.1)
                if sock.connect_ex((host, port)) == 0:
                    webbrowser.open(f"http://{host}:{port}{path}")
                    return
            time.sleep(0.1)

    threading.Thread(target=_wait_and_open, daemon=True).start()


if __name__ == "__main__":
    if "--cli" in sys.argv:
        run_cli()
    else:
        run_server()
