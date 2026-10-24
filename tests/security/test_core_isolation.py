"""The core package must not depend on the API layer or network clients."""

from __future__ import annotations

import subprocess
import sys

FORBIDDEN = ["fastapi", "starlette", "uvicorn", "khmerocr_api", "requests", "httpx",
             "urllib3", "aiohttp"]


def test_core_import_does_not_load_api_or_http_clients() -> None:
    code = (
        "import sys, khmerocr, khmerocr.engine, khmerocr.adapters.cli, khmerocr.models\n"
        f"loaded = [m for m in {FORBIDDEN!r} if m in sys.modules]\n"
        "print(','.join(loaded))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True).stdout.strip()
    assert out == "", f"core imported forbidden modules: {out}"


def test_no_telemetry_or_upload_code_in_core() -> None:
    from pathlib import Path

    import khmerocr

    root = Path(khmerocr.__file__).parent
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for needle in ("urllib.request", "http.client", "requests.", "httpx", "socket."):
            assert needle not in text, f"{path.name} references {needle}"
