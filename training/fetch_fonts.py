"""Download the pinned OFL training fonts listed in fonts.json.

Fonts come from github.com/google/fonts at a fixed commit. Each family's
OFL.txt is downloaded alongside, and a lock file records every file's
SHA-256 so a training run can be reproduced exactly.

    python training/fetch_fonts.py            # -> training/data/fonts/
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
RAW = "https://raw.githubusercontent.com/google/fonts/{commit}/{path}"


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "khmerocr-training/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main() -> int:
    spec = json.loads((HERE / "fonts.json").read_text(encoding="utf-8"))
    out = HERE / "data" / "fonts"
    lock: dict[str, dict[str, str]] = {}
    licenses: set[str] = set()
    for entry in spec["fonts"]:
        path = entry["path"]
        target = out / path
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            url = RAW.format(commit=spec["commit"], path=urllib.parse.quote(path))
            target.write_bytes(fetch(url))
            time.sleep(0.05)
        lock[path] = {"sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
        # Repo paths are URL paths; Path would join them with "\" on Windows.
        licenses.add(str(PurePosixPath(path).parent / "OFL.txt"))
    for path in sorted(licenses):
        target = out / path
        if target.is_file():
            continue
        try:
            target.write_bytes(fetch(RAW.format(commit=spec["commit"], path=path)))
        except urllib.error.HTTPError:
            # Some families (e.g. Tinos) declare the license only in METADATA.pb.
            metadata = str(PurePosixPath(path).parent / "METADATA.pb")
            data = fetch(RAW.format(commit=spec["commit"], path=metadata))
            if b'license: "OFL"' not in data:
                raise RuntimeError(f"no OFL declaration for {path}") from None
            (out / metadata).write_bytes(data)
    lock_doc = {"commit": spec["commit"], "files": lock}
    (out / "fonts.lock.json").write_text(json.dumps(lock_doc, indent=2) + "\n")
    print(f"{len(lock)} fonts and {len(licenses)} license files in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
