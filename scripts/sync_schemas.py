"""Copy the packaged JSON Schemas (source of truth) to the repository-level
``schemas/`` directory."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "packages" / "core-python" / "src" / "khmerocr" / "jsonschema"


def main() -> None:
    target = ROOT / "schemas"
    target.mkdir(exist_ok=True)
    for path in sorted(SOURCE.glob("*.schema.json")):
        shutil.copyfile(path, target / path.name)
        print(f"synced {path.name}")


if __name__ == "__main__":
    main()
