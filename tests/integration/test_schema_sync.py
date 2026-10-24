"""Repository-level schemas must match the copies bundled in the package."""

from __future__ import annotations

import json
from pathlib import Path

from khmerocr.jsonschema import SCHEMA_FILES, load_schema

REPO_SCHEMAS = Path(__file__).resolve().parents[2] / "schemas"


def test_repository_schemas_match_packaged_schemas() -> None:
    for filename in SCHEMA_FILES.values():
        repo = json.loads((REPO_SCHEMAS / filename).read_text(encoding="utf-8"))
        assert repo == load_schema(filename), (
            f"schemas/{filename} drifted; run scripts/sync_schemas.py")


def test_model_registry_entries_have_required_keys() -> None:
    registry = json.loads((REPO_SCHEMAS.parent / "models" / "manifest.json").read_text())
    required = {"model_id", "task", "runtime", "recognition_model", "weights_sha256",
                "license", "model_version", "schema_version"}
    for entry in registry["models"]:
        assert required <= set(entry)
