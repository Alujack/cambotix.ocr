"""Release gate: fail unless every model in the release registry has verified,
approved licensing and recorded checksums.

Usage:
    python scripts/check_release.py [--bundle DIR ...]

Without ``--bundle`` only ``models/manifest.json`` is checked. With bundles,
each bundle's manifest must match its registry entry and pass checksum
verification.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDERS = {"selected-after-benchmark", "record-during-release",
                "verified-license-required", "", "TODO", "TBD"}
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def check_registry(registry: dict) -> list[str]:
    problems: list[str] = []
    models = registry.get("models", [])
    if not models:
        problems.append("models/manifest.json lists no models")
    for entry in models:
        model_id = entry.get("model_id", "<missing model_id>")
        for key in ("recognition_model", "weights_sha256", "license", "model_version"):
            if str(entry.get(key, "")).strip() in PLACEHOLDERS:
                problems.append(f"{model_id}: '{key}' is still a placeholder")
        if not SHA256.match(str(entry.get("weights_sha256", ""))):
            problems.append(f"{model_id}: weights_sha256 is not a SHA-256 digest")
        if entry.get("license_approved") is not True:
            problems.append(f"{model_id}: license has not been approved")
        if not entry.get("license_approved_by"):
            problems.append(f"{model_id}: license_approved_by is not recorded")
    return problems


def check_bundle(bundle: Path, registry: dict) -> list[str]:
    sys.path.insert(0, str(ROOT / "packages" / "core-python" / "src"))
    from khmerocr.errors import KhmerOCRError
    from khmerocr.models import ModelManifest, verify_bundle

    try:
        manifest = ModelManifest.load(bundle)
        verify_bundle(bundle, manifest)
    except KhmerOCRError as exc:
        return [f"{bundle}: {exc.message}"]
    entries = {e["model_id"]: e for e in registry.get("models", [])}
    entry = entries.get(manifest.model_id)
    if entry is None:
        return [f"{bundle}: model {manifest.model_id} is not in the registry"]
    problems = []
    if entry.get("weights_sha256") != manifest.weights_sha256:
        problems.append(f"{manifest.model_id}: bundle digest does not match the registry")
    if not manifest.license_approved:
        problems.append(f"{manifest.model_id}: bundle manifest license is not approved")
    if not manifest.license_components:
        problems.append(f"{manifest.model_id}: bundle lists no license_components")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, action="append", default=[])
    parser.add_argument("--registry", type=Path, default=ROOT / "models" / "manifest.json")
    args = parser.parse_args()

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    problems = check_registry(registry)
    for bundle in args.bundle:
        problems.extend(check_bundle(bundle, registry))
    if problems:
        print("Release blocked:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print("Release checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
