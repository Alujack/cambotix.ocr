"""Model bundles: manifests, integrity checks, and offline installation.

A model bundle is a directory containing ``manifest.json`` and the files it
lists (weights, language data, vocabularies). Bundles are installed from a
local directory or archive only. Nothing in this package downloads models.

Example ``manifest.json``::

    {
      "model_id": "khmerocr-id-v1",
      "task": "khmer_id_extraction",
      "runtime": "local",
      "recognition_model": "tesseract-tessdata_fast-4.1.0",
      "model_version": "1.0.0",
      "schema_version": "1.0",
      "license": "Apache-2.0",
      "license_approved": false,
      "license_components": [{"name": "tessdata_fast", "license": "Apache-2.0",
                              "url": "https://github.com/tesseract-ocr/tessdata_fast"}],
      "recognizer": {"backend": "tesseract", "options": {"languages": "khm+eng"}},
      "files": [{"path": "tessdata/khm.traineddata", "sha256": "..."}],
      "weights_sha256": "<bundle digest over all files>"
    }
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import ModelNotAvailableError

MANIFEST_FILENAME = "manifest.json"
REQUIRED_KEYS = ("model_id", "task", "runtime", "recognition_model", "model_version",
                 "schema_version", "license", "recognizer", "files", "weights_sha256")


@dataclass(frozen=True)
class ModelFile:
    path: str
    sha256: str


@dataclass(frozen=True)
class ModelManifest:
    model_id: str
    task: str
    runtime: str
    recognition_model: str
    model_version: str
    schema_version: str
    license: str
    license_approved: bool
    recognizer_backend: str
    recognizer_options: dict[str, Any]
    files: tuple[ModelFile, ...]
    weights_sha256: str
    license_components: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelManifest:
        missing = [key for key in REQUIRED_KEYS if key not in data]
        if missing:
            raise ModelNotAvailableError("The model manifest is incomplete.",
                                         details={"missing_keys": missing})
        try:
            recognizer = data["recognizer"]
            return cls(
                model_id=str(data["model_id"]),
                task=str(data["task"]),
                runtime=str(data["runtime"]),
                recognition_model=str(data["recognition_model"]),
                model_version=str(data["model_version"]),
                schema_version=str(data["schema_version"]),
                license=str(data["license"]),
                license_approved=data.get("license_approved") is True,
                recognizer_backend=str(recognizer["backend"]),
                recognizer_options=dict(recognizer.get("options", {})),
                files=tuple(ModelFile(str(f["path"]), str(f["sha256"]).lower())
                            for f in data["files"]),
                weights_sha256=str(data["weights_sha256"]).lower(),
                license_components=tuple(data.get("license_components", [])),
            )
        except (KeyError, TypeError, AttributeError) as exc:
            raise ModelNotAvailableError("The model manifest is malformed.") from exc

    @classmethod
    def load(cls, model_dir: Path) -> ModelManifest:
        path = model_dir / MANIFEST_FILENAME
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ModelNotAvailableError(
                "No model manifest found. Install a model bundle first.",
                details={"model_dir": str(model_dir)},
            ) from exc
        except (OSError, ValueError) as exc:
            raise ModelNotAvailableError("The model manifest could not be read.") from exc
        if not isinstance(data, dict):
            raise ModelNotAvailableError("The model manifest is malformed.")
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "task": self.task,
            "runtime": self.runtime,
            "recognition_model": self.recognition_model,
            "model_version": self.model_version,
            "schema_version": self.schema_version,
            "license": self.license,
            "license_approved": self.license_approved,
            "license_components": list(self.license_components),
            "recognizer": {"backend": self.recognizer_backend,
                           "options": dict(self.recognizer_options)},
            "files": [{"path": f.path, "sha256": f.sha256} for f in self.files],
            "weights_sha256": self.weights_sha256,
        }

    def summary(self) -> dict[str, Any]:
        """Public description, safe to expose over the API."""
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "task": self.task,
            "runtime": self.runtime,
            "recognition_model": self.recognition_model,
            "schema_version": self.schema_version,
            "license": self.license,
            "license_approved": self.license_approved,
            "recognizer_backend": self.recognizer_backend,
            "weights_sha256": self.weights_sha256,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_digest(files: tuple[ModelFile, ...] | list[ModelFile]) -> str:
    """Digest identifying the exact set of files in a bundle."""
    lines = "".join(f"{f.path}\0{f.sha256}\n" for f in sorted(files, key=lambda f: f.path))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def verify_bundle(model_dir: Path, manifest: ModelManifest) -> None:
    """Raise ``ModelNotAvailableError`` unless every listed file is present and
    matches its recorded checksum."""
    root = model_dir.resolve()
    if bundle_digest(manifest.files) != manifest.weights_sha256:
        raise ModelNotAvailableError("The model bundle digest does not match its file list.")
    for entry in manifest.files:
        path = (root / entry.path).resolve()
        if not path.is_relative_to(root):
            raise ModelNotAvailableError("The model manifest references a file outside the bundle.")
        if not path.is_file():
            raise ModelNotAvailableError("A model file is missing.",
                                         details={"path": entry.path})
        if sha256_file(path) != entry.sha256:
            raise ModelNotAvailableError("A model file failed checksum verification.",
                                         details={"path": entry.path})


def build_manifest(model_dir: Path, *, file_paths: list[str], **fields: Any) -> dict[str, Any]:
    """Create manifest data for files already placed in ``model_dir``."""
    files = [ModelFile(p, sha256_file(model_dir / p)) for p in sorted(file_paths)]
    data = dict(fields)
    data["files"] = [{"path": f.path, "sha256": f.sha256} for f in files]
    data["weights_sha256"] = bundle_digest(files)
    data.setdefault("license_approved", False)
    return data


def models_home() -> Path:
    return Path(os.environ.get("KHMEROCR_HOME", Path.home() / ".khmerocr")) / "models"


def resolve_model_dir(model: str | os.PathLike[str], home: Path | None = None) -> Path:
    """Resolve a bundle path, or an installed model ID, to a directory."""
    candidate = Path(model).expanduser()
    if (candidate / MANIFEST_FILENAME).is_file():
        return candidate
    installed = (home or models_home()) / str(model)
    if not candidate.is_absolute() and os.sep not in str(model) and (
            installed / MANIFEST_FILENAME).is_file():
        return installed
    raise ModelNotAvailableError(
        "The model is not installed. Install it from a local bundle with "
        "'khmerocr models install <model_id> --from <bundle>'.",
        details={"model": str(model)},
    )


def list_installed_models(home: Path | None = None) -> list[ModelManifest]:
    root = home or models_home()
    manifests: list[ModelManifest] = []
    if not root.is_dir():
        return manifests
    for child in sorted(root.iterdir()):
        if (child / MANIFEST_FILENAME).is_file():
            try:
                manifests.append(ModelManifest.load(child))
            except ModelNotAvailableError:
                continue
    return manifests


def install_bundle(model_id: str, source: Path, *, home: Path | None = None,
                   force: bool = False) -> Path:
    """Verify and install a bundle from a local directory, ``.tar.gz`` or ``.zip``."""
    if not model_id or any(c in model_id for c in "/\\") or model_id in (".", ".."):
        raise ModelNotAvailableError("Invalid model ID.")
    root = home or models_home()
    target = root / model_id
    if target.exists() and not force:
        raise ModelNotAvailableError("The model is already installed. Use --force to replace it.",
                                     details={"model_id": model_id})
    root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=root, prefix=".install-") as tmp:
        staging = Path(tmp) / "bundle"
        if source.is_dir():
            shutil.copytree(source, staging)
        elif source.is_file():
            _extract_archive(source, staging)
        else:
            raise ModelNotAvailableError("The bundle source does not exist.",
                                         details={"source": str(source)})
        bundle_dir = _bundle_root(staging)
        manifest = ModelManifest.load(bundle_dir)
        if manifest.model_id != model_id:
            raise ModelNotAvailableError("The bundle is for a different model.",
                                         details={"expected": model_id,
                                                  "found": manifest.model_id})
        verify_bundle(bundle_dir, manifest)
        if target.exists():
            shutil.rmtree(target)
        bundle_dir.rename(target)
    return target


def _bundle_root(staging: Path) -> Path:
    if (staging / MANIFEST_FILENAME).is_file():
        return staging
    children = [p for p in staging.iterdir() if p.is_dir()]
    if len(children) == 1 and (children[0] / MANIFEST_FILENAME).is_file():
        return children[0]
    raise ModelNotAvailableError("The bundle does not contain a manifest.json.")


def _extract_archive(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    dest = destination.resolve()
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                if not (dest / name).resolve().is_relative_to(dest):
                    raise ModelNotAvailableError("The bundle archive contains unsafe paths.")
            zf.extractall(dest)  # noqa: S202 - members validated above
        return
    try:
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                if not (member.isfile() or member.isdir()):
                    raise ModelNotAvailableError("The bundle archive contains links or devices.")
                if not (dest / member.name).resolve().is_relative_to(dest):
                    raise ModelNotAvailableError("The bundle archive contains unsafe paths.")
            if hasattr(tarfile, "data_filter"):
                tf.extractall(dest, filter="data")
            else:  # pragma: no cover - Python < 3.10.12
                tf.extractall(dest)  # noqa: S202 - members validated above
    except tarfile.TarError as exc:
        raise ModelNotAvailableError("The bundle archive could not be read.") from exc
