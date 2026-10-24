"""Recognizer backend registry.

Backends are selected by the ``recognizer.backend`` key of a model bundle's
manifest. Third-party packages can add backends without modifying this
package by exposing a factory under the ``khmerocr.recognizers`` entry-point
group, e.g. in their ``pyproject.toml``::

    [project.entry-points."khmerocr.recognizers"]
    netra = "khmerocr_netra:create_recognizer"
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

from ..errors import ModelNotAvailableError
from .base import Recognizer

RecognizerFactory = Callable[[Path, Mapping[str, Any]], Recognizer]

ENTRY_POINT_GROUP = "khmerocr.recognizers"

_REGISTRY: dict[str, RecognizerFactory] = {}


def register_recognizer(name: str, factory: RecognizerFactory) -> None:
    _REGISTRY[name] = factory


def available_backends() -> list[str]:
    names = set(_REGISTRY)
    names.update(ep.name for ep in entry_points(group=ENTRY_POINT_GROUP))
    return sorted(names)


def create_recognizer(name: str, model_dir: Path, options: Mapping[str, Any]) -> Recognizer:
    factory = _REGISTRY.get(name)
    if factory is None:
        for ep in entry_points(group=ENTRY_POINT_GROUP):
            if ep.name == name:
                factory = ep.load()
                break
    if factory is None:
        raise ModelNotAvailableError(
            f"Recognizer backend '{name}' is not installed.",
            details={"available_backends": available_backends()},
        )
    return factory(model_dir, options)


def _tesseract_factory(model_dir: Path, options: Mapping[str, Any]) -> Recognizer:
    from .tesseract import TesseractRecognizer

    tessdata_dir = (model_dir / str(options.get("tessdata_dir", "tessdata"))).resolve()
    if not tessdata_dir.is_relative_to(model_dir.resolve()):
        raise ModelNotAvailableError("The tessdata directory must be inside the model bundle.")
    # The binary path deliberately comes from the environment, never from a
    # model manifest, so a model bundle cannot choose which program runs.
    return TesseractRecognizer(
        tessdata_dir=tessdata_dir,
        default_languages=str(options.get("languages", "khm+eng")),
    )


register_recognizer("tesseract", _tesseract_factory)
