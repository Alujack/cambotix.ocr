"""Versioned document layout templates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Literal, cast

from ..errors import UnsupportedDocumentError
from ..recognition.base import RegionHint, Script

_SCRIPTS = ("khmer", "latin", "digits", "mixed")


@dataclass(frozen=True)
class FieldSpec:
    name: str
    type: str
    # Normalized (x, y, width, height) relative to the canonical card size.
    box: tuple[float, float, float, float]
    script: Script
    critical: bool
    source: str
    labels: tuple[str, ...]
    languages: str | None = None
    char_whitelist: str | None = None
    pattern: str | None = None

    def hint(self) -> RegionHint:
        return RegionHint(name=self.name, script=self.script, languages=self.languages,
                          char_whitelist=self.char_whitelist)

    def options(self) -> dict[str, Any]:
        return {"pattern": self.pattern} if self.pattern else {}


@dataclass(frozen=True)
class DocumentTemplate:
    template_id: str
    template_version: str
    document_type: str
    verified: bool
    canonical_size: tuple[int, int]
    fields: tuple[FieldSpec, ...]

    @property
    def critical_fields(self) -> list[str]:
        return [f.name for f in self.fields if f.critical]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentTemplate:
        try:
            fields = tuple(_field_from_dict(f) for f in data["fields"])
            width, height = data["canonical_size"]
            return cls(
                template_id=str(data["template_id"]),
                template_version=str(data["template_version"]),
                document_type=str(data["document_type"]),
                verified=bool(data["verified"]),
                canonical_size=(int(width), int(height)),
                fields=fields,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise UnsupportedDocumentError("The document template is malformed.") from exc

    @classmethod
    def from_file(cls, path: Path) -> DocumentTemplate:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_builtin_template(template_id: str) -> DocumentTemplate:
    resource = resources.files("khmerocr.documents") / "templates" / f"{template_id}.json"
    if not resource.is_file():
        raise UnsupportedDocumentError(f"Unknown document template '{template_id}'.")
    return DocumentTemplate.from_dict(json.loads(resource.read_text(encoding="utf-8")))


def _field_from_dict(data: dict[str, Any]) -> FieldSpec:
    box = [float(v) for v in data["box"]]
    if len(box) != 4 or not all(0.0 <= v <= 1.0 for v in box):
        raise ValueError("box must contain four values in [0, 1]")
    script = data.get("script", "mixed")
    if script not in _SCRIPTS:
        raise ValueError(f"unknown script {script!r}")
    return FieldSpec(
        name=str(data["name"]),
        type=str(data["type"]),
        box=(box[0], box[1], box[2], box[3]),
        script=cast(Literal["khmer", "latin", "digits", "mixed"], script),
        critical=bool(data.get("critical", False)),
        source=str(data.get("source", "front_region")),
        labels=tuple(data.get("labels", [])),
        languages=data.get("languages"),
        char_whitelist=data.get("char_whitelist"),
        pattern=data.get("pattern"),
    )
