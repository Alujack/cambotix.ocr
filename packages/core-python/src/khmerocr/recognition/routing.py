"""Route each region to the recognizer best suited to its script.

Khmer-only models such as seanghay/KhmerOCR cannot read Latin letters or
ASCII digits and return confident Khmer-looking garbage for them, so they
must only see regions the template marks as Khmer. Everything else goes to the
default backend.

A route may list several backends. The first one answers ``recognize_line``;
``recognize_candidates`` yields each in turn so the extractor can fall back
when a read does not validate (e.g. Khmer text over a patterned background
that one engine handles better than another).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

from ..preprocessing.io import ImageArray
from ..schemas import TextRegion
from .base import RecognizedText, Recognizer, RegionHint, Script


class ScriptRouter(Recognizer):
    name = "router"

    def __init__(self, backends: Mapping[str, Recognizer],
                 routes: Mapping[str, str | Sequence[str]], *,
                 default: str, page: str | None = None) -> None:
        self.routes: dict[str, list[str]] = {
            script: [names] if isinstance(names, str) else list(names)
            for script, names in routes.items()
        }
        referenced = {default, page or default}
        for names in self.routes.values():
            if not names:
                raise ValueError("a route must list at least one backend")
            referenced.update(names)
        unknown = referenced - set(backends)
        if unknown:
            raise ValueError(f"routes reference unknown backends: {sorted(unknown)}")
        self.backends = dict(backends)
        self.default = default
        self.page = page or default

    def backends_for(self, script: Script) -> list[Recognizer]:
        return [self.backends[name] for name in self.routes.get(script, [self.default])]

    def recognize_line(self, image: ImageArray, hint: RegionHint | None = None) -> RecognizedText:
        return self.backends_for((hint or RegionHint()).script)[0].recognize_line(image, hint)

    def recognize_candidates(self, image: ImageArray,
                             hint: RegionHint | None = None) -> Iterator[RecognizedText]:
        for backend in self.backends_for((hint or RegionHint()).script):
            yield backend.recognize_line(image, hint)

    def recognize_page(self, image: ImageArray) -> list[TextRegion]:
        return self.backends[self.page].recognize_page(image)

    def describe(self) -> dict[str, str]:
        description = {"backend": self.name, "default": self.default}
        for script, names in sorted(self.routes.items()):
            description[f"route.{script}"] = ",".join(names)
        for name, recognizer in sorted(self.backends.items()):
            description[f"backends.{name}"] = recognizer.describe().get("backend", name)
        return description

    def close(self) -> None:
        for backend in self.backends.values():
            backend.close()
