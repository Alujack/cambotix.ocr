"""Test helpers: deterministic recognizers and synthetic fixtures.

Nothing in this package is used by the engine at runtime.
"""

from .recognizers import ScriptedRecognizer, scripted_factory
from .synthetic import (
    SyntheticIdentity,
    add_blur,
    add_glare,
    place_on_background,
    random_identity,
    render_card,
    to_png_bytes,
)

__all__ = [
    "ScriptedRecognizer",
    "SyntheticIdentity",
    "add_blur",
    "add_glare",
    "place_on_background",
    "random_identity",
    "render_card",
    "scripted_factory",
    "to_png_bytes",
]
