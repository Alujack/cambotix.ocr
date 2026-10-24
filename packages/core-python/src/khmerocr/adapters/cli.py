"""``khmerocr`` command-line interface.

Results are written to stdout as JSON. Errors are written to stderr using the
same error envelope as the REST API, with exit status 1.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from .._version import __version__
from ..errors import KhmerOCRError, ModelNotAvailableError
from ..models import (
    ModelManifest,
    install_bundle,
    list_installed_models,
    models_home,
    resolve_model_dir,
    verify_bundle,
)

DEFAULT_MODEL = "khmerocr-id-v1"


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 2
    try:
        output = args.handler(args)
    except KhmerOCRError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False), file=sys.stderr)
        return 1
    if output is not None:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="khmerocr", description="Local Khmer/English OCR.")
    parser.add_argument("--version", action="version", version=f"khmerocr {__version__}")
    sub = parser.add_subparsers(metavar="command")

    models = sub.add_parser("models", help="Manage installed model bundles.")
    models_sub = models.add_subparsers(metavar="action")
    p = models_sub.add_parser("list", help="List installed models.")
    p.set_defaults(handler=_models_list)
    p = models_sub.add_parser("install", help="Install a model bundle from a local path.")
    p.add_argument("model_id")
    p.add_argument("--from", dest="source", type=Path,
                   help="Bundle directory, .tar.gz or .zip. Models are never downloaded.")
    p.add_argument("--force", action="store_true", help="Replace an installed model.")
    p.set_defaults(handler=_models_install)
    p = models_sub.add_parser("verify", help="Verify an installed model's checksums.")
    p.add_argument("model", nargs="?", default=_default_model())
    p.set_defaults(handler=_models_verify)

    for name, handler, help_text in (
        ("check", _check, "Check image quality."),
        ("recognize", _recognize, "Recognize general text."),
        ("extract-id", _extract_id, "Extract Cambodian National ID fields."),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("image", type=Path)
        p.add_argument("--model", default=_default_model(),
                       help="Model ID or bundle directory (default: %(default)s).")
        if name == "extract-id":
            p.add_argument("--include-text-regions", action="store_true")
        p.set_defaults(handler=handler)

    p = sub.add_parser("serve", help="Run the local REST API (requires khmerocr-api).")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model", default=_default_model())
    p.set_defaults(handler=_serve)
    return parser


def _default_model() -> str:
    return os.environ.get("KHMEROCR_MODEL", DEFAULT_MODEL)


def _models_list(args: argparse.Namespace) -> dict[str, Any]:
    return {"models_home": str(models_home()),
            "models": [m.summary() for m in list_installed_models()]}


def _models_install(args: argparse.Namespace) -> dict[str, Any]:
    if args.source is None:
        raise ModelNotAvailableError(
            "Specify the bundle to install with --from. KhmerOCR never downloads models; "
            "obtain the bundle separately and verify its source before installing."
        )
    target = install_bundle(args.model_id, args.source, force=args.force)
    manifest = ModelManifest.load(target)
    return {"installed": manifest.summary(), "path": str(target)}


def _models_verify(args: argparse.Namespace) -> dict[str, Any]:
    path = resolve_model_dir(args.model)
    manifest = ModelManifest.load(path)
    verify_bundle(path, manifest)
    return {"verified": True, "model": manifest.summary()}


def _check(args: argparse.Namespace) -> dict[str, Any]:
    from ..config import EngineConfig
    from ..pipeline import check_quality, decode

    config = EngineConfig()
    return check_quality(decode(args.image, config), config).to_dict()


def _recognize(args: argparse.Namespace) -> dict[str, Any]:
    from ..engine import KhmerOCR

    with KhmerOCR.from_pretrained(args.model) as ocr:
        return ocr.recognize(args.image).to_dict()


def _extract_id(args: argparse.Namespace) -> dict[str, Any]:
    from ..engine import KhmerOCR

    with KhmerOCR.from_pretrained(args.model) as ocr:
        result = ocr.extract_id(args.image)
        return result.to_dict(include_text_regions=args.include_text_regions)


def _serve(args: argparse.Namespace) -> None:
    try:
        import khmerocr_api  # noqa: F401
        import uvicorn
    except ImportError:
        print("The REST API is not installed: pip install khmerocr-api", file=sys.stderr)
        raise SystemExit(1) from None
    # Engine logs carry codes, counts and timings only, never document content.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s:     %(name)s: %(message)s")
    os.environ["KHMEROCR_MODEL"] = args.model
    uvicorn.run("khmerocr_api.main:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
