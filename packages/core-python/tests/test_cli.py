from __future__ import annotations

import io
import json
import sys

from khmerocr.adapters.cli import main

from .conftest import make_bundle


def test_check_command(tmp_path, card_photo_png, capsys) -> None:
    image = tmp_path / "card.png"
    image.write_bytes(card_photo_png)
    assert main(["check", str(image)]) == 0
    assert json.loads(capsys.readouterr().out)["usable"] is True


def test_invalid_image_reports_error_envelope(tmp_path, capsys) -> None:
    image = tmp_path / "card.png"
    image.write_bytes(b"nope")
    assert main(["check", str(image)]) == 1
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "INVALID_IMAGE"


def test_models_install_list_verify_and_extract(tmp_path, monkeypatch, card_photo_png,
                                                capsys) -> None:
    monkeypatch.setenv("KHMEROCR_HOME", str(tmp_path / "home"))
    source = make_bundle(tmp_path / "src")
    assert main(["models", "install", "test-model", "--from", str(source)]) == 0
    assert main(["models", "list"]) == 0
    assert main(["models", "verify", "test-model"]) == 0
    capsys.readouterr()

    image = tmp_path / "card.png"
    image.write_bytes(card_photo_png)
    assert main(["extract-id", str(image), "--model", "test-model"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["fields"]["id_number"]["value"] == "000000001"


def test_khmer_output_survives_a_legacy_code_page(tmp_path, monkeypatch,
                                                  card_photo_png) -> None:
    # Windows pipes default to cp1252, which cannot encode Khmer.
    monkeypatch.setenv("KHMEROCR_HOME", str(tmp_path / "home"))
    assert main(["models", "install", "test-model", "--from",
                 str(make_bundle(tmp_path / "src"))]) == 0
    image = tmp_path / "card.png"
    image.write_bytes(card_photo_png)
    raw = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw, encoding="cp1252"))
    assert main(["extract-id", str(image), "--model", "test-model"]) == 0
    sys.stdout.flush()
    result = json.loads(raw.getvalue().decode("utf-8"))
    assert result["fields"]["full_name_km"]["value"] == "គំរូ សាកល្បង"


def test_install_without_source_refuses_to_download(capsys) -> None:
    assert main(["models", "install", "khmerocr-id-v1"]) == 1
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "MODEL_NOT_AVAILABLE" and "never downloads" in error["message"]
