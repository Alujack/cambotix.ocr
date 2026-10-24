from __future__ import annotations

import json
import tarfile

import pytest
from khmerocr import KhmerOCR, ModelNotAvailableError
from khmerocr.models import (
    ModelManifest,
    install_bundle,
    list_installed_models,
    resolve_model_dir,
    verify_bundle,
)

from .conftest import make_bundle


def test_from_pretrained_loads_verified_bundle(tmp_path, card_photo_png) -> None:
    bundle = make_bundle(tmp_path)
    with KhmerOCR.from_pretrained(bundle) as engine:
        result = engine.extract_id(card_photo_png)
    assert result.fields["id_number"].value == "000000001"
    assert result.metadata["model_id"] == "test-model"


def test_tampered_weights_are_rejected(tmp_path) -> None:
    bundle = make_bundle(tmp_path)
    (bundle / "weights" / "model.bin").write_bytes(b"tampered")
    with pytest.raises(ModelNotAvailableError, match="checksum"):
        KhmerOCR.from_pretrained(bundle)


def test_manifest_file_list_tampering_is_rejected(tmp_path) -> None:
    bundle = make_bundle(tmp_path)
    data = json.loads((bundle / "manifest.json").read_text())
    data["files"] = []
    (bundle / "manifest.json").write_text(json.dumps(data))
    with pytest.raises(ModelNotAvailableError, match="digest"):
        verify_bundle(bundle, ModelManifest.load(bundle))


def test_missing_bundle_never_downloads(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KHMEROCR_HOME", str(tmp_path))
    with pytest.raises(ModelNotAvailableError, match="not installed"):
        KhmerOCR.from_pretrained("khmerocr-id-v1")


def test_unknown_backend(tmp_path) -> None:
    bundle = make_bundle(tmp_path)
    data = json.loads((bundle / "manifest.json").read_text())
    data["recognizer"]["backend"] = "does-not-exist"
    (bundle / "manifest.json").write_text(json.dumps(data))
    with pytest.raises(ModelNotAvailableError, match="not installed"):
        KhmerOCR.from_pretrained(bundle)


def test_install_from_directory_and_resolve_by_id(tmp_path) -> None:
    source = make_bundle(tmp_path / "src")
    home = tmp_path / "home"
    target = install_bundle("test-model", source, home=home)
    assert resolve_model_dir("test-model", home=home) == target
    assert [m.model_id for m in list_installed_models(home)] == ["test-model"]
    with pytest.raises(ModelNotAvailableError, match="already installed"):
        install_bundle("test-model", source, home=home)
    install_bundle("test-model", source, home=home, force=True)


def test_install_from_tarball(tmp_path) -> None:
    source = make_bundle(tmp_path / "src")
    archive = tmp_path / "bundle.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(source, arcname="test-model")
    target = install_bundle("test-model", archive, home=tmp_path / "home")
    assert (target / "manifest.json").is_file()


def test_install_rejects_wrong_model_id(tmp_path) -> None:
    source = make_bundle(tmp_path / "src")
    with pytest.raises(ModelNotAvailableError, match="different model"):
        install_bundle("other-model", source, home=tmp_path / "home")


def test_install_rejects_path_traversal_archive(tmp_path) -> None:
    evil = tmp_path / "evil.txt"
    evil.write_text("x")
    archive = tmp_path / "evil.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(evil, arcname="../../escape.txt")
    with pytest.raises(ModelNotAvailableError, match="unsafe"):
        install_bundle("test-model", archive, home=tmp_path / "home")
    assert not (tmp_path / "escape.txt").exists()


def test_manifest_cannot_reference_files_outside_bundle(tmp_path) -> None:
    bundle = make_bundle(tmp_path)
    data = json.loads((bundle / "manifest.json").read_text())
    data["files"][0]["path"] = "../outside.bin"
    from khmerocr.models import ModelFile, bundle_digest

    data["weights_sha256"] = bundle_digest([ModelFile(f["path"], f["sha256"])
                                            for f in data["files"]])
    (bundle / "manifest.json").write_text(json.dumps(data))
    with pytest.raises(ModelNotAvailableError, match="outside"):
        verify_bundle(bundle, ModelManifest.load(bundle))
