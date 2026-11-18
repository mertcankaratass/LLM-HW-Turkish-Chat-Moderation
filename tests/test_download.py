import hashlib
import zipfile

import pytest

from chatmod import download


def make_zip(path, files):
    with zipfile.ZipFile(path, "w") as z:
        for name in files:
            z.writestr(name, f"content of {name}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_files(models_dir, files):
    for name in files:
        (models_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (models_dir / name).write_text(f"content of {name}")


@pytest.fixture
def setup(tmp_path, monkeypatch):
    models = tmp_path / "models"
    archive = tmp_path / "source.zip"
    sha = make_zip(archive, download.REQUIRED_FILES)
    calls = []

    def fake_hf(models_dir):  # stands in for the Hugging Face Hub download
        calls.append("hf")
        write_files(models_dir, download.REQUIRED_FILES)

    def fake_drive(file_id, output):  # stands in for the Google Drive download
        calls.append("drive")
        output.write_bytes(archive.read_bytes())

    monkeypatch.setattr(download, "_fetch_hf", fake_hf)
    monkeypatch.setattr(download, "_fetch", fake_drive)
    monkeypatch.setattr(download, "MODELS_SHA256", sha)
    monkeypatch.setattr(download, "HF_REVISION", "abc123")
    return models, calls


def test_downloads_from_hugging_face_when_models_are_missing(setup):
    models, calls = setup
    download.ensure_models(models)
    assert calls == ["hf"]
    assert all((models / f).is_file() for f in download.REQUIRED_FILES)


def test_does_nothing_when_models_are_present(setup):
    models, calls = setup
    download.ensure_models(models)
    download.ensure_models(models)
    assert calls == ["hf"]  # second call: no download


def test_falls_back_to_drive_when_hugging_face_fails(setup, monkeypatch):
    models, calls = setup

    def failing_hf(models_dir):
        calls.append("hf")
        raise OSError("network down")

    monkeypatch.setattr(download, "_fetch_hf", failing_hf)
    download.ensure_models(models)
    assert calls == ["hf", "drive"]
    assert all((models / f).is_file() for f in download.REQUIRED_FILES)
    assert not list(models.parent.glob("*.part"))  # temporary archive removed


def test_falls_back_to_drive_when_hugging_face_download_is_incomplete(setup, monkeypatch):
    models, calls = setup
    monkeypatch.setattr(download, "_fetch_hf", lambda d: write_files(d, download.REQUIRED_FILES[:3]))
    download.ensure_models(models)
    assert calls == ["drive"]
    assert all((models / f).is_file() for f in download.REQUIRED_FILES)


def test_rejects_a_corrupted_drive_download(setup, monkeypatch):
    models, _ = setup
    monkeypatch.setattr(download, "_fetch_hf", lambda d: (_ for _ in ()).throw(OSError("down")))
    monkeypatch.setattr(download, "MODELS_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="checksum"):
        download.ensure_models(models)
    assert not (models / "lexicon.json").exists()
