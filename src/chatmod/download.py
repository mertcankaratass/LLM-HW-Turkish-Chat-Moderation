"""Download the trained models on first run (weights are not stored in the repository).

ensure_models() does nothing when every required file is already in models/. Otherwise it downloads them:
1. from the Hugging Face Hub model repository HF_REPO_ID, pinned to the commit HF_REVISION (a commit id is a
   hash of the content, so the files are exactly the ones evaluated in results/);
2. if that fails, from the Google Drive mirror: one archive, checked against MODELS_SHA256, then extracted.
Manual alternative (see README): download the files from HF_URL (or the archive from MODELS_URL) into models/.
"""
import hashlib
import zipfile
from pathlib import Path

from chatmod.common import MODELS

HF_REPO_ID = "mertcankaratas/turkish-chat-moderation"
HF_REVISION = "e0d565972eed9e4419339cfdeae47ab65eef7730"  # commit of the uploaded models (scripts/upload_models.py prints it)
HF_URL = f"https://huggingface.co/{HF_REPO_ID}"

DRIVE_FILE_ID = "1YgSwdN9ispbIZBIR4Ocdg_s6bcXh6YTL"  # dist/chatmod-models-v1.zip (scripts/package_models.py)
MODELS_SHA256 = "c2357ae01ca4089e281b67305c304353326f559d237c6aab2b56c4d8eb3a2f96"
MODELS_URL = f"https://drive.google.com/file/d/{DRIVE_FILE_ID}/view"

REQUIRED_FILES = [
    "lexicon.json",
    "tfidf.joblib",
    "w2v.model",
    "ffn.pt",
    "berturk-fp32/config.json",
    "berturk-fp32/model.safetensors",
    "berturk-fp32/threshold.json",
    "berturk-fp32/tokenizer.json",
    "berturk-fp32/tokenizer_config.json",
    "berturk-fp32/special_tokens_map.json",
    "berturk-fp32/vocab.txt",
    "berturk-int8/config.json",
    "berturk-int8/model_int8.pt",
    "berturk-int8/threshold.json",
    "berturk-int8/tokenizer.json",
    "berturk-int8/tokenizer_config.json",
    "berturk-int8/special_tokens_map.json",
    "berturk-int8/vocab.txt",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _missing(models_dir: Path) -> list[str]:
    return [f for f in REQUIRED_FILES if not (models_dir / f).is_file()]


def _fetch_hf(models_dir: Path) -> None:
    from huggingface_hub import constants, snapshot_download

    # Plain HTTPS download, the same on every machine: it needs no optional "hf_xet" package (without it the
    # Hub prints a notice) and avoids the Xet transfer stalls we saw on Windows.
    constants.HF_HUB_DISABLE_XET = True
    snapshot_download(repo_id=HF_REPO_ID, revision=HF_REVISION, local_dir=str(models_dir),
                      allow_patterns=REQUIRED_FILES)


def _fetch(file_id: str, output: Path) -> None:
    import gdown

    gdown.download(id=file_id, output=str(output), quiet=False)


def _download_from_drive(models_dir: Path) -> None:
    archive = models_dir.parent / "chatmod-models.zip.part"
    archive.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading the trained models (one 577 MB archive) from the Google Drive mirror: {MODELS_URL}")
    try:
        _fetch(DRIVE_FILE_ID, archive)
        if not archive.is_file() or _sha256(archive) != MODELS_SHA256:
            raise RuntimeError("Downloaded model archive failed the checksum test; please download the models "
                               f"manually from {HF_URL} or {MODELS_URL} into the models/ folder (see README).")
        with zipfile.ZipFile(archive) as z:
            z.extractall(models_dir)
    finally:
        archive.unlink(missing_ok=True)


def ensure_models(models_dir: Path = MODELS) -> None:
    models_dir = Path(models_dir)
    if not _missing(models_dir):
        return
    models_dir.mkdir(parents=True, exist_ok=True)
    try:
        if HF_REVISION.startswith("PASTE_"):
            raise RuntimeError("no Hugging Face revision configured")
        print(f"Downloading the trained models (650 MB, one time) from the Hugging Face Hub: {HF_URL}")
        _fetch_hf(models_dir)
        if _missing(models_dir):
            raise RuntimeError(f"files missing after download: {_missing(models_dir)}")
    except Exception as err:  # network error, Hub outage, ... -> use the mirror
        print(f"Hugging Face download failed ({err}); trying the Google Drive mirror.")
        _download_from_drive(models_dir)
    missing = _missing(models_dir)
    if missing:
        raise RuntimeError(f"Model download is incomplete, missing: {missing}")
    print("Models ready in", models_dir)
