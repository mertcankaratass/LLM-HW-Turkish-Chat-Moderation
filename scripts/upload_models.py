"""Upload the trained models to the Hugging Face Hub (weights are not stored in the GitHub repository).

Needs a one-time `hf auth login` with a write token. Uploads REQUIRED_FILES from models/ and a model card
(hub/MODEL_CARD.md + the current results table), checks that every uploaded file matches the local file
(SHA-256), and prints the commit id to paste into HF_REVISION in src/chatmod/download.py.
"""
import hashlib

from huggingface_hub import HfApi

from chatmod.common import MODELS, RESULTS, ROOT
from chatmod.download import HF_REPO_ID, REQUIRED_FILES

GITHUB_URL = "https://github.com/mertcankaratass/LLM-HW-Turkish-Chat-Moderation"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    missing = [f for f in REQUIRED_FILES if not (MODELS / f).is_file()]
    assert not missing, f"train the models first, missing: {missing}"
    card = (ROOT / "hub" / "MODEL_CARD.md").read_text(encoding="utf-8")
    card = card.replace("{results}", (RESULTS / "results_table.md").read_text(encoding="utf-8").strip())
    assert not GITHUB_URL.startswith("PASTE_"), "set GITHUB_URL first (link shown on the model card)"
    card = card.replace("{github}", GITHUB_URL)

    api = HfApi()
    api.create_repo(HF_REPO_ID, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=HF_REPO_ID, folder_path=MODELS, allow_patterns=REQUIRED_FILES,
                      commit_message="Upload trained models")
    commit = api.upload_file(repo_id=HF_REPO_ID, path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md",
                             commit_message="Update model card")

    # Verify: every file on the Hub has the same SHA-256 as the local file (LFS files) or the same size.
    info = api.model_info(HF_REPO_ID, revision=commit.oid, files_metadata=True)
    remote = {s.rfilename: s for s in info.siblings}
    for name in REQUIRED_FILES:
        local, r = MODELS / name, remote[name]
        if r.lfs is not None:
            assert r.lfs.sha256 == sha256(local), f"checksum mismatch: {name}"
        else:
            assert r.size == local.stat().st_size, f"size mismatch: {name}"
    print(f"All {len(REQUIRED_FILES)} files verified on https://huggingface.co/{HF_REPO_ID}")
    print("HF_REVISION =", commit.oid)


if __name__ == "__main__":
    main()
