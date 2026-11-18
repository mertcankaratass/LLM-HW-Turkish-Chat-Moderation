"""Pack every trained model into one archive for Google Drive (weights are not stored in the repository).

Output: dist/chatmod-models-v1.zip (+ its SHA-256, to paste into src/chatmod/download.py)
"""
import hashlib
import zipfile
from pathlib import Path

from chatmod.common import MODELS, ROOT
from chatmod.download import REQUIRED_FILES


def main() -> None:
    out = ROOT / "dist" / "chatmod-models-v1.zip"
    out.parent.mkdir(exist_ok=True)
    missing = [f for f in REQUIRED_FILES if not (MODELS / f).is_file()]
    assert not missing, f"train the models first, missing: {missing}"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in REQUIRED_FILES:
            z.write(MODELS / name, arcname=name)
    with zipfile.ZipFile(out) as z:
        assert z.testzip() is None and sorted(z.namelist()) == sorted(REQUIRED_FILES)
    sha = hashlib.sha256(Path(out).read_bytes()).hexdigest()
    print(f"{out}  {out.stat().st_size / 2**20:.1f} MB")
    print("SHA-256:", sha)


if __name__ == "__main__":
    main()
