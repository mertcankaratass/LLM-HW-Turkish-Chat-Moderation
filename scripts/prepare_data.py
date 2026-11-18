"""Download OffensEval-TR 2020 and create the train / validation / test splits.

Source: Coltekin (2020), "A Corpus of Turkish Offensive Language on Social Media", LREC 2020.
https://coltekin.github.io/offensive-turkish/  (annotations licensed CC-BY)

Splits:
- test : the official OffensEval 2020 Turkish test set (3,528 tweets), used ONLY for the final
         comparison of all models.
- train / val : the official training set (31,756 tweets) split 90 / 10, stratified, seed 42.
                val is used for model selection and threshold tuning.
"""
import hashlib
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

URL = "https://coltekin.github.io/offensive-turkish/offenseval2020-turkish.zip"
SHA256 = "7977e96255dbc9b8d14893f1b14cbe3dec53c70358503c062c5a59720ec9c2f2"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
SEED = 42


def read_tsv(path: Path, n_fields: int) -> list[list[str]]:
    # The files use no quoting: split each line at tab characters manually (see dataset README).
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    rows = [line.split("\t") for line in lines[1:] if line.strip()]
    assert all(len(r) == n_fields for r in rows), f"unexpected field count in {path}"
    return rows


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    zip_path = RAW / "offenseval2020-turkish.zip"
    if not zip_path.exists():
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, zip_path)
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    assert digest == SHA256, f"checksum mismatch: {digest}"
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(RAW)

    base = RAW / "offenseval2020-turkish"
    train_rows = read_tsv(base / "offenseval-tr-training-v1" / "offenseval-tr-training-v1.tsv", 3)
    test_rows = read_tsv(base / "offenseval-tr-testset-v1" / "offenseval-tr-testset-v1.tsv", 2)
    gold_lines = (base / "offenseval-tr-testset-v1" / "offenseval-tr-labela-v1.tsv").read_text(
        encoding="utf-8-sig").splitlines()
    gold = dict(line.split(",") for line in gold_lines if line.strip())

    full = pd.DataFrame(train_rows, columns=["id", "text", "label"])
    test = pd.DataFrame(test_rows, columns=["id", "text"])
    test["label"] = test["id"].map(gold)
    assert len(full) == 31756 and len(test) == 3528, (len(full), len(test))
    assert test["label"].notna().all(), "test tweet without a gold label"

    for df in (full, test):
        assert set(df["label"]) == {"NOT", "OFF"}
        df["label"] = (df["label"] == "OFF").astype(int)  # 1 = offensive

    train, val = train_test_split(full, test_size=0.1, stratify=full["label"], random_state=SEED)
    for name, df in (("train", train), ("val", val), ("test", test)):
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name:5s}: {len(df):6d} tweets, offensive = {df['label'].mean():.2%}")


if __name__ == "__main__":
    main()
