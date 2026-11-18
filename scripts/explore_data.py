"""Basic statistics of the prepared splits (used for the 'Data' slide). Writes results/data_stats.json."""
import json
import re
from pathlib import Path

import warnings

import pandas as pd

warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression")

ROOT = Path(__file__).resolve().parents[1]
splits = {n: pd.read_csv(ROOT / "data" / "processed" / f"{n}.csv", keep_default_na=False)
          for n in ("train", "val", "test")}

stats = {}
for name, df in splits.items():
    words = df["text"].str.split().str.len()
    stats[name] = {
        "tweets": len(df),
        "offensive": int(df["label"].sum()),
        "offensive_ratio": round(float(df["label"].mean()), 4),
        "words_mean": round(float(words.mean()), 1),
        "words_p95": int(words.quantile(0.95)),
        "words_max": int(words.max()),
        "with_@USER": int(df["text"].str.contains("@USER").sum()),
        "with_URL": int(df["text"].str.contains("URL").sum()),
        "with_repeated_letters(3+)": int(df["text"].str.contains(r"(\w)\1\1", regex=True).sum()),
        "with_digit_inside_word": int(df["text"].str.contains(r"[a-zçğıöşü]\d[a-zçğıöşü]",
                                                              flags=re.I, regex=True).sum()),
        "duplicate_texts": int(df["text"].duplicated().sum()),
    }

train_texts = set(splits["train"]["text"])
stats["overlap"] = {
    "val_texts_in_train": int(splits["val"]["text"].isin(train_texts).sum()),
    "test_texts_in_train": int(splits["test"]["text"].isin(train_texts).sum()),
}

out = ROOT / "results" / "data_stats.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(stats, indent=2, ensure_ascii=False))
