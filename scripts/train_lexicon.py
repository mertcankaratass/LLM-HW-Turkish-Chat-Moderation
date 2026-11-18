"""Build the offensive-word lexicon from the training split and tune its thresholds on validation.

Tuned on validation: min_count, min_ratio, and prefix_match (suffix tolerance for Turkish).

The test split is NOT used here. Output: models/lexicon.json, results/lexicon_tuning.json
"""
import json
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

from chatmod.lexicon import LexiconMatcher, build_lexicon

ROOT = Path(__file__).resolve().parents[1]
GRID_MIN_COUNT = [5, 10, 20, 30, 50]
GRID_MIN_RATIO = [0.5, 0.6, 0.7, 0.8, 0.9]


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "processed" / f"{name}.csv", keep_default_na=False)


def main() -> None:
    train, val = load("train"), load("val")
    runs = []
    for prefix_match in (False, True):
        for min_count in GRID_MIN_COUNT:
            for min_ratio in GRID_MIN_RATIO:
                lexicon = build_lexicon(train["text"], train["label"], min_count, min_ratio)
                matcher = LexiconMatcher(lexicon, prefix_match=prefix_match)
                t0 = time.perf_counter()
                pred = [matcher.predict(t) for t in val["text"]]
                runs.append({
                    "prefix_match": prefix_match, "min_count": min_count, "min_ratio": min_ratio,
                    "lexicon_size": len(lexicon),
                    "val_macro_f1": round(f1_score(val["label"], pred, average="macro"), 4),
                    "val_seconds": round(time.perf_counter() - t0, 2),
                })
                print(runs[-1])

    best = max(runs, key=lambda r: r["val_macro_f1"])
    lexicon = sorted(build_lexicon(train["text"], train["label"], best["min_count"], best["min_ratio"]))
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "models" / "lexicon.json").write_text(
        json.dumps({"params": best, "words": lexicon}, ensure_ascii=False, indent=1), encoding="utf-8")
    (ROOT / "results" / "lexicon_tuning.json").write_text(json.dumps(runs, indent=1), encoding="utf-8")
    print("best:", best)


if __name__ == "__main__":
    main()
