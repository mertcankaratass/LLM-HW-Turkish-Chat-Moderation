"""Paired bootstrap test of macro-F1 differences on the test set (is a difference in the results table real?).

Resamples the 3,528 test messages with replacement (N_BOOT times, seed 42); every model is scored on the same
resample, with its validation-tuned threshold from results/results.json. Reports the observed difference, its
95 % percentile interval and a two-sided bootstrap p-value.
Input: results/test_predictions.csv, results/results.json. Output: results/significance.json, .md
"""
import json

import numpy as np
import pandas as pd

from chatmod.common import RESULTS, SEED

N_BOOT = 10_000
PAIRS = [("bert_fp32", "bert_int8"), ("bert_int8", "tfidf"), ("bert_fp32", "tfidf"),
         ("tfidf", "ffn"), ("ffn", "lexicon")]


def macro_f1(y: np.ndarray, y_hat: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Macro-F1 of each bootstrap resample; idx has shape (n_boot, n)."""
    yt, yp = y[idx], y_hat[idx]
    tp = (yt & yp).sum(1)
    fp = (~yt & yp).sum(1)
    fn = (yt & ~yp).sum(1)
    tn = (~yt & ~yp).sum(1)
    f1_pos = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    f1_neg = 2 * tn / np.maximum(2 * tn + fn + fp, 1)
    return (f1_pos + f1_neg) / 2


def main() -> None:
    results = {m["id"]: m for m in json.loads((RESULTS / "results.json").read_text(encoding="utf-8"))["models"]}
    preds = pd.read_csv(RESULTS / "test_predictions.csv")
    y = preds["label"].to_numpy().astype(bool)
    hats = {k: (preds[f"p_{k}"] >= m["threshold"]).to_numpy() for k, m in results.items()}
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(y), size=(N_BOOT, len(y)))
    full = np.arange(len(y))[None, :]
    rows = []
    for a, b in PAIRS:
        observed = float(macro_f1(y, hats[a], full)[0] - macro_f1(y, hats[b], full)[0])
        diff = macro_f1(y, hats[a], idx) - macro_f1(y, hats[b], idx)
        lo, hi = np.percentile(diff, [2.5, 97.5])
        p = min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))
        rows.append({"a": a, "b": b, "model_a": results[a]["model"] + f" ({results[a]['precision']})",
                     "model_b": results[b]["model"] + f" ({results[b]['precision']})",
                     "diff_macro_f1": round(observed, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
                     "p_value": round(float(p), 4), "significant_5pct": bool(lo > 0 or hi < 0)})
        print(rows[-1])
    out = {"n_boot": N_BOOT, "seed": SEED, "test_size": len(y), "pairs": rows}
    (RESULTS / "significance.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    lines = [f"Paired bootstrap on the test set ({len(y)} messages, {N_BOOT:,} resamples, seed {SEED}).", "",
             "| Model A | Model B | Macro-F1 A − B | 95 % interval | p | Significant (5 %) |",
             "|---|---|---|---|---|---|"]
    lines += [f"| {r['model_a']} | {r['model_b']} | {r['diff_macro_f1']:+.4f} | "
              f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] | {r['p_value']:.4f} | "
              f"{'yes' if r['significant_5pct'] else 'no'} |" for r in rows]
    (RESULTS / "significance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
