"""Error analysis on the test set, from results/test_predictions.csv (written by evaluate_all.py).

- confusion matrix of every model
- effect of quantization: how many test predictions change between BERTurk FP32 and INT8
- noisy-text subset (leetspeak / repeated letters / missing Turkish characters): macro-F1 per model
- where the models disagree (e.g. tf-idf right but BERTurk wrong)
- example errors of the deployed model (INT8): ids only in results/error_examples.md (published);
  texts in results/local/error_examples_text.md (git-ignored, local use only)
Output: results/error_analysis.json, results/error_examples.md
"""
import json
import re

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from chatmod.common import MODELS, RESULTS, load_split
from chatmod.lexicon import LexiconMatcher, tokenize
from chatmod.normalize import normalize, turkish_lower

N_EXAMPLES = 8


def main() -> None:
    results = json.loads((RESULTS / "results.json").read_text(encoding="utf-8"))
    pred = pd.read_csv(RESULTS / "test_predictions.csv")
    test = load_split("test")
    assert (pred["id"].to_numpy() == test["id"].to_numpy()).all()
    y = pred["label"].to_numpy()
    yhat = {m["id"]: (pred[f"p_{m['id']}"].to_numpy() >= m["threshold"]).astype(int) for m in results["models"]}

    out = {"confusion_matrices": {}, "noisy_subset": {}, "disagreements": {}}
    for key, p in yhat.items():
        tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
        out["confusion_matrices"][key] = {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}

    flips = yhat["bert_fp32"] != yhat["bert_int8"]
    out["quantization_effect"] = {
        "predictions_changed": int(flips.sum()), "of": len(y), "share": round(float(flips.mean()), 4),
        "changed_and_int8_correct": int((flips & (yhat["bert_int8"] == y)).sum()),
        "changed_and_int8_wrong": int((flips & (yhat["bert_int8"] != y)).sum()),
        "mean_abs_prob_diff": round(float(np.abs(pred["p_bert_fp32"] - pred["p_bert_int8"]).mean()), 4)}

    # Noisy text: the raw tweet changes under leetspeak / repeat normalization, or has no Turkish letters
    # although its words contain ASCII stand-ins (approximation: no ç ğ ı ö ş ü at all in a 5+ word tweet).
    lower = test["text"].map(lambda t: " ".join(turkish_lower(t).split()))
    changed = test["text"].map(normalize) != lower
    no_tr = ~test["text"].str.contains("[çğıöşüÇĞİÖŞÜ]", regex=True) & (test["text"].str.split().str.len() >= 5)
    for name, mask in (("leet_or_repeats", changed), ("no_turkish_letters", no_tr)):
        out["noisy_subset"][name] = {"tweets": int(mask.sum()), "offensive": int(y[mask].sum())}
        for key, p in yhat.items():
            out["noisy_subset"][name][key] = round(f1_score(y[mask], p[mask], average="macro"), 4)

    right = {k: v == y for k, v in yhat.items()}
    out["disagreements"] = {
        "only_bert_int8_right_vs_tfidf": int((right["bert_int8"] & ~right["tfidf"]).sum()),
        "only_tfidf_right_vs_bert_int8": int((right["tfidf"] & ~right["bert_int8"]).sum()),
        "all_five_wrong": int(np.all([~r for r in right.values()], axis=0).sum()),
        "all_five_right": int(np.all(list(right.values()), axis=0).sum()),
    }
    (RESULTS / "error_analysis.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))

    # Example errors of the deployed model. Words of the offensive lexicon (with edit-distance tolerance)
    # are masked so that the file can be published with the repository.
    lexicon = set(json.loads((MODELS / "lexicon.json").read_text(encoding="utf-8"))["words"])
    matcher = LexiconMatcher(lexicon, prefix_match=True)

    def masked(text: str) -> str:
        bad = {tok for tok, _, _ in matcher.matches(text)}
        words = []
        for w in text.split():
            toks = tokenize(w)
            words.append(w[0] + "*" * (len(w) - 1) if toks and any(t in bad for t in toks) else w)
        return re.sub(r"\s+", " ", " ".join(words))

    p8 = pred["p_bert_int8"].to_numpy()
    groups = {
        "False negatives of BERTurk INT8 (offensive, predicted clean)": np.where((y == 1) & (yhat["bert_int8"] == 0))[0],
        "False positives of BERTurk INT8 (clean, predicted offensive)": np.where((y == 0) & (yhat["bert_int8"] == 1))[0],
        "tf-idf right, BERTurk INT8 wrong": np.where(right["tfidf"] & ~right["bert_int8"])[0],
        "BERTurk INT8 right, tf-idf and word2vec+FFN wrong": np.where(right["bert_int8"] & ~right["tfidf"] & ~right["ffn"])[0],
    }
    rng = np.random.default_rng(0)
    head = ["# Error examples (OffensEval-TR 2020 test set)", "",
            "p = P(offensive); label 1 = offensive. Look tweets up by id in the official test file.", ""]
    public, local = list(head), list(head)
    for title, idx in groups.items():
        sample = rng.choice(idx, size=min(N_EXAMPLES, len(idx)), replace=False)
        public += [f"## {title} — {len(idx)} tweets", "", "| id | label | p INT8 | p tf-idf |", "|---|---|---|---|"]
        local += [f"## {title} — {len(idx)} tweets", "", "| id | label | p INT8 | p tf-idf | tweet |",
                  "|---|---|---|---|---|"]
        for i in sample:
            row = f"| {test['id'].iloc[i]} | {y[i]} | {p8[i]:.2f} | {pred['p_tfidf'].iloc[i]:.2f} |"
            public.append(row)
            local.append(row + f" {masked(test['text'].iloc[i]).replace('|', '/')} |")
        public.append("")
        local.append("")
    # Published with the repository: ids only (texts are in the public dataset).
    (RESULTS / "error_examples.md").write_text("\n".join(public), encoding="utf-8")
    # Local, git-ignored: texts with lexicon words partially masked, used to pick slide examples.
    (RESULTS / "local").mkdir(exist_ok=True)
    (RESULTS / "local" / "error_examples_text.md").write_text("\n".join(local), encoding="utf-8")


if __name__ == "__main__":
    main()
