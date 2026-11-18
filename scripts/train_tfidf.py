"""Classical baseline: tf-idf (word 1-2 grams + character 2-5 grams) + logistic regression.

Hyper-parameter C and the decision threshold are chosen on the validation split.
Output: models/tfidf.joblib, results/tfidf_tuning.json
"""
import json

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from chatmod.common import MODELS, RESULTS, SEED, load_split, tune_threshold
from chatmod.normalize import classical_preprocess


def make_pipeline(c: float) -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=classical_preprocess, ngram_range=(1, 2), min_df=2,
                                 sublinear_tf=True)),
        ("char", TfidfVectorizer(preprocessor=classical_preprocess, analyzer="char_wb",
                                 ngram_range=(2, 5), min_df=3, sublinear_tf=True)),
    ])
    clf = LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=SEED)
    return Pipeline([("tfidf", features), ("clf", clf)])


def main() -> None:
    train, val = load_split("train"), load_split("val")
    runs, best = [], None
    for c in [0.25, 0.5, 1, 2, 4, 8, 16]:
        pipe = make_pipeline(c).fit(train["text"], train["label"])
        p = pipe.predict_proba(val["text"])[:, 1]
        threshold, f1 = tune_threshold(val["label"], p)
        run = {"C": c, "threshold": threshold, "val_macro_f1": round(f1, 4),
               "val_macro_f1_at_0.5": round(f1_score(val["label"], p >= 0.5, average="macro"), 4)}
        runs.append(run)
        print(run)
        if best is None or f1 > best[0]["val_macro_f1"]:
            best = (run, pipe)

    run, pipe = best
    joblib.dump({"pipeline": pipe, "threshold": run["threshold"], "params": run}, MODELS / "tfidf.joblib")
    (RESULTS / "tfidf_tuning.json").write_text(json.dumps(runs, indent=1), encoding="utf-8")
    print("best:", run)


if __name__ == "__main__":
    main()
