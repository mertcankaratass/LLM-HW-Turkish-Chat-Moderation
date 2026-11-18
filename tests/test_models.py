"""Integrity tests of the data splits, the trained models and the published results.

They need data/ and models/ (created by scripts/prepare_data.py and the first app run); without them they are
skipped, so the unit tests still run in a fresh clone.
"""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from chatmod.common import DATA, MODELS, RESULTS, load_split
from chatmod.download import REQUIRED_FILES

needs_data = pytest.mark.skipif(not (DATA / "test.csv").is_file(), reason="run scripts/prepare_data.py first")
needs_models = pytest.mark.skipif(any(not (MODELS / f).is_file() for f in REQUIRED_FILES),
                                  reason="models not downloaded yet")
KEYS = ["lexicon", "tfidf", "ffn", "bert_fp32", "bert_int8"]


@needs_data
def test_splits_have_the_documented_sizes_and_do_not_overlap():
    train, val, test = load_split("train"), load_split("val"), load_split("test")
    assert (len(train) + len(val), len(test)) == (31756, 3528)
    assert abs(len(val) / (len(train) + len(val)) - 0.1) < 0.001
    ids = [set(df["id"]) for df in (train, val, test)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert set(train["label"]) == set(val["label"]) == set(test["label"]) == {0, 1}
    # stratified split: same share of offensive messages in train and val
    assert abs(train["label"].mean() - val["label"].mean()) < 0.002


@needs_data
def test_reported_macro_f1_is_recomputable_from_the_saved_test_predictions():
    results = json.loads((RESULTS / "results.json").read_text(encoding="utf-8"))
    preds = pd.read_csv(RESULTS / "test_predictions.csv")
    test = load_split("test")
    assert list(preds["id"]) == list(test["id"]) and list(preds["label"]) == list(test["label"])
    for m in results["models"]:
        y_hat = (preds[f"p_{m['id']}"] >= m["threshold"]).astype(int)
        assert round(f1_score(preds["label"], y_hat, average="macro"), 4) == m["macro_f1"], m["id"]


@pytest.fixture(scope="module")
def predictors():
    from chatmod.predictors import load_all

    return load_all()


@needs_models
def test_every_model_returns_a_label_and_a_confidence(predictors):
    assert list(predictors) == KEYS
    for key, model in predictors.items():
        for text in ["gg wp, çok iyi oynadınız", "", "s4l4k mısın", "😀😀😀"]:
            label, conf = model.predict(text)
            assert label in (0, 1) and 0.0 <= conf <= 1.0, (key, text)
            # label follows the validation-tuned threshold; conf is the probability of that label
            p_off = float(model.predict_proba([text])[0])
            assert label == int(p_off >= model.threshold), (key, text)
            assert abs(conf - (p_off if label else 1 - p_off)) < 1e-9, (key, text)


@needs_models
def test_thresholds_match_the_published_results(predictors):
    results = {m["id"]: m for m in json.loads((RESULTS / "results.json").read_text(encoding="utf-8"))["models"]}
    for key, model in predictors.items():
        assert model.threshold == results[key]["threshold"], key


@needs_models
@needs_data
def test_saved_test_predictions_are_reproduced_by_the_loaded_models(predictors):
    """The models served by the app are exactly the models that produced the results table."""
    preds = pd.read_csv(RESULTS / "test_predictions.csv")
    texts = list(load_split("test")["text"][:50])
    for key, model in predictors.items():
        p = model.predict_proba(texts)
        np.testing.assert_allclose(p, preds[f"p_{key}"][:50], atol=1e-4, err_msg=key)


@needs_models
def test_quantized_model_flags_a_direct_insult_and_passes_gaming_praise(predictors):
    m = predictors["bert_int8"]
    assert m.predict("sen tam bir aptalsın")[0] == 1
    assert m.predict("gg wp, çok iyi oynadınız")[0] == 0
