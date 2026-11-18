"""One common interface for every model, used by both evaluate_all.py and the application.

Each predictor has: name, precision, threshold, size_mb, predict_proba(texts) -> P(offensive) per text.
predict(text) returns (label, confidence) where confidence is the probability of the predicted class.
"""
import json
from pathlib import Path

import joblib
import numpy as np

from chatmod.common import CPU_THREADS, MODELS, request_full_cpu_speed
from chatmod.lexicon import LexiconMatcher
from chatmod.normalize import normalize


def _size_mb(paths) -> float:
    return round(sum(Path(p).stat().st_size for p in paths) / 2**20, 4)


class Predictor:
    name = precision = short = ""
    probabilistic = True  # False: rule-based, outputs only 0 / 1
    threshold = 0.5
    size_mb = 0.0

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def predict(self, text: str) -> tuple[int, float]:
        p = float(self.predict_proba([text])[0])
        label = int(p >= self.threshold)
        return label, (p if label else 1 - p)


class LexiconPredictor(Predictor):
    name, precision, short = "Lexicon + edit distance (rule)", "rules", "Lexicon"
    probabilistic = False

    def __init__(self):
        path = MODELS / "lexicon.json"
        bundle = json.loads(path.read_text(encoding="utf-8"))
        self.matcher = LexiconMatcher(set(bundle["words"]), prefix_match=bundle["params"]["prefix_match"])
        self.size_mb = _size_mb([path])

    def predict_proba(self, texts):
        return np.array([float(self.matcher.predict(t)) for t in texts])


class TfidfPredictor(Predictor):
    name, precision, short = "tf-idf + logistic regression", "FP64", "tf-idf"

    def __init__(self):
        path = MODELS / "tfidf.joblib"
        bundle = joblib.load(path)
        self.pipeline, self.threshold = bundle["pipeline"], bundle["threshold"]
        self.size_mb = _size_mb([path])

    def predict_proba(self, texts):
        return self.pipeline.predict_proba(list(texts))[:, 1]


class FFNPredictor(Predictor):
    name, precision, short = "word2vec + feedforward NN", "FP32", "w2v+FFN"

    def __init__(self):
        import torch
        from gensim.models import Word2Vec

        from chatmod.ffn import FeedForward

        self.torch = torch
        self.w2v = Word2Vec.load(str(MODELS / "w2v.model"))
        ckpt = torch.load(MODELS / "ffn.pt", map_location="cpu", weights_only=True)
        self.model = FeedForward(ckpt["in_dim"], ckpt["hidden"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.threshold = ckpt["threshold"]
        self.size_mb = _size_mb(list(MODELS.glob("w2v.model*")) + [MODELS / "ffn.pt"])

    def predict_proba(self, texts):
        from chatmod.ffn import embed

        with self.torch.no_grad():
            x = self.torch.from_numpy(embed(list(texts), self.w2v))
            return self.torch.softmax(self.model(x), dim=1)[:, 1].numpy()


class BertPredictor(Predictor):
    """BERTurk on CPU. quantized=True loads the dynamic-INT8 weights written by scripts/quantize.py.

    batch_size=1 by default: the chat server classifies one message per request, and dynamic quantization
    computes activation scales per batch (padding and the other messages would change the INT8 output).
    Evaluation, threshold tuning and the app therefore all score messages one at a time.
    """

    MAX_LEN = 128

    def __init__(self, quantized: bool = False, batch_size: int = 1):
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

        self.torch, self.batch_size = torch, batch_size
        if quantized:
            # Self-contained: build the architecture from config.json, quantize it, then load the INT8
            # weights. The FP32 checkpoint is not needed to run the INT8 model.
            model_dir = MODELS / "berturk-int8"
            model = AutoModelForSequenceClassification.from_config(AutoConfig.from_pretrained(model_dir))
            model = torch.ao.quantization.quantize_dynamic(model.eval(), {torch.nn.Linear}, dtype=torch.qint8)
            model.load_state_dict(torch.load(model_dir / "model_int8.pt", map_location="cpu", weights_only=True))
            self.name, self.precision, self.short = "BERTurk (dynamic quantization)", "INT8", "BERTurk INT8"
            self.size_mb = _size_mb([model_dir / "model_int8.pt"])
        else:
            model_dir = MODELS / "berturk-fp32"
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            self.name, self.precision, self.short = "BERTurk (fine-tuned)", "FP32", "BERTurk FP32"
            self.size_mb = _size_mb([model_dir / "model.safetensors"])
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        meta = json.loads((model_dir / "threshold.json").read_text(encoding="utf-8"))
        self.model, self.threshold = model.eval(), meta["threshold"]

    def predict_proba(self, texts):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                chunk = [normalize(t) for t in texts[i:i + self.batch_size]]
                enc = self.tokenizer(chunk, truncation=True, max_length=self.MAX_LEN, padding=True,
                                     return_tensors="pt")
                logits = self.model(**enc).logits
                out.append(self.torch.softmax(logits, dim=1)[:, 1].numpy())
        return np.concatenate(out)


def load_all() -> dict[str, Predictor]:
    """All five models of the ladder, keyed by a short id (used by the app's model switcher)."""
    import torch

    torch.set_num_threads(CPU_THREADS)
    request_full_cpu_speed()  # same CPU conditions for the app and the evaluation
    return {
        "lexicon": LexiconPredictor(),
        "tfidf": TfidfPredictor(),
        "ffn": FFNPredictor(),
        "bert_fp32": BertPredictor(quantized=False),
        "bert_int8": BertPredictor(quantized=True),
    }
