"""Train our own word2vec on the training messages, then a feedforward network with SGD + backprop.

- word2vec: skip-gram, trained only on the training split (no pretrained vectors).
- FFN: average word vector -> Linear -> ReLU -> Dropout -> Linear(2); class-weighted cross-entropy
  (classes are imbalanced); optimizer torch.optim.SGD with momentum; early stopping on validation.
- Hidden size, learning rate and the decision threshold are chosen on the validation split.
Output: models/w2v.model, models/ffn.pt, results/ffn_tuning.json
"""
import json
import random

import numpy as np
import torch
from gensim.models import Word2Vec
from torch import nn

from chatmod.common import MODELS, RESULTS, SEED, load_split, tune_threshold
from chatmod.ffn import FeedForward, embed
from chatmod.lexicon import tokenize

W2V_DIM, EPOCHS, BATCH, PATIENCE = 200, 120, 64, 10


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_ffn(x_tr, y_tr, x_val, y_val, hidden: int, lr: float):
    set_seed(SEED)
    model = FeedForward(x_tr.shape[1], hidden)
    weights = torch.tensor(len(y_tr) / (2 * np.bincount(y_tr)), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    xt, yt = torch.from_numpy(x_tr), torch.from_numpy(y_tr)
    xv = torch.from_numpy(x_val)
    best = {"f1": -1.0}
    stale = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()  # backpropagation
            opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.softmax(model(xv), dim=1)[:, 1].numpy()
        threshold, f1 = tune_threshold(y_val, p)
        if f1 > best["f1"]:
            best = {"f1": f1, "threshold": threshold, "epoch": epoch,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    return best


def main() -> None:
    train, val = load_split("train"), load_split("val")
    sentences = [tokenize(t) for t in train["text"]]
    w2v = Word2Vec(sentences, vector_size=W2V_DIM, window=5, min_count=2, sg=1, epochs=10,
                   seed=SEED, workers=1)
    print(f"word2vec vocabulary: {len(w2v.wv)} words")

    x_tr, x_val = embed(train["text"], w2v), embed(val["text"], w2v)
    y_tr, y_val = train["label"].to_numpy(), val["label"].to_numpy()

    runs, best = [], None
    for hidden in [128, 256]:
        for lr in [0.005, 0.01, 0.05, 0.1]:
            res = train_ffn(x_tr, y_tr, x_val, y_val, hidden, lr)
            run = {"hidden": hidden, "lr": lr, "best_epoch": res["epoch"],
                   "threshold": res["threshold"], "val_macro_f1": round(res["f1"], 4)}
            runs.append(run)
            print(run)
            if best is None or res["f1"] > best[1]["f1"]:
                best = (run, res)

    run, res = best
    w2v.save(str(MODELS / "w2v.model"))
    torch.save({"state_dict": res["state"], "in_dim": W2V_DIM, "hidden": run["hidden"],
                "threshold": run["threshold"], "params": run}, MODELS / "ffn.pt")
    (RESULTS / "ffn_tuning.json").write_text(
        json.dumps({"w2v_vocab": len(w2v.wv), "runs": runs}, indent=1), encoding="utf-8")
    print("best:", run)


if __name__ == "__main__":
    main()
