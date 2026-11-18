"""Technique area C: fine-tune BERTurk (dbmdz/bert-base-turkish-uncased) for offensive / clean classification.

Input text: normalize() output (lowercased, leetspeak decoded, repeats collapsed) -> uncased BERTurk.
Training on GPU (mixed precision); the best epoch and the decision threshold are chosen on validation.
Output: models/berturk-fp32/ (weights + tokenizer + threshold.json), results/bert_training.json
"""
import json
import random
import time

# chatmod.common imports pandas; it must be loaded before transformers (import-order crash seen on Windows).
from chatmod.common import MODELS, RESULTS, SEED, load_split, tune_threshold
from chatmod.normalize import normalize

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

MODEL_NAME = "dbmdz/bert-base-turkish-uncased"
# The main branch only has pytorch_model.bin; refs/pr/1 holds the same weights converted to safetensors
# by Hugging Face's official SFconvertbot (data-only format, loads with our PyTorch version).
MODEL_REVISION = "refs/pr/1"
MAX_LEN, BATCH, EPOCHS, LR = 128, 32, 3, 2e-5
OUT = MODELS / "berturk-fp32"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batches(tokenizer, texts, labels, shuffle):
    data = list(zip([normalize(t) for t in texts], labels))

    def collate(items):
        enc = tokenizer([t for t, _ in items], truncation=True, max_length=MAX_LEN, padding=True,
                        return_tensors="pt")
        enc["labels"] = torch.tensor([y for _, y in items])
        return enc

    return DataLoader(data, batch_size=BATCH, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def predict_proba(model, loader, device) -> np.ndarray:
    model.eval()
    probs = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(**batch).logits
        probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    train, val = load_split("train"), load_split("val")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, num_labels=2, use_safetensors=True).to(device)

    lengths = [len(tokenizer(normalize(t))["input_ids"]) for t in train["text"]]
    print(f"token length: mean {np.mean(lengths):.1f}, p99 {np.percentile(lengths, 99):.0f}, "
          f"truncated at {MAX_LEN}: {np.mean(np.array(lengths) > MAX_LEN):.2%}")

    train_loader = batches(tokenizer, train["text"], train["label"], shuffle=True)
    val_loader = batches(tokenizer, val["text"], val["label"], shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = len(train_loader) * EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    log, best_f1 = [], -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0, total = time.time(), 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        p = predict_proba(model, val_loader, device)
        threshold, f1 = tune_threshold(val["label"], p)
        entry = {"epoch": epoch, "train_loss": round(total / len(train_loader), 4),
                 "val_macro_f1": round(f1, 4), "threshold": threshold,
                 "minutes": round((time.time() - t0) / 60, 1)}
        log.append(entry)
        print(entry)
        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(OUT)
            tokenizer.save_pretrained(OUT)
            (OUT / "threshold.json").write_text(json.dumps(entry, indent=1), encoding="utf-8")

    (RESULTS / "bert_training.json").write_text(json.dumps(
        {"model": MODEL_NAME, "revision": MODEL_REVISION, "max_len": MAX_LEN, "batch": BATCH, "lr": LR, "epochs": log,
         "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}, indent=1),
        encoding="utf-8")
    print("best val macro-F1:", round(best_f1, 4))


if __name__ == "__main__":
    main()
