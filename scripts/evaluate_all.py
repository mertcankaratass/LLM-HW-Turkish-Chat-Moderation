"""Final comparison of the whole model ladder on the SAME held-out test set (R3, R4).

For every model: macro-F1 (task metric), per-class precision/recall, accuracy, size on disk (MB) and
CPU latency per message (batch size 1, mean over N_LATENCY test messages after warm-up, i.e. >= 100 runs).
Everything runs on the CPU, as in the use case ("CPU-only server").
Output: results/results.json, results/results_table.md, results/test_predictions.csv
"""
import json
import platform
import time

from chatmod.common import RESULTS, load_split
from chatmod.download import ensure_models
from chatmod.predictors import load_all

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

N_WARMUP, N_LATENCY = 20, 300
BUDGET_MS = 50.0


def cpu_name() -> str:
    try:
        import subprocess
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Processor).Name"], capture_output=True, text=True)
        return out.stdout.strip() or platform.processor()
    except OSError:
        return platform.processor()


def latency_ms(predictor, texts) -> dict:
    for t in texts[:N_WARMUP]:
        predictor.predict(t)
    times = []
    for t in texts[N_WARMUP:N_WARMUP + N_LATENCY]:
        t0 = time.perf_counter()
        predictor.predict(t)
        times.append((time.perf_counter() - t0) * 1000)
    return {"mean": round(float(np.mean(times)), 2), "p95": round(float(np.percentile(times, 95)), 2),
            "runs": len(times)}


def main() -> None:
    test = load_split("test")
    texts, y = list(test["text"]), test["label"].to_numpy()
    ensure_models()
    predictors = load_all()  # also fixes the number of PyTorch threads (CPU_THREADS)
    hardware = {"cpu": cpu_name(), "torch_threads": torch.get_num_threads(), "torch": torch.__version__,
                "os": platform.platform(), "device": "cpu"}
    print(hardware)

    rows, preds = [], {"id": test["id"], "label": y}
    for key, predictor in predictors.items():
        p = predictor.predict_proba(texts)
        yhat = (p >= predictor.threshold).astype(int)
        prec, rec, _, _ = precision_recall_fscore_support(y, yhat, labels=[0, 1], zero_division=0)
        lat = latency_ms(predictor, texts)
        row = {"id": key, "model": predictor.name, "precision": predictor.precision,
               "macro_f1": round(f1_score(y, yhat, average="macro"), 4),
               "accuracy": round(accuracy_score(y, yhat), 4),
               "precision_offensive": round(float(prec[1]), 4), "recall_offensive": round(float(rec[1]), 4),
               "size_mb": predictor.size_mb, "latency_ms_mean": lat["mean"], "latency_ms_p95": lat["p95"],
               "latency_runs": lat["runs"], "meets_budget": lat["mean"] <= BUDGET_MS,
               "threshold": predictor.threshold}
        rows.append(row)
        preds[f"p_{key}"] = p  # full precision, so every metric can be recomputed exactly
        print(row)

    (RESULTS / "results.json").write_text(json.dumps(
        {"hardware": hardware, "budget_ms": BUDGET_MS, "test_size": len(test), "models": rows}, indent=1),
        encoding="utf-8")
    import pandas as pd
    pd.DataFrame(preds).to_csv(RESULTS / "test_predictions.csv", index=False)

    lines = [f"Test set: OffensEval-TR 2020 official test ({len(test)} tweets). "
             f"Latency: CPU, batch size 1, mean of {N_LATENCY} runs after {N_WARMUP} warm-up runs. "
             f"Hardware: {hardware['cpu']}, {hardware['torch_threads']} PyTorch thread(s). "
             f"Budget: {BUDGET_MS:.0f} ms per message on one CPU core.", "",
             "| Model | Precision | Macro-F1 | Size (MB) | Latency mean (ms) | Latency p95 (ms) | ≤ budget |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['precision']} | {r['macro_f1']:.4f} | {r['size_mb']:.3g} | "
                     f"{r['latency_ms_mean']:.2f} | {r['latency_ms_p95']:.2f} | "
                     f"{'yes' if r['meets_budget'] else 'no'} |")
    (RESULTS / "results_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
