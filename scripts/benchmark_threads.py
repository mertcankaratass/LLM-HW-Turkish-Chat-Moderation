"""Latency of BERTurk FP32 vs INT8 with 1, 2 and 4 CPU threads (transparency for the latency budget).

The budget (50 ms per message) is defined for ONE core; this script shows how the picture changes with
more cores. Batch size 1, test messages, 20 warm-up runs, then mean / p95 over N_RUNS runs.
Output: results/latency_threads.json, results/latency_threads.md
"""
import json
import time

from chatmod.common import RESULTS, load_split, request_full_cpu_speed
from chatmod.predictors import BertPredictor

import numpy as np
import torch

THREADS, N_WARMUP, N_RUNS = [1, 2, 4], 20, 200


def main() -> None:
    texts = list(load_split("test")["text"])
    request_full_cpu_speed()  # same CPU conditions as evaluate_all.py and the app
    models = {"FP32": BertPredictor(quantized=False), "INT8": BertPredictor(quantized=True)}
    rows = []
    for n in THREADS:
        torch.set_num_threads(n)
        for precision, model in models.items():
            for t in texts[:N_WARMUP]:
                model.predict(t)
            times = []
            for t in texts[N_WARMUP:N_WARMUP + N_RUNS]:
                t0 = time.perf_counter()
                model.predict(t)
                times.append((time.perf_counter() - t0) * 1000)
            rows.append({"threads": n, "precision": precision, "runs": len(times),
                         "mean_ms": round(float(np.mean(times)), 2),
                         "p95_ms": round(float(np.percentile(times, 95)), 2)})
            print(rows[-1])
    (RESULTS / "latency_threads.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    lines = ["| CPU threads | BERTurk | Mean (ms) | p95 (ms) |", "|---|---|---|---|"]
    lines += [f"| {r['threads']} | {r['precision']} | {r['mean_ms']:.2f} | {r['p95_ms']:.2f} |" for r in rows]
    (RESULTS / "latency_threads.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
