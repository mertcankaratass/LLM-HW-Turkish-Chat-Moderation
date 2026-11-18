"""Figures for the presentation, drawn from the result files (no number is typed by hand).

1. results/figures/f1_vs_latency.png   - macro-F1 vs CPU latency of the whole model ladder, with the budget
2. results/figures/latency_threads.png - BERTurk FP32 vs INT8 latency with 1 / 2 / 4 CPU threads
3. results/figures/quantization.png    - FP32 vs INT8: size, latency, macro-F1 side by side
"""
import json

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chatmod.common import RESULTS

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1 and 2 of the validated reference palette
OUT = RESULTS / "figures"


def style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=13, fontweight="bold", pad=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=10)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name: str) -> None:
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", OUT / name)


def f1_vs_latency(res) -> None:
    budget = res["budget_ms"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    style(ax, "Accuracy vs. speed on one CPU core (test set)")
    for m in res["models"]:
        is_int8 = m["id"] == "bert_int8"
        ax.scatter(m["latency_ms_mean"], m["macro_f1"], s=90 if is_int8 else 60, color=BLUE if is_int8 else INK2,
                   edgecolor=SURFACE, linewidth=2, zorder=3)
        label = f"{m['model']} [{m['precision']}]\nF1 {m['macro_f1']:.3f} · {m['latency_ms_mean']:.2f} ms"
        dx, dy, ha = {"lexicon": (8, -16, "left"), "ffn": (8, 10, "left"), "tfidf": (8, 10, "left"),
                      "bert_fp32": (10, 14, "left"), "bert_int8": (-10, -18, "right")}[m["id"]]
        ax.annotate(label, (m["latency_ms_mean"], m["macro_f1"]), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va="center", fontsize=8.5,
                    color=INK if is_int8 else INK2, fontweight="bold" if is_int8 else "normal")
    ax.axvline(budget, color=ORANGE, linewidth=2, linestyle=(0, (4, 3)), zorder=2)
    ax.text(budget * 1.06, 0.655, f"budget {budget:.0f} ms", color=INK2, fontsize=9, va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(0.01, 1500)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylim(0.64, 0.86)
    ax.set_xlabel("Mean latency per message (ms, log scale)", color=INK2)
    ax.set_ylabel("Macro-F1", color=INK2)
    save(fig, "f1_vs_latency.png")


def latency_threads(rows, budget: float) -> None:
    threads = sorted({r["threads"] for r in rows})
    fig, ax = plt.subplots(figsize=(8, 4.4))
    style(ax, "BERTurk latency per message vs. CPU threads")
    ax.set_xlabel(f"Separate benchmark run: batch size 1, mean of {rows[0]['runs']} test messages per bar",
                  color=MUTED, fontsize=9, labelpad=10)
    width = 0.36
    for k, (prec, color) in enumerate((("FP32", ORANGE), ("INT8", BLUE))):
        xs = [i + (k - 0.5) * (width + 0.03) for i in range(len(threads))]
        ys = [next(r["mean_ms"] for r in rows if r["threads"] == t and r["precision"] == prec) for t in threads]
        ax.bar(xs, ys, width=width, color=color, label=f"BERTurk {prec}", zorder=3)
        for x, yv in zip(xs, ys):
            ax.text(x, yv + 1, f"{yv:.0f}", ha="center", va="bottom", fontsize=9, color=INK2)
    ax.axhline(budget, color=INK2, linewidth=1.5, linestyle=(0, (4, 3)), zorder=4)
    ax.text(len(threads) - 0.5, budget + 1.5, f"budget {budget:.0f} ms", ha="right", color=INK2, fontsize=9)
    ax.set_xticks(range(len(threads)), [f"{t} thread" + ("s" if t > 1 else "") for t in threads])
    ax.set_ylabel("Mean latency (ms)", color=INK2)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, labelcolor=INK2, loc="upper right")
    save(fig, "latency_threads.png")


def quantization(res) -> None:
    m = {r["id"]: r for r in res["models"]}
    panels = [("Size on disk (MB)", "size_mb", "{:.0f}"), ("Mean latency, 1 core (ms)", "latency_ms_mean", "{:.1f}"),
              ("Macro-F1 (test)", "macro_f1", "{:.3f}")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, (title, field, fmt) in zip(axes, panels):
        style(ax, title)
        ax.title.set_fontsize(11)
        vals = [m["bert_fp32"][field], m["bert_int8"][field]]
        ax.bar([0, 1], vals, width=0.6, color=[ORANGE, BLUE], zorder=3)
        for x, v in zip([0, 1], vals):
            ax.text(x, v * 1.01, fmt.format(v), ha="center", va="bottom", fontsize=10, color=INK)
        ax.set_xticks([0, 1], ["FP32", "INT8"])
        ax.set_ylim(0, max(vals) * 1.18)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Dynamic INT8 quantization of BERTurk", x=0.01, ha="left", color=INK, fontsize=13,
                 fontweight="bold")
    save(fig, "quantization.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res = json.loads((RESULTS / "results.json").read_text(encoding="utf-8"))
    rows = json.loads((RESULTS / "latency_threads.json").read_text(encoding="utf-8"))
    f1_vs_latency(res)
    latency_threads(rows, res["budget_ms"])
    quantization(res)


if __name__ == "__main__":
    main()
