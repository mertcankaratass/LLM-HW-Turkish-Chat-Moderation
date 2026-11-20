"""Architecture diagram of the application (requirement R6) -> docs/architecture.svg

Every box names the function / file that implements it. Model sizes and decision thresholds are read from
results/results.json, the lexicon size from models/lexicon.json and the network sizes from models/ffn.pt, so
the diagram stays in sync with the code and the results.
"""
import json
from html import escape

import torch

from chatmod.common import MODELS, RESULTS, ROOT

W, H = 1600, 970
INK, MUTED, LINE = "#1f2933", "#52606d", "#7b8794"
ACCENT, ACCENT_BG = "#0b8a78", "#e3f6f2"
CPU, GPU = "#2a6fdb", "#c2410c"
FONT = "Segoe UI, Helvetica, Arial, sans-serif"


def box(x, y, w, h, title, lines=(), accent=False, fill="#ffffff", title_size=16):
    stroke = ACCENT if accent else "#9aa5b1"
    bg = ACCENT_BG if accent else fill
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{bg}" stroke="{stroke}" '
           f'stroke-width="{2.5 if accent else 1.3}"/>',
           f'<text x="{x + 12}" y="{y + 24}" font-size="{title_size}" font-weight="700" fill="{INK}">'
           f'{escape(title)}</text>']
    for i, line in enumerate(lines):
        out.append(f'<text x="{x + 12}" y="{y + 46 + i * 19}" font-size="13.5" fill="{MUTED}">{line}</text>')
    return "\n".join(out)


def container(x, y, w, h, label, where=None, where_color=CPU):
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="#f5f7fa" stroke="{LINE}" '
           f'stroke-width="1.3" stroke-dasharray="7 5"/>',
           f'<text x="{x + 14}" y="{y + 22}" font-size="13" font-weight="700" letter-spacing="1.2" '
           f'fill="{MUTED}">{escape(label)}</text>']
    if where:
        out.append(pill(x + w - 14, y + 8, where, where_color, anchor="end"))
    return "\n".join(out)


def pill(x, y, text, color, anchor="start"):
    width = 7.2 * len(text) + 18
    x0 = x - width if anchor == "end" else x
    return (f'<rect x="{x0:.0f}" y="{y}" width="{width:.0f}" height="21" rx="10.5" fill="{color}"/>'
            f'<text x="{x0 + width / 2:.0f}" y="{y + 15}" font-size="12" font-weight="700" fill="#ffffff" '
            f'text-anchor="middle">{escape(text)}</text>')


def arrow(points, dashed=False, color=LINE, label=None, label_at=None):
    d = "M " + " L ".join(f"{x},{y}" for x, y in points)
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    out = [f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.8"{dash} marker-end="url(#arrow)"/>']
    if label:
        lx, ly = label_at
        out.append(f'<text x="{lx}" y="{ly}" font-size="12.5" fill="{MUTED}">{escape(label)}</text>')
    return "\n".join(out)


def main() -> None:
    res = {m["id"]: m for m in json.loads((RESULTS / "results.json").read_text(encoding="utf-8"))["models"]}
    hw = json.loads((RESULTS / "results.json").read_text(encoding="utf-8"))["hardware"]
    lex = json.loads((MODELS / "lexicon.json").read_text(encoding="utf-8"))
    ffn = torch.load(MODELS / "ffn.pt", map_location="cpu", weights_only=True)

    def mb(key):
        return f"{res[key]['size_mb']:.3g} MB"

    s = []
    s.append(f'<text x="40" y="44" font-size="26" font-weight="700" fill="{INK}">Architecture · real-time '
             f'Turkish chat moderation</text>')
    s.append(f'<text x="40" y="70" font-size="14.5" fill="{MUTED}">Inference path (top): one message, '
             f'one model call, CPU only, {hw["torch_threads"]} PyTorch thread · offline path (bottom): how the '
             f'models were built. Code: app.py, src/chatmod/.</text>')

    # ---- browser (input) ----
    s.append(container(30, 110, 250, 560, "BROWSER · INPUT", "Gradio UI", "#52606d"))
    s.append(box(45, 150, 220, 112, "Chat message", ["typed live in the lobby box,", "e.g. “s4l4k mısın”",
                                                       "→ send() in app.py"]))
    s.append(box(45, 285, 220, 150, "Model switcher", ["radio list of the 5 models,", "default: BERTurk INT8",
                                                         "one model per message", "→ classify() in app.py"]))
    s.append(box(45, 458, 220, 112, "Compare all five", ["button: runs the message", "through every model",
                                                           "→ compare() in app.py"]))

    # ---- server ----
    s.append(container(300, 110, 1020, 560, "PYTHON SERVER · python app.py · runs locally",
                       f"CPU · {hw['torch_threads']} thread / message", CPU))
    # preprocessing
    s.append(f'<text x="318" y="160" font-size="13" font-weight="700" fill="{MUTED}">1 · PREPROCESSING</text>')
    s.append(box(315, 172, 220, 178, "normalize()", ["normalize.py · every model", "Turkish lowercase I→ı, İ→i",
                                                     "leetspeak: s4l4k → salak", "repeated letters: aaa → a",
                                                     "whitespace squeezing", "regex, no learned weights"]))
    s.append(box(345, 378, 190, 96, "asciify() + words", ["ç ğ ı ö ş ü → c g i o s u", "lexicon.tokenize()",
                                                          "classical_preprocess()"]))
    s.append(box(315, 500, 220, 96, "WordPiece tokenizer", ["BERTurk vocabulary", "max 128 tokens",
                                                            "AutoTokenizer"]))
    # models
    s.append(f'<text x="573" y="160" font-size="13" font-weight="700" fill="{MUTED}">2 · MODELS · '
             f'predictors.py (name · precision · size)</text>')
    models = [
        ("lexicon", "Lexicon + edit distance", f"rules · {mb('lexicon')}",
         f"{len(lex['words'])} words learned from train · own Levenshtein"),
        ("tfidf", "tf-idf + logistic regression", f"FP64 · {mb('tfidf')}",
         "word 1–2-grams + char 2–5-grams · scikit-learn"),
        ("ffn", "word2vec + feedforward NN", f"FP32 · {mb('ffn')}",
         f"own word2vec {ffn['in_dim']}-d (mean) → {ffn['in_dim']}→{ffn['hidden']}→2 · PyTorch"),
        ("bert_fp32", "BERTurk fine-tuned", f"FP32 · {mb('bert_fp32')}",
         "dbmdz/bert-base-turkish-uncased + classifier"),
        ("bert_int8", "BERTurk quantized", f"INT8 · {mb('bert_int8')} · app default",
         "torch quantize_dynamic: nn.Linear → INT8"),
    ]
    ys = [172, 258, 344, 430, 516]
    centers = {}
    for (key, name, prec, detail), y in zip(models, ys):
        accent = key == "bert_int8"
        s.append(f'<rect x="570" y="{y}" width="400" height="72" rx="8" fill="{ACCENT_BG if accent else "#fff"}" '
                 f'stroke="{ACCENT if accent else "#9aa5b1"}" stroke-width="{2.5 if accent else 1.3}"/>')
        s.append(f'<text x="584" y="{y + 25}" font-size="16" font-weight="700" fill="{INK}">{escape(name)}</text>')
        s.append(f'<text x="956" y="{y + 25}" font-size="13" font-weight="700" text-anchor="end" '
                 f'fill="{ACCENT if accent else MUTED}">{escape(prec)}</text>')
        s.append(f'<text x="584" y="{y + 50}" font-size="13.5" fill="{MUTED}">{escape(detail)}</text>')
        centers[key] = y + 36
    # post-processing
    s.append(f'<text x="1008" y="160" font-size="13" font-weight="700" fill="{MUTED}">3 · POST-PROCESSING · '
             f'predictors.py · app.py</text>')
    thr = " · ".join(f"{lbl} {res[k]['threshold']:.2f}" for k, lbl in
                     [("tfidf", "tf-idf"), ("ffn", "FFN"), ("bert_fp32", "FP32"), ("bert_int8", "INT8")])
    s.append(box(1005, 172, 300, 130, "Decision · Predictor.predict()", ["P(offensive) ≥ threshold → offensive",
                                                   "thresholds tuned on validation:", thr,
                                                   "lexicon: any rule hit"]))
    s.append(box(1005, 320, 300, 76, "Masking · app.py", ["flagged text → redaction bars",
                                                 "(original never shown)"]))
    s.append(box(1005, 414, 300, 76, "Normalization tags", ["normalization_steps(): which",
                                                            "regex rules fired (no words shown)"]))
    s.append(box(1005, 508, 300, 96, "Latency · app.py classify()", ["time.perf_counter() around the", "model call · budget 50 ms (mean)",
                                                 "over 50 ms: shown in red"]))

    # ---- browser (output) ----
    s.append(container(1340, 110, 230, 560, "BROWSER · OUTPUT", None))
    s.append(box(1352, 150, 206, 200, "Chat feed", ["one line per message:", "CLEAN / FLAGGED + P(off),",
                                                    "masked text,", "normalization tags,", "ms and model"]))
    s.append(box(1352, 368, 206, 112, "Compare table", ["5 models side by side:", "verdict + P(off), ms;",
                                                        "threshold on hover"]))
    s.append(box(1352, 498, 206, 150, "Response-time log", ["side panel: ms of every", "message, count, flagged,",
                                                            "mean ms, messages", "within 50 ms"]))

    # ---- arrows: inference ----
    s.append(arrow([(265, 206), (315, 206)]))                                   # message -> normalize
    s.append(arrow([(425, 350), (425, 378)]))                                   # normalize -> asciify
    s.append(arrow([(330, 350), (330, 500)]))                                   # normalize -> wordpiece
    for k in ("lexicon", "tfidf", "ffn"):
        s.append(arrow([(535, 426), (552, 426), (552, centers[k]), (570, centers[k])]))
    for k in ("bert_fp32", "bert_int8"):
        s.append(arrow([(535, 548), (560, 548), (560, centers[k]), (570, centers[k])]))
    s.append(arrow([(265, 350), (290, 350), (290, 94), (930, 94), (930, 172)], dashed=True, color=ACCENT,
                   label="model switcher selects the model · Compare runs all five", label_at=(480, 88)))
    s.append(arrow([(265, 514), (282, 514), (282, 350)], dashed=True, color=ACCENT))
    for k in centers:                                                          # models -> bus -> decision
        s.append(f'<path d="M 970,{centers[k]} L 988,{centers[k]}" stroke="{LINE}" stroke-width="1.8"/>')
    s.append(f'<path d="M 988,{centers["lexicon"]} L 988,{centers["bert_int8"]}" stroke="{LINE}" '
             f'stroke-width="1.8"/>')
    s.append(arrow([(988, 237), (1005, 237)]))
    s.append(f'<text x="988" y="{centers["bert_int8"] + 20}" font-size="12" fill="{MUTED}" '
             f'text-anchor="middle">P(off)</text>')
    s.append(arrow([(1155, 302), (1155, 320)]))                                 # decision -> masking
    s.append(arrow([(1305, 215), (1352, 215)]))                                 # decision -> feed
    s.append(arrow([(1305, 358), (1352, 300)]))                                 # masking -> feed
    s.append(arrow([(1305, 452), (1352, 335)]))                                 # tags -> feed
    s.append(arrow([(1305, 556), (1352, 573)]))                                 # latency -> log
    s.append(arrow([(1305, 270), (1326, 270), (1326, 424), (1352, 424)], dashed=True))  # decision -> compare

    # ---- offline ----
    s.append(container(30, 700, 1540, 250, "OFFLINE · BUILT ONCE (scripts/)", None))
    s.append(box(45, 742, 330, 176, "Data · prepare_data.py",
                 ["OffensEval-TR 2020 (Çöltekin, CC-BY)", "official train → train 28,580 /",
                  "val 3,176 (90/10, stratified, seed 42)", "official test 3,528: final comparison",
                  "only, never used for tuning"]))
    s.append(box(405, 742, 380, 176, "Training · train_*.py",
                 ["lexicon, tf-idf + LR, word2vec + FFN (SGD)", "BERTurk fine-tuning (mixed precision)",
                  "every threshold / hyper-parameter", "chosen on the validation split"]))
    s.append(pill(417, 880, "CPU", CPU))
    s.append(pill(470, 880, "GPU · RTX 4060 (BERTurk)", GPU))
    s.append(box(815, 742, 360, 176, "Quantize + evaluate",
                 ["quantize.py: dynamic INT8 + thresholds", "tuned on val (CPU, 1 thread, 1 message)",
                  "evaluate_all.py: same test set,", f"latency on {escape(hw['cpu'].replace('13th Gen ', ''))}"]))
    s.append(pill(827, 880, "CPU · 1 thread", CPU))
    s.append(box(1205, 742, 350, 176, "Model hosting",
                 ["Hugging Face Hub (pinned commit)", "mirror: Google Drive (SHA-256)",
                  "download.py · ensure_models():", "first start → models/ (local disk)",
                  "weights are not in the GitHub repo"]))
    s.append(arrow([(375, 840), (405, 840)]))
    s.append(arrow([(785, 840), (815, 840)]))
    s.append(arrow([(1175, 840), (1205, 840)]))
    s.append(arrow([(1380, 742), (1380, 718), (1260, 718), (1260, 670)], color=ACCENT,
                   label="first start: download models", label_at=(1092, 690)))

    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
           f'font-family="{FONT}">\n<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
           f'markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" '
           f'fill="context-stroke"/></marker></defs>\n<rect width="{W}" height="{H}" fill="#ffffff"/>\n'
           + "\n".join(s) + "\n</svg>\n")
    out = ROOT / "docs" / "architecture.svg"
    out.parent.mkdir(exist_ok=True)
    out.write_text(svg, encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
