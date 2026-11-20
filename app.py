"""Real-time chat moderation demo (Gradio).

- Chat-like interface (an in-game lobby chat): every message is labeled live (clean / offensive) with a
  confidence score; offensive messages are masked before they are shown. Tags show which regex
  normalization rules fired (leetspeak, repeated letters, Turkish casing).
- Side panel: model switcher (default: INT8-quantized BERTurk) and a log of the response time of every
  message with a summary: count, flagged, mean and how many stayed within the budget (50 ms per message
  on one CPU core).
- "Compare all five models" runs the message in the box (or, if the box is empty, the last chat message)
  through the whole model ladder side by side.
Run: python app.py
"""
import html
import time
from datetime import datetime

import gradio as gr

from chatmod.common import CPU_THREADS
from chatmod.download import ensure_models
from chatmod.normalize import normalization_steps
from chatmod.predictors import load_all

BUDGET_MS = 50.0
ensure_models()  # first run: downloads the trained models (Hugging Face Hub, Google Drive mirror as fallback)
PREDICTORS = load_all()  # also fixes PyTorch to CPU_THREADS thread(s), as in the evaluation
for _p in PREDICTORS.values():  # warm-up: the first call of a PyTorch model is slow (one-time setup)
    for _ in range(3):
        _p.predict("merhaba")
DEFAULT_MODEL = "bert_int8"
CHOICE_LABELS = {"lexicon": "Lexicon + edit distance · rules", "tfidf": "tf-idf + logistic regression · FP64",
                 "ffn": "word2vec + feedforward NN · FP32", "bert_fp32": "BERTurk · FP32",
                 "bert_int8": "BERTurk · INT8 quantized (default)"}
CHOICES = [(CHOICE_LABELS[k], k) for k in PREDICTORS]
# One of each kind: clean, gaming slang, leetspeak insult, negation.
EXAMPLES = ["gg wp, çok iyi oynadınız 👏", "kafasına sık, tek mermi kaldı", "s4l4k mısın",
            "aptal değilim ama bu bölüm zor"]


# ---------- small HTML helpers (all user text goes through html.escape) ----------

def ms_span(ms: float) -> str:
    """Response time; over the 50 ms budget it is shown in the 'offensive' tone."""
    over = " over" if ms > BUDGET_MS else ""
    return f'<span class="ms{over}" title="{ms:.1f} ms of the {BUDGET_MS:.0f} ms budget">{ms:.1f} ms</span>'


def redacted(text: str) -> str:
    return "".join(f'<span class="bar" style="width:{min(len(w), 14) * 0.62:.2f}em"></span>' for w in text.split())


def verdict_chip(label: int, p_off: float, threshold: float, probabilistic: bool) -> str:
    """Label + the model's probability that the message is offensive (its confidence score).

    The label uses the threshold tuned on the validation split (not 0.5), so P(off) is shown together with it.
    """
    if probabilistic:
        score = f"{p_off:.0%}"
        tip = f"P(offensive) = {p_off:.2f}; flagged when ≥ {threshold:.2f} (threshold tuned on validation data)"
    else:
        score, tip = ("rule hit" if label else "no rule hit"), "rule-based: no probability"
    return (f'<span class="chip {"bad" if label else "good"}" title="{tip}">{"FLAGGED" if label else "CLEAN"}'
            f'<b>{score}</b></span>')


def render_feed(history: list[dict]) -> str:
    if not history:
        return ('<div class="feed empty" lang="en"><div class="empty-note">The lobby is quiet.<br>'
                'Type a message below. Every line is checked before other players see it.</div></div>')
    lines = []
    for h in reversed(history):  # column-reverse keeps the newest line at the bottom, scrolled into view
        body = (f'<span class="hidden-bars">{redacted(h["text"])}</span><em>message hidden</em>' if h["label"]
                else html.escape(h["text"]))
        tags = "".join(f'<span class="tag">{s}</span>' for s in h["steps"])
        lines.append(
            f'<div class="line {"flagged" if h["label"] else ""}">'
            f'<span class="ts">{h["time"]}</span>'
            f'<div class="body"><span class="msg">{body}</span>'
            f'<div class="sub"><span class="model">{html.escape(h["model"])}</span>'
            f'{f"<span class=tags>normalized: {tags}</span>" if tags else ""}</div></div>'
            f'<span class="meta">{verdict_chip(h["label"], h["p_off"], h["thr"], h["prob"])}{ms_span(h["ms"])}</span>'
            f'</div>')
    return f'<div class="feed" lang="en">{"".join(lines)}</div>'


def render_hud(history: list[dict]) -> str:
    """Side panel: summary of the response times and one log row per message."""
    n = len(history)
    if not n:
        return '<div class="hud" lang="en"><p class="hint">The response time of every message appears here.</p></div>'
    flagged = sum(h["label"] for h in history)
    mean = sum(h["ms"] for h in history) / n
    within = sum(h["ms"] <= BUDGET_MS for h in history)
    stats = [(f"{n}", "messages"), (f"{flagged}", "flagged"), (f"{mean:.1f}", "mean ms"),
             (f"{within}/{n}", f"≤ {BUDGET_MS:.0f} ms")]
    stat_html = "".join(f'<div class="stat"><b>{v}</b><span>{k}</span></div>' for v, k in stats)
    rows = "".join(
        f'<tr><td>{h["time"]}</td><td>{html.escape(h["model"])}</td>'
        f'<td class="{"bad" if h["label"] else "good"}">{"flagged" if h["label"] else "clean"}</td>'
        f'<td class="num">{ms_span(h["ms"])}</td></tr>'
        for h in reversed(history[-60:]))
    table = (f'<table class="log"><thead><tr><th>time</th><th>model</th><th>label</th><th class="num">latency</th>'
             f'</tr></thead><tbody>{rows}</tbody></table>')
    return f'<div class="hud" lang="en"><div class="stats">{stat_html}</div>{table}</div>'


def render_compare(rows: list[tuple]) -> str:
    if not rows:
        return ('<p class="hint" lang="en">Runs the message in the box through all five models '
                '(empty box: the last chat message). Hover a verdict to see its threshold.</p>')
    body = "".join(
        f'<tr><td>{html.escape(short)}</td><td class="dim">{prec}</td><td>{chip}</td>'
        f'<td class="num">{ms_span(ms)}</td></tr>' for short, prec, chip, ms in rows)
    return (f'<table class="log compare" lang="en"><thead><tr><th>model</th><th>precision</th><th>verdict</th>'
            f'<th class="num">latency</th></tr></thead><tbody>{body}</tbody></table>')


# ---------- event handlers ----------

def classify(key: str, text: str):
    """Label, P(offensive) and latency (ms) of one message: exactly one model call, as in the evaluation."""
    t0 = time.perf_counter()
    label, confidence = PREDICTORS[key].predict(text)
    ms = (time.perf_counter() - t0) * 1000
    return label, (confidence if label else 1 - confidence), ms


def send(message, history, model_key):
    message = (message or "").strip()
    if not message:
        return "", history, gr.update(), gr.update()
    label, p_off, ms = classify(model_key, message)
    p = PREDICTORS[model_key]
    history = history + [{"time": datetime.now().strftime("%H:%M:%S"), "text": message, "label": label,
                          "p_off": p_off, "thr": p.threshold, "prob": p.probabilistic, "ms": ms, "model": p.short,
                          "key": model_key, "steps": normalization_steps(message)}]
    return "", history, render_feed(history), render_hud(history)


def compare(message, history):
    # The box is cleared after sending, so an empty box means: compare the last message of the chat.
    message = (message or "").strip() or (history[-1]["text"] if history else "")
    rows = []
    if message:
        for key, p in PREDICTORS.items():
            label, p_off, ms = classify(key, message)
            rows.append((p.short, p.precision, verdict_chip(label, p_off, p.threshold, p.probabilistic), ms))
    return render_compare(rows)


def clear():
    return [], render_feed([]), render_hud([])


CSS = """
/* "Graphite": a calm, neutral dark theme with one soft sage accent. */
:root, .dark { --bg:#17191c; --panel:#1e2125; --panel-2:#23272c; --line:#2c3036; --ink:#e6e4df;
  --ink-2:#a3a29b; --ink-3:#74756f; --accent:#8fb8a4; --good:#8fb8a4; --bad:#d19a92; --flag:#221e1e;
  --body-background-fill: var(--bg); --background-fill-primary: var(--bg); --background-fill-secondary: var(--panel);
  --block-background-fill: var(--panel); --block-border-color: var(--line); --border-color-primary: var(--line);
  --input-background-fill: var(--panel); --input-border-color: var(--line); --input-border-color-focus: var(--accent);
  --body-text-color: var(--ink); --body-text-color-subdued: var(--ink-2); --block-label-text-color: var(--ink-2);
  --button-secondary-background-fill: var(--panel-2); --button-secondary-background-fill-hover: #2a2f35;
  --button-secondary-text-color: var(--ink); --button-secondary-border-color: var(--line);
  --button-primary-background-fill: var(--accent); --button-primary-background-fill-hover: #a0c5b3;
  --button-primary-text-color: #17191c; --button-primary-border-color: var(--accent);
  --checkbox-background-color-selected: var(--accent); --checkbox-border-color-selected: var(--accent);
  --checkbox-border-color-focus: var(--accent); --checkbox-label-background-fill-selected: transparent;
  --color-accent: var(--accent); --color-accent-soft: rgba(143,184,164,.12); --link-text-color: var(--accent); }
html, body, gradio-app, .gradio-container, .main, .wrap.sidebar-parent { background: var(--bg) !important; }
.gradio-container { max-width: none !important; color: var(--ink); }
footer { display: none !important; }
#stage { max-width: 980px; width: 100%; margin: 0 auto; }
/* Gradio offsets the content by only half of the sidebar width; with a wide sidebar the chat would slide
   under it. On desktop widths the open sidebar pushes the content by its full width instead. */
@media (min-width: 900px) {
  .gradio-container .main { max-width: none !important; margin-left: 0 !important; margin-right: 0 !important; }
  .wrap.sidebar-parent { padding-right: 0 !important; transition: padding-left .2s ease; }
  .wrap.sidebar-parent:has(.sidebar.open) { padding-left: 356px !important; }
}
.sidebar { background: var(--panel) !important; border-right: 1px solid var(--line); }
.ts, .ms, .num, .stat b, .chip b { font-family: ui-monospace, "Cascadia Mono", Consolas, monospace;
  font-variant-numeric: tabular-nums; }
#title h1 { font-size: 1.3rem; font-weight: 600; margin: 0; color: var(--ink); }
#title p { color: var(--ink-3); margin: .3rem 0 0; font-size: .86rem; }
.section-label { color: var(--ink-3); font-size: .7rem; text-transform: uppercase; letter-spacing: .08em;
  margin: 8px 0 2px; }
/* chat feed */
.feed { height: clamp(240px, calc(100vh - 420px), 520px); overflow-y: auto; display: flex;
  flex-direction: column-reverse; gap: 2px; background: var(--bg); border: 1px solid var(--line);
  border-radius: 10px; padding: 8px; }
.feed.empty { justify-content: center; align-items: center; }
.empty-note { color: var(--ink-3); text-align: center; line-height: 1.7; font-size: .9rem; }
.line { display: flex; align-items: flex-start; flex-wrap: wrap; gap: 4px 14px; padding: 8px 10px;
  border-radius: 6px; animation: in .18s ease-out; }
.line.flagged { background: var(--flag); }
.ts { color: var(--ink-3); font-size: .75rem; flex: none; padding-top: 3px; }
.body { flex: 1 1 260px; min-width: 0; }
.msg { color: var(--ink); overflow-wrap: anywhere; }
.sub { margin-top: 2px; font-size: .72rem; display: flex; flex-wrap: wrap; gap: 4px 10px; }
.sub, .sub .model, .sub .tags { color: var(--ink-3) !important; font-weight: 400 !important; }
.hidden-bars { display: inline-flex; flex-wrap: wrap; gap: 4px; vertical-align: middle; }
.line em { color: var(--ink-3); font-size: .78rem; font-style: normal; margin-left: 8px; }
.bar { display: inline-block; height: .85em; border-radius: 2px; background: #3a3e44; }
.tag { margin-left: 4px; padding: 0 5px; border-radius: 3px; border: 1px solid var(--line); color: var(--ink-2); }
.meta { display: flex; align-items: center; gap: 12px; margin-left: auto; padding-top: 1px; }
.chip { display: inline-flex; gap: 6px; align-items: center; font-size: .68rem; font-weight: 600;
  letter-spacing: .05em; padding: 1px 8px; border-radius: 999px; border: 1px solid; cursor: help; }
.chip b { font-weight: 500; letter-spacing: 0; }
.chip.good { color: var(--good); border-color: rgba(143,184,164,.4); }
.chip.bad { color: var(--bad); border-color: rgba(209,154,146,.45); }
.ms { color: var(--ink-2); font-size: .76rem; }
.meta .ms { min-width: 4.6em; text-align: right; }
.ms.over { color: var(--bad); }
/* side panel */
.hud .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 4px; margin: 4px 0 10px; }
.stat { text-align: left; padding: 2px 2px 0; }
.stat b { display: block; font-size: 1.05rem; font-weight: 500; color: var(--ink); }
.stat span { font-size: .66rem; color: var(--ink-3); }
.log { width: 100%; border-collapse: collapse; font-size: .76rem; border: none !important; }
.log tr, .log thead, .log tbody { border: none !important; background: transparent !important; }
.log th { text-align: left; color: var(--ink-3); font-weight: 500; font-size: .68rem; padding: 4px 6px;
  font-family: inherit !important;
  border: none !important; border-bottom: 1px solid var(--line) !important; background: transparent !important; }
.log td { padding: 5px 6px; color: var(--ink-2); white-space: nowrap; background: transparent !important;
  border: none !important; border-bottom: 1px solid #24282d !important; }
.log td.good { color: var(--good); } .log td.bad { color: var(--bad); } .log td.dim { color: var(--ink-3); }
.log .num { text-align: right; }
.hint { color: var(--ink-3); font-size: .84rem; margin: 4px 0; }
/* model switcher: one model per row, the active one marked by the accent */
.sidebar fieldset .wrap { flex-direction: column; gap: 6px; }
.sidebar fieldset label { width: 100%; background: transparent !important; border: 1px solid var(--line) !important;
  color: var(--ink-2) !important; box-shadow: none !important; }
.sidebar fieldset label:hover { border-color: #3d424a !important; color: var(--ink) !important; }
.sidebar fieldset label.selected { border-color: rgba(143,184,164,.6) !important;
  background: rgba(143,184,164,.08) !important; color: var(--ink) !important; }
#composer { align-items: stretch; }
.gallery { gap: 6px !important; }
.gallery-item { font-size: 12.5px !important; }
#send-btn { min-width: 100px; }
@keyframes in { from { opacity: 0; transform: translateY(3px); } to { opacity: 1; transform: none; } }
@media (prefers-reduced-motion: reduce) { .line { animation: none; } }
"""

# The lobby is dark by design: switch Gradio to its dark theme (documented ?__theme=dark URL parameter).
# Runs in <head>, before the app renders, so the page never flashes in the light theme. The page language
# is English (the UI text is English); this also keeps browser upper-casing from turning "i" into "İ".
FORCE_DARK = """<script>
  (function () {
    document.documentElement.lang = 'en';
    var url = new URL(window.location.href);
    if (url.searchParams.get('__theme') !== 'dark') {
      url.searchParams.set('__theme', 'dark');
      window.location.replace(url.href);
    }
  })();
</script>"""

with gr.Blocks(title="Real-Time Chat Moderation") as demo:
    history = gr.State([])
    with gr.Sidebar(open=True, width=340):
        gr.HTML('<div class="section-label" lang="en">Active model</div>')
        model_key = gr.Radio(choices=CHOICES, value=DEFAULT_MODEL, show_label=False, container=False)
        gr.HTML('<div class="section-label" lang="en">Response-time log</div>')
        hud = gr.HTML(render_hud([]))
    with gr.Column(elem_id="stage"):
        gr.HTML(f'<div id="title" lang="en"><h1>Lobby chat · live moderation</h1>'
                f'<p>Turkish messages are checked before other players see them; offensive lines are masked. '
                f'CPU only · {CPU_THREADS} core per message · budget {BUDGET_MS:.0f}&nbsp;ms.</p></div>')
        feed = gr.HTML(render_feed([]))
        with gr.Row(elem_id="composer"):
            box = gr.Textbox(placeholder="Say something in the lobby and press Enter…", show_label=False,
                             lines=1, max_lines=1, scale=5, autofocus=True, container=False)
            send_btn = gr.Button("Send", variant="primary", scale=0, elem_id="send-btn")
        gr.Examples(EXAMPLES, inputs=box, label="Try an example")
        with gr.Accordion("Compare all five models on a message", open=False):
            compare_btn = gr.Button("Run all five models", size="sm")
            compare_out = gr.HTML(render_compare([]))
        clear_btn = gr.Button("Clear chat and log", size="sm", variant="secondary")

    box.submit(send, [box, history, model_key], [box, history, feed, hud])
    send_btn.click(send, [box, history, model_key], [box, history, feed, hud])
    compare_btn.click(compare, [box, history], [compare_out])
    clear_btn.click(clear, None, [history, feed, hud])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Base(primary_hue="green", neutral_hue="stone"), css=CSS, head=FORCE_DARK,
                footer_links=[])
