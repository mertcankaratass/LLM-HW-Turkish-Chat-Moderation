"""Real-time chat moderation demo (Gradio).

- Chat-like interface (an in-game lobby chat): every message is labeled live (clean / offensive) with a
  confidence score; offensive messages are masked before they are shown. Tags show which regex
  normalization rules fired (leetspeak, repeated letters, Turkish casing).
- Side panel: model switcher (default: INT8-quantized BERTurk) and a log of the response time of every
  message: statistics, a latency chart with the budget line, and a game-style "ping" meter per message
  (budget: 50 ms per message on one CPU core).
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
# One of each kind: clean, gaming slang, offensive, leetspeak, negation, irony.
EXAMPLES = ["gg wp, çok iyi oynadınız 👏", "kafasına sık, tek mermi kaldı", "sen tam bir aptalsın", "s4l4k mısın",
            "aptal değilim ama bu bölüm zor", "ne kadar da zekisin, bravo"]


# ---------- small HTML helpers (all user text goes through html.escape) ----------

def ping_level(ms: float) -> tuple[int, str]:
    ratio = ms / BUDGET_MS
    if ratio <= 0.25:
        return 4, "good"
    if ratio <= 0.5:
        return 3, "good"
    return (2, "warn") if ratio <= 1 else (1, "bad")


def ping(ms: float) -> str:
    """Game-style ping meter: 4 green bars far below the budget, 1 red bar over it."""
    bars, level = ping_level(ms)
    cells = "".join(f'<i class="{"on" if i < bars else ""}"></i>' for i in range(4))
    return f'<span class="ping {level}" title="{ms:.1f} ms of {BUDGET_MS:.0f} ms budget">{cells}</span>'


def redacted(text: str) -> str:
    return "".join(f'<span class="bar" style="width:{min(len(w), 14) * 0.62:.2f}em"></span>' for w in text.split())


def verdict_chip(label: int, p_off: float, threshold: float, probabilistic: bool) -> str:
    """Label + the model's probability that the message is offensive (its confidence score).

    The label uses the threshold tuned on the validation split (not 0.5), so P(off) is shown together with it.
    """
    if probabilistic:
        score = f"P(off) {p_off:.0%}"
        tip = f"P(offensive) = {p_off:.2f}; flagged when ≥ {threshold:.2f} (threshold tuned on validation data)"
    else:
        score, tip = ("rule hit" if label else "no rule hit"), "rule-based: no probability"
    return (f'<span class="chip {"bad" if label else "good"}" title="{tip}">{"FLAGGED" if label else "CLEAN"}'
            f'<b>{score}</b></span>')


def render_feed(history: list[dict]) -> str:
    if not history:
        return ('<div class="feed empty" lang="en"><div class="empty-note">Lobby is quiet.<br>'
                'Type a message below — every line is checked before other players see it.</div></div>')
    lines = []
    for h in reversed(history):  # column-reverse keeps the newest line at the bottom, scrolled into view
        body = (f'<span class="hidden-bars">{redacted(h["text"])}</span><em>message hidden</em>' if h["label"]
                else html.escape(h["text"]))
        tags = "".join(f'<span class="tag">{s}</span>' for s in h["steps"])
        lines.append(
            f'<div class="line {"flagged" if h["label"] else ""} {h["key"]}">'
            f'<span class="ts">{h["time"]}</span>'
            f'<div class="body"><span class="who">you</span><span class="msg">{body}</span>'
            f'{f"<div class=tags>normalized: {tags}</div>" if tags else ""}</div>'
            f'<span class="meta">{verdict_chip(h["label"], h["p_off"], h["thr"], h["prob"])}{ping(h["ms"])}'
            f'<span class="ms">{h["ms"]:.1f} ms</span><span class="model">{html.escape(h["model"])}</span></span>'
            f'</div>')
    return f'<div class="feed" lang="en">{"".join(lines)}</div>'


def sparkline(history: list[dict], width: int = 360, height: int = 86) -> str:
    """Latency of the last 40 messages with the budget as a dashed line."""
    pts = [h["ms"] for h in history[-40:]]
    top = max(max(pts) * 1.15, BUDGET_MS * 1.3)
    pad = 6

    def y(v):
        return pad + (height - 2 * pad) * (1 - v / top)

    step = (width - 2 * pad) / max(len(pts) - 1, 1)
    xs = [pad + i * step for i in range(len(pts))] if len(pts) > 1 else [width / 2]
    path = " ".join(f"{x:.1f},{y(v):.1f}" for x, v in zip(xs, pts))
    dots = "".join(f'<circle class="{ping_level(v)[1]}" cx="{x:.1f}" cy="{y(v):.1f}" r="3"/>' for x, v in zip(xs, pts))
    by = y(BUDGET_MS)
    return (f'<svg class="spark" viewBox="0 0 {width} {height}" preserveAspectRatio="none" role="img" '
            f'aria-label="Latency of recent messages">'
            f'<line class="budget-line" x1="0" x2="{width}" y1="{by:.1f}" y2="{by:.1f}"/>'
            f'<text x="{width - 4}" y="{by - 4:.1f}" text-anchor="end">{BUDGET_MS:.0f} ms budget</text>'
            f'<polyline points="{path}"/>{dots}</svg>')


def render_hud(history: list[dict]) -> str:
    n = len(history)
    flagged = sum(h["label"] for h in history)
    mean = sum(h["ms"] for h in history) / n if n else 0.0
    within = sum(h["ms"] <= BUDGET_MS for h in history) / n if n else 0.0
    tiles = [("messages", f"{n}"), ("flagged", f"{flagged}"), ("mean ms", f"{mean:.1f}" if n else "–"),
             ("≤ budget", f"{within:.0%}" if n else "–")]
    stat_html = "".join(f'<div class="stat"><b>{v}</b><span>{k}</span></div>' for k, v in tiles)
    legend = (f'<div class="legend">{ping(BUDGET_MS * 0.2)} ≤ {BUDGET_MS / 2:.0f} ms'
              f'{ping(BUDGET_MS * 0.8)} ≤ {BUDGET_MS:.0f} ms{ping(BUDGET_MS * 1.2)} over budget</div>')
    if not n:
        return (f'<div class="hud" lang="en"><div class="stats">{stat_html}</div>{legend}'
                f'<p class="hint">Response times appear here.</p></div>')
    rows = "".join(
        f'<tr><td>{h["time"]}</td><td>{html.escape(h["model"])}</td>'
        f'<td class="{"bad" if h["label"] else "good"}">{"flagged" if h["label"] else "clean"}</td>'
        f'<td class="num">{h["ms"]:.1f}</td><td>{ping(h["ms"])}</td></tr>'
        for h in reversed(history[-60:]))
    table = (f'<table class="log"><thead><tr><th>time</th><th>model</th><th>label</th><th class="num">ms</th>'
             f'<th>ping</th></tr></thead><tbody>{rows}</tbody></table>')
    return f'<div class="hud" lang="en"><div class="stats">{stat_html}</div>{sparkline(history)}{legend}{table}</div>'


def render_compare(rows: list[tuple]) -> str:
    if not rows:
        return ('<p class="hint" lang="en">Runs the message in the box through all five models '
                '(empty box: the last chat message).</p>')
    body = "".join(
        f'<tr><td>{html.escape(short)}</td><td>{prec}</td><td>{chip}</td><td class="num">{thr}</td>'
        f'<td class="num">{ms:.1f}</td><td>{ping(ms)}</td></tr>' for short, prec, chip, thr, ms in rows)
    return (f'<table class="log compare" lang="en"><thead><tr><th>model</th><th>precision</th><th>verdict</th>'
            f'<th class="num">threshold</th><th class="num">ms</th><th>ping</th></tr></thead><tbody>{body}</tbody>'
            f'</table>')


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
            thr = f"{p.threshold:.2f}" if p.probabilistic else "rule"
            rows.append((p.short, p.precision, verdict_chip(label, p_off, p.threshold, p.probabilistic), thr, ms))
    return render_compare(rows)


def clear():
    return [], render_feed([]), render_hud([])


CSS = """
:root, .dark { --bg:#0b0f14; --panel:#111821; --panel-2:#16202b; --line:#233040; --ink:#e7edf3;
  --ink-2:#9aa8b6; --ink-3:#6b7b8f; --accent:#36d1b2; --good:#3ddc84; --warn:#f2b441; --bad:#ff5f5f; }
html, body, gradio-app, .gradio-container, .main, .wrap.sidebar-parent { background: var(--bg) !important; }
.gradio-container { max-width: none !important; color: var(--ink); }
footer { display: none !important; }
#stage { max-width: 1040px; width: 100%; margin: 0 auto; }
/* Gradio offsets the content by only half of the sidebar width; with a wide sidebar the chat would slide
   under it. On desktop widths the open sidebar pushes the content by its full width instead. */
@media (min-width: 900px) {
  .gradio-container .main { max-width: none !important; margin-left: 0 !important; margin-right: 0 !important; }
  .wrap.sidebar-parent { padding-right: 0 !important; transition: padding-left .2s ease; }
  .wrap.sidebar-parent:has(.sidebar.open) { padding-left: 396px !important; }
}
.sidebar { background: var(--panel) !important; border-right: 1px solid var(--line); }
.ts, .ms, .num, .stat b, .chip b { font-family: ui-monospace, "Cascadia Mono", Consolas, monospace;
  font-variant-numeric: tabular-nums; }
#title h1 { font-size: 1.4rem; letter-spacing: .01em; margin: 0; color: var(--ink); }
#title p { color: var(--ink-2); margin: .3rem 0 0; font-size: .9rem; }
#title .tag-accent { color: var(--accent); }
.section-label { color: var(--ink-3); font-size: .7rem; text-transform: uppercase; letter-spacing: .09em;
  margin: 6px 0 2px; }
/* chat feed */
.feed { height: clamp(260px, calc(100vh - 430px), 560px); overflow-y: auto; display: flex;
  flex-direction: column-reverse; gap: 2px; background: #0d131a; border: 1px solid var(--line);
  border-radius: 12px; padding: 10px 12px; }
.feed.empty { justify-content: center; align-items: center; }
.empty-note { color: var(--ink-3); text-align: center; line-height: 1.7; }
.line { display: flex; align-items: flex-start; flex-wrap: wrap; gap: 4px 12px; padding: 7px 10px;
  border-radius: 8px; border-left: 2px solid transparent; animation: in .18s ease-out; }
.line:hover { background: var(--panel); }
.line.flagged { border-left-color: var(--bad); background: rgba(255,95,95,.05); }
.ts { color: var(--ink-3); font-size: .78rem; flex: none; padding-top: 2px; }
.body { flex: 1 1 260px; min-width: 0; }
.who { color: var(--accent); font-weight: 600; margin-right: 8px; }
.who::after { content: ":"; color: var(--ink-3); }
.msg { color: var(--ink); overflow-wrap: anywhere; }
.hidden-bars { display: inline-flex; flex-wrap: wrap; gap: 4px; vertical-align: middle; }
.line em { color: var(--ink-3); font-size: .78rem; font-style: normal; margin-left: 8px; }
.bar { display: inline-block; height: .9em; border-radius: 3px; background: #3a4554; }
.tags { margin-top: 4px; font-size: .7rem; color: var(--ink-3); }
.tag { display: inline-block; margin-left: 4px; padding: 0 6px; border-radius: 4px; color: #b9a7ff;
  background: rgba(155,135,245,.12); border: 1px solid rgba(155,135,245,.3); }
.meta { display: flex; align-items: center; gap: 8px; margin-left: auto; }
.chip { display: inline-flex; gap: 6px; align-items: center; font-size: .7rem; font-weight: 700;
  letter-spacing: .06em; padding: 2px 8px; border-radius: 999px; border: 1px solid; }
.chip b { font-weight: 500; letter-spacing: 0; }
.chip.good { color: var(--good); border-color: rgba(61,220,132,.35); background: rgba(61,220,132,.08); }
.chip.bad { color: var(--bad); border-color: rgba(255,95,95,.4); background: rgba(255,95,95,.1); }
.ms { color: var(--ink-2); font-size: .78rem; min-width: 4.4em; text-align: right; }
.model { font-size: .7rem; min-width: 6.6em; padding: 1px 6px; border-radius: 4px; color: var(--ink-3);
  border: 1px solid var(--line); text-align: center; }
.line.bert_int8 .model { color: var(--accent); border-color: rgba(54,209,178,.4); }
.line.bert_fp32 .model { color: var(--warn); border-color: rgba(242,180,65,.4); }
/* ping meter */
.ping { display: inline-flex; align-items: flex-end; gap: 2px; height: 12px; vertical-align: middle; }
.ping i { display: block; width: 3px; border-radius: 1px; background: #2b3542; }
.ping i:nth-child(1) { height: 4px; } .ping i:nth-child(2) { height: 6.5px; }
.ping i:nth-child(3) { height: 9px; } .ping i:nth-child(4) { height: 12px; }
.ping.good i.on { background: var(--good); } .ping.warn i.on { background: var(--warn); }
.ping.bad i.on { background: var(--bad); }
/* HUD */
.hud .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; }
.stat { background: var(--panel-2); border: 1px solid var(--line); border-radius: 8px; padding: 8px 4px;
  text-align: center; }
.stat b { display: block; font-size: 1.15rem; color: var(--ink); }
.stat span { font-size: .64rem; color: var(--ink-3); text-transform: uppercase; letter-spacing: .06em; }
.spark { width: 100%; height: 86px; margin-top: 10px; background: var(--panel-2); border: 1px solid var(--line);
  border-radius: 8px; }
.spark polyline { fill: none; stroke: #5c6f85; stroke-width: 1.5; vector-effect: non-scaling-stroke; }
.spark circle.good { fill: var(--good); } .spark circle.warn { fill: var(--warn); }
.spark circle.bad { fill: var(--bad); }
.spark .budget-line { stroke: var(--warn); stroke-dasharray: 4 3; stroke-width: 1; vector-effect: non-scaling-stroke; }
.spark text { fill: var(--warn); font-size: 10px; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 10px; align-items: center; margin: 8px 2px; font-size: .72rem;
  color: var(--ink-3); }
.legend .ping { margin-right: 3px; }
.log { width: 100%; border-collapse: collapse; font-size: .78rem; }
.log th { text-align: left; color: var(--ink-3); font-weight: 600; text-transform: uppercase; font-size: .64rem;
  letter-spacing: .06em; padding: 4px 6px; border-bottom: 1px solid var(--line); }
.log td { padding: 5px 6px; border-bottom: 1px solid #1a2430; color: var(--ink-2); white-space: nowrap; }
.log td.good { color: var(--good); } .log td.bad { color: var(--bad); }
.log .num { text-align: right; }
.hint { color: var(--ink-3); font-size: .85rem; }
/* model switcher: one model per row, the active one clearly marked */
.sidebar fieldset .wrap { flex-direction: column; gap: 6px; }
.sidebar fieldset label { width: 100%; background: transparent !important; border: 1px solid var(--line) !important;
  color: var(--ink-2) !important; }
.sidebar fieldset label:hover { border-color: #3a4a5e !important; color: var(--ink) !important; }
.sidebar fieldset label.selected { border-color: var(--accent) !important; background: rgba(54,209,178,.12) !important;
  color: var(--ink) !important; box-shadow: inset 3px 0 0 var(--accent); }
#composer { align-items: stretch; }
/* example messages: compact chips that fit on one row */
.gallery { gap: 6px !important; }
.gallery-item { font-size: 12.5px !important; }
#send-btn { min-width: 110px; }
@keyframes in { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }
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
    with gr.Sidebar(open=True, width=380):
        gr.HTML('<div class="section-label" lang="en">Active model</div>')
        model_key = gr.Radio(choices=CHOICES, value=DEFAULT_MODEL, show_label=False, container=False)
        gr.HTML('<div class="section-label" lang="en">Response-time log</div>')
        hud = gr.HTML(render_hud([]))
    with gr.Column(elem_id="stage"):
        gr.HTML(f'<div id="title" lang="en"><h1>🎮 Lobby chat · <span class="tag-accent">live moderation</span></h1>'
                f'<p>Turkish chat messages are checked before other players see them; offensive lines are '
                f'masked. CPU only · {CPU_THREADS} core per message · budget {BUDGET_MS:.0f}&nbsp;ms.</p></div>')
        feed = gr.HTML(render_feed([]))
        with gr.Row(elem_id="composer"):
            box = gr.Textbox(placeholder="Say something in the lobby and press Enter…", show_label=False,
                             lines=1, max_lines=1, scale=5, autofocus=True, container=False)
            send_btn = gr.Button("Send", variant="primary", scale=0, elem_id="send-btn")
        gr.Examples(EXAMPLES, inputs=box, label="Try an example", examples_per_page=10)
        with gr.Accordion("Compare all five models on a message", open=False):
            compare_btn = gr.Button("Run all five models", size="sm")
            compare_out = gr.HTML(render_compare([]))
        clear_btn = gr.Button("Clear chat and log", size="sm", variant="secondary")

    box.submit(send, [box, history, model_key], [box, history, feed, hud])
    send_btn.click(send, [box, history, model_key], [box, history, feed, hud])
    compare_btn.click(compare, [box, history], [compare_out])
    clear_btn.click(clear, None, [history, feed, hud])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Base(primary_hue="teal", neutral_hue="slate"), css=CSS, head=FORCE_DARK,
                footer_links=[])
