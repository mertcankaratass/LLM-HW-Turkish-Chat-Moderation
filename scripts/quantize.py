"""Technique area D: dynamic INT8 quantization of the fine-tuned BERTurk (PyTorch, CPU).

All nn.Linear layers (attention projections, feed-forward layers, classifier) get INT8 weights;
activations are quantized on the fly at inference time.
Both BERTurk decision thresholds (FP32 and INT8) are then tuned on the validation split under the deployment
condition: CPU, CPU_THREADS thread(s), one message at a time (same as evaluate_all.py and app.py). INT8 output
depends on the thread count and on the batch, and the FP32 threshold from GPU training (mixed precision,
batches of 64) is replaced by the one tuned on the CPU. Same tuning procedure as every other model.
Output: models/berturk-int8/ (model_int8.pt, config.json, tokenizer files, threshold.json),
        updated threshold in models/berturk-fp32/threshold.json
"""
import json

from chatmod.common import CPU_THREADS, MODELS, load_split, request_full_cpu_speed, tune_threshold
from chatmod.predictors import BertPredictor

import torch


def tune_on_cpu(predictor: BertPredictor, texts, labels) -> tuple[float, float]:
    return tune_threshold(labels, predictor.predict_proba(texts))


def main() -> None:
    torch.set_num_threads(CPU_THREADS)
    request_full_cpu_speed()
    out = MODELS / "berturk-int8"
    out.mkdir(parents=True, exist_ok=True)
    fp32 = BertPredictor(quantized=False)
    int8_model = torch.ao.quantization.quantize_dynamic(fp32.model, {torch.nn.Linear}, dtype=torch.qint8)
    torch.save(int8_model.state_dict(), out / "model_int8.pt")
    fp32.model.config.save_pretrained(out)  # architecture, so the INT8 model loads without the FP32 weights
    fp32.tokenizer.save_pretrained(out)
    (out / "threshold.json").write_text(json.dumps({"threshold": 0.5}), encoding="utf-8")

    val = load_split("val")
    texts, labels = list(val["text"]), val["label"]
    condition = f"CPU, {CPU_THREADS} thread(s), batch size 1"

    int8 = BertPredictor(quantized=True)  # reload from disk to make sure the saved file works
    threshold, f1 = tune_on_cpu(int8, texts, labels)
    meta = {"threshold": threshold, "val_macro_f1": round(f1, 4), "tuned_on": condition,
            "method": "torch dynamic quantization, nn.Linear -> qint8",
            "size_mb_fp32": fp32.size_mb, "size_mb_int8": int8.size_mb}
    (out / "threshold.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("INT8", meta)

    fp32_meta_path = MODELS / "berturk-fp32" / "threshold.json"
    fp32_meta = json.loads(fp32_meta_path.read_text(encoding="utf-8"))
    fp32_meta.setdefault("threshold_gpu_training", fp32_meta["threshold"])
    fp32_meta.setdefault("val_macro_f1_gpu_training", fp32_meta["val_macro_f1"])
    threshold, f1 = tune_on_cpu(fp32, texts, labels)
    fp32_meta.update({"threshold": threshold, "val_macro_f1": round(f1, 4), "tuned_on": condition})
    fp32_meta_path.write_text(json.dumps(fp32_meta, indent=1), encoding="utf-8")
    print("FP32", fp32_meta)


if __name__ == "__main__":
    main()
