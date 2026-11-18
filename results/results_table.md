Test set: OffensEval-TR 2020 official test (3528 tweets). Latency: CPU, batch size 1, mean of 300 runs after 20 warm-up runs. Hardware: 13th Gen Intel(R) Core(TM) i7-13700H, 1 PyTorch thread(s). Budget: 50 ms per message on one CPU core.

| Model | Precision | Macro-F1 | Size (MB) | Latency mean (ms) | Latency p95 (ms) | ≤ budget |
|---|---|---|---|---|---|---|
| Lexicon + edit distance (rule) | rules | 0.6703 | 0.0009 | 0.03 | 0.06 | yes |
| tf-idf + logistic regression | FP64 | 0.7726 | 3.88 | 0.81 | 1.09 | yes |
| word2vec + feedforward NN | FP32 | 0.6806 | 44.3 | 0.10 | 0.16 | yes |
| BERTurk (fine-tuned) | FP32 | 0.8224 | 422 | 63.97 | 117.54 | no |
| BERTurk (dynamic quantization) | INT8 | 0.8157 | 177 | 32.78 | 62.97 | yes |
