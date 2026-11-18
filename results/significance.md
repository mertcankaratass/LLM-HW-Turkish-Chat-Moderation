Paired bootstrap on the test set (3528 messages, 10,000 resamples, seed 42).

| Model A | Model B | Macro-F1 A − B | 95 % interval | p | Significant (5 %) |
|---|---|---|---|---|---|
| BERTurk (fine-tuned) (FP32) | BERTurk (dynamic quantization) (INT8) | +0.0067 | [+0.0004, +0.0131] | 0.0352 | yes |
| BERTurk (dynamic quantization) (INT8) | tf-idf + logistic regression (FP64) | +0.0431 | [+0.0263, +0.0595] | 0.0000 | yes |
| BERTurk (fine-tuned) (FP32) | tf-idf + logistic regression (FP64) | +0.0498 | [+0.0329, +0.0666] | 0.0000 | yes |
| tf-idf + logistic regression (FP64) | word2vec + feedforward NN (FP32) | +0.0920 | [+0.0715, +0.1125] | 0.0000 | yes |
| word2vec + feedforward NN (FP32) | Lexicon + edit distance (rule) (rules) | +0.0104 | [-0.0128, +0.0337] | 0.3870 | no |
