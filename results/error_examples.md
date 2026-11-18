# Error examples (OffensEval-TR 2020 test set)

p = P(offensive); label 1 = offensive. Look tweets up by id in the official test file.

## False negatives of BERTurk INT8 (offensive, predicted clean) — 248 tweets

| id | label | p INT8 | p tf-idf |
|---|---|---|---|
| 20133 | 1 | 0.11 | 0.60 |
| 27370 | 1 | 0.32 | 0.25 |
| 19214 | 1 | 0.05 | 0.16 |
| 22265 | 1 | 0.16 | 0.29 |
| 15114 | 1 | 0.04 | 0.14 |
| 20435 | 1 | 0.11 | 0.19 |
| 34083 | 1 | 0.16 | 0.76 |
| 34385 | 1 | 0.08 | 0.16 |

## False positives of BERTurk INT8 (clean, predicted offensive) — 151 tweets

| id | label | p INT8 | p tf-idf |
|---|---|---|---|
| 42864 | 0 | 0.48 | 0.43 |
| 44390 | 0 | 0.67 | 0.45 |
| 14294 | 0 | 0.86 | 0.25 |
| 43277 | 0 | 0.39 | 0.77 |
| 34168 | 0 | 0.48 | 0.78 |
| 17109 | 0 | 0.83 | 0.64 |
| 16693 | 0 | 0.62 | 0.29 |
| 12002 | 0 | 0.45 | 0.55 |

## tf-idf right, BERTurk INT8 wrong — 123 tweets

| id | label | p INT8 | p tf-idf |
|---|---|---|---|
| 22312 | 0 | 0.36 | 0.17 |
| 42864 | 0 | 0.48 | 0.43 |
| 28333 | 1 | 0.05 | 0.72 |
| 19419 | 0 | 0.66 | 0.44 |
| 44566 | 0 | 0.81 | 0.53 |
| 31826 | 0 | 0.62 | 0.24 |
| 43271 | 1 | 0.27 | 0.66 |
| 40716 | 0 | 0.43 | 0.37 |

## BERTurk INT8 right, tf-idf and word2vec+FFN wrong — 116 tweets

| id | label | p INT8 | p tf-idf |
|---|---|---|---|
| 28988 | 1 | 0.40 | 0.27 |
| 18636 | 0 | 0.16 | 0.60 |
| 36755 | 0 | 0.18 | 0.68 |
| 21142 | 1 | 0.46 | 0.20 |
| 14818 | 1 | 0.42 | 0.32 |
| 37509 | 0 | 0.32 | 0.75 |
| 39877 | 0 | 0.24 | 0.81 |
| 35271 | 0 | 0.05 | 0.63 |
