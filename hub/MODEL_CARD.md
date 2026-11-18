---
language: tr
license: mit
library_name: transformers
pipeline_tag: text-classification
base_model: dbmdz/bert-base-turkish-uncased
datasets:
- offenseval2020_tr
tags:
- offensive-language
- chat-moderation
- quantization
---

# Turkish chat moderation models (SWE015 homework, use case 4)

> **Content warning:** these models detect offensive language; the training data contains offensive tweets.

Trained models of the GitHub project **[{github}]({github})**. The application downloads this repository
automatically on its first run; the code, training scripts and full documentation are in the GitHub repository.

Task: label a Turkish chat message as **clean** or **offensive** within 50 ms (mean) on one CPU core.

| Path | Model | Precision |
|---|---|---|
| `lexicon.json` | lexicon learned from the training split + edit-distance matcher (rule baseline) | rules |
| `tfidf.joblib` | tf-idf (word 1–2-grams + char 2–5-grams) + logistic regression | FP64 |
| `w2v.model`, `ffn.pt` | own word2vec (trained on the training split) + feedforward network | FP32 |
| `berturk-fp32/` | BERTurk fine-tuned for classification | FP32 |
| `berturk-int8/` | the same model after PyTorch dynamic INT8 quantization (`nn.Linear`) — used by the app | INT8 |

Each BERTurk folder has a `threshold.json`: the decision threshold tuned on the validation split.
`berturk-int8/model_int8.pt` is a state dict: build the model from `config.json`, apply
`torch.ao.quantization.quantize_dynamic` to `nn.Linear`, then `load_state_dict` (see `src/chatmod/predictors.py`
in the GitHub repository).

## How to use

The models are in sub-folders, so the generic "Use this model" snippet of this page does not apply. The full
pipeline (text normalization, all five models, thresholds) is `src/chatmod/predictors.py` in the GitHub
repository. The FP32 BERTurk alone:

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

repo = "mertcankaratas/turkish-chat-moderation"
tokenizer = AutoTokenizer.from_pretrained(repo, subfolder="berturk-fp32")
model = AutoModelForSequenceClassification.from_pretrained(repo, subfolder="berturk-fp32").eval()
with torch.no_grad():
    logits = model(**tokenizer("gg wp, çok iyi oynadınız", return_tensors="pt")).logits
p_offensive = torch.softmax(logits, -1)[0, 1].item()  # flagged when >= threshold in berturk-fp32/threshold.json
```

The app first normalizes the text (`src/chatmod/normalize.py`: Turkish lowercasing, leetspeak, repeated letters).

## Results (official OffensEval-TR 2020 test set)

{results}

## Training data and sources

- Çağrı Çöltekin. *A Corpus of Turkish Offensive Language on Social Media.* LREC 2020.
  https://coltekin.github.io/offensive-turkish/ — annotations CC-BY. Train / validation = official training
  split 90 / 10 (stratified, seed 42); the official test split is used only for the final comparison.
- Stefan Schweter. *BERTurk – BERT models for Turkish.* 2020. https://huggingface.co/dbmdz/bert-base-turkish-uncased

## Limitations

Trained on tweets, not game chat; implicit or sarcastic insults are often missed; the learned lexicon
inherits the dataset's biases. Intended for a course project and research, not for unsupervised moderation.
