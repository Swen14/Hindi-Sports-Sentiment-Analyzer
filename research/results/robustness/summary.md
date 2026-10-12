# Robustness results (macro F1 %, mean ± std over seeds 42, 7, 123)

| Model | Trained on | Split | Synthetic test | Real test (comment split) | Real test (video split) |
|---|---|---|---|---|---|
| MuRIL | Synthetic | – | 100.00 ± 0.00 | 46.05 ± 2.92 | 48.05 ± 2.39 |
| IndicBERT v2 | Synthetic | – | 100.00 ± 0.00 | 48.86 ± 6.17 | 50.98 ± 4.32 |
| XLM-RoBERTa | Synthetic | – | 99.97 ± 0.06 | 44.69 ± 1.48 | 44.19 ± 2.16 |
| MuRIL | Real | comment | 66.59 ± 6.79 | 70.16 ± 1.85 | – |
| IndicBERT v2 | Real | comment | 51.03 ± 11.81 | 74.89 ± 1.37 | – |
| XLM-RoBERTa | Real | comment | 66.91 ± 6.23 | 70.80 ± 0.62 | – |
| MuRIL | Real | video | 62.96 ± 7.74 | – | 67.76 ± 3.21 |
| IndicBERT v2 | Real | video | 57.55 ± 1.08 | – | 74.40 ± 0.85 |
| XLM-RoBERTa | Real | video | 62.49 ± 5.11 | – | 71.58 ± 0.28 |
| TF-IDF + LR | Synthetic | – | 100.00 | 36.04 | 33.40 |
| TF-IDF + LR | Real | comment | 38.29 | 68.07 | – |
| TF-IDF + LR | Real | video | 34.89 | – | 63.44 |

## Accuracy (%) on the real-world test set

| Model | Synthetic-trained (comment test) | Real-trained, comment split | Real-trained, video split |
|---|---|---|---|
| MuRIL | 57.71 ± 2.71 | 74.82 ± 1.24 | 74.93 ± 2.02 |
| IndicBERT v2 | 60.31 ± 2.02 | 78.49 ± 0.81 | 80.88 ± 0.68 |
| XLM-RoBERTa | 57.80 ± 3.56 | 75.90 ± 1.35 | 78.21 ± 0.83 |

## Neutral-class F1 (%) on the real-world test set

| Model | Real-trained, comment split | Real-trained, video split |
|---|---|---|
| MuRIL | 52.90 ± 4.39 | 44.91 ± 7.11 |
| IndicBERT v2 | 59.52 ± 4.21 | 52.73 ± 1.70 |
| XLM-RoBERTa | 50.78 ± 1.51 | 49.01 ± 2.52 |
