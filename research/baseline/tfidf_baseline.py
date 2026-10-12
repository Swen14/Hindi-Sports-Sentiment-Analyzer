"""
Classical baseline: TF-IDF + Logistic Regression.

Answers the question "do we need transformers at all?". Two feature sets are
combined, as is common for noisy, morphologically rich text:
  - word unigrams and bigrams
  - character 2-5-grams (robust to spelling variation and emojis)

The baseline is trained on each training set (synthetic, real comment-level
split, real video-level split) and evaluated on the same test sets as the
transformer models. C is chosen on the validation set.

Usage (from the project root, after run_robustness.py has written
research/results/robustness/video_split_assignment.csv):
    venv\\Scripts\\python research\\baseline\\tfidf_baseline.py
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
REAL_DIR = os.path.join(PROJECT_ROOT, "dataset", "real_world")
SYN_DIR = os.path.join(PROJECT_ROOT, "dataset")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "research", "results")
VIDEO_SPLIT = os.path.join(RESULTS_DIR, "robustness", "video_split_assignment.csv")

LABEL_NAMES = ["Negative", "Neutral", "Positive"]
C_VALUES = [0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]


def read(path):
    df = pd.read_csv(path)
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    return df


def load_splits():
    synthetic = {s: read(os.path.join(SYN_DIR, f"final_{s}.csv"))
                 for s in ["train", "validation", "test"]}
    comment = {s: read(os.path.join(REAL_DIR, f"real_{s}.csv"))
               for s in ["train", "validation", "test"]}

    data = read(os.path.join(REAL_DIR, "real_world_dataset.csv"))
    assignment = pd.read_csv(VIDEO_SPLIT)[["comment_id", "video_split"]]
    data = data.merge(assignment, on="comment_id")
    video = {s: data[data["video_split"] == s].reset_index(drop=True)
             for s in ["train", "validation", "test"]}
    return {"synthetic": synthetic, "real_comment": comment, "real_video": video}


def scores(true_labels, predictions):
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average="macro", zero_division=0
    )
    _, _, class_f1, _ = precision_recall_fscore_support(
        true_labels, predictions, labels=[0, 1, 2], average=None, zero_division=0
    )
    return {
        "samples": int(len(true_labels)),
        "accuracy": round(float(accuracy_score(true_labels, predictions)), 4),
        "macro_precision": round(float(precision), 4),
        "macro_recall": round(float(recall), 4),
        "macro_f1": round(float(f1), 4),
        "class_f1": {n: round(float(v), 4) for n, v in zip(LABEL_NAMES, class_f1)},
        "confusion_matrix": confusion_matrix(true_labels, predictions, labels=[0, 1, 2]).tolist()
    }


class TfidfLR:

    def __init__(self, C):
        self.word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                                    sublinear_tf=True, token_pattern=r"\S+")
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                    sublinear_tf=True)
        self.clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")

    def features(self, texts, fit=False):
        if fit:
            return hstack([self.word.fit_transform(texts), self.char.fit_transform(texts)]).tocsr()
        return hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()

    def fit(self, texts, labels):
        self.clf.fit(self.features(texts, fit=True), labels)
        return self

    def predict(self, texts):
        return self.clf.predict(self.features(texts))


def main():
    splits = load_splits()
    test_sets = {
        "synthetic": splits["synthetic"]["test"],
        "real_comment": splits["real_comment"]["test"],
        "real_video": splits["real_video"]["test"]
    }
    results = {}

    for train_name, data in splits.items():
        best_c, best_f1 = None, -1
        for C in C_VALUES:
            model = TfidfLR(C).fit(data["train"]["text"], data["train"]["label"])
            f1 = scores(data["validation"]["label"], model.predict(data["validation"]["text"]))["macro_f1"]
            if f1 > best_f1:
                best_c, best_f1 = C, f1

        model = TfidfLR(best_c).fit(data["train"]["text"], data["train"]["label"])
        evaluated = {}
        for test_name, df in test_sets.items():
            # A model trained on one real split is not evaluated on the other
            # real split, because the two share comments (train/test overlap).
            if train_name.startswith("real") and test_name.startswith("real") and test_name != train_name:
                continue
            evaluated[test_name] = scores(df["label"].to_numpy(), model.predict(df["text"]))

        results[train_name] = {"C": best_c, "val_macro_f1": best_f1, "test": evaluated}
        print(train_name, "C =", best_c, {k: v["macro_f1"] for k, v in evaluated.items()})

    os.makedirs(os.path.join(RESULTS_DIR, "baseline"), exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "baseline", "tfidf_lr_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
