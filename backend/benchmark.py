"""
Evaluate a model on one of the test sets and cache the result as JSON.
Used by the API (/evaluation) and by research/evaluate_all.py.
"""

import json

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

from .model_loader import load_model
from .model_registry import (
    PROJECT_ROOT,
    MODELS,
    TEST_SETS,
    ARCHITECTURES,
    evaluation_cache_path,
    read_json
)


LABEL_NAMES = ["Negative", "Neutral", "Positive"]

HINDI_LABELS = {
    "नकारात्मक": 0,
    "तटस्थ": 1,
    "सकारात्मक": 2,
    "negative": 0,
    "neutral": 1,
    "positive": 2
}


def load_test_set(test_set: str):

    if test_set not in TEST_SETS:
        raise ValueError(
            f"Unknown test set '{test_set}'. "
            f"Available: {list(TEST_SETS)}"
        )

    df = pd.read_csv(PROJECT_ROOT / TEST_SETS[test_set]["file"])

    if "label" in df.columns:
        labels = df["label"].astype(int).to_numpy()
    else:
        labels = (
            df["sentiment"].astype(str).str.strip().str.lower()
            .map(HINDI_LABELS).astype(int).to_numpy()
        )

    return df["text"].fillna("").astype(str).tolist(), labels


def predict_probabilities(model_id: str, texts, batch_size: int = 16):

    tokenizer, model = load_model(model_id)
    device = next(model.parameters()).device
    batches = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            inputs = tokenizer(
                texts[start:start + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            logits = model(**inputs).logits
            batches.append(torch.softmax(logits, dim=-1).cpu().numpy())

    return np.concatenate(batches, axis=0)


def count_parameters(model_id: str) -> int:
    _, model = load_model(model_id)
    return int(sum(p.numel() for p in model.parameters()))


def compute_evaluation(model_id: str, test_set: str) -> dict:

    texts, true_labels = load_test_set(test_set)
    probabilities = predict_probabilities(model_id, texts)
    predictions = np.argmax(probabilities, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average="macro", zero_division=0
    )

    report = classification_report(
        true_labels, predictions, labels=[0, 1, 2],
        target_names=LABEL_NAMES, output_dict=True, zero_division=0
    )

    cm = confusion_matrix(true_labels, predictions, labels=[0, 1, 2])

    roc_data = {}
    roc_auc = {}

    for class_id, class_name in enumerate(LABEL_NAMES):

        binary_labels = (true_labels == class_id).astype(int)
        fpr, tpr, _ = roc_curve(binary_labels, probabilities[:, class_id])

        # Keep the payload small for the frontend
        if len(fpr) > 100:
            indexes = np.linspace(0, len(fpr) - 1, 100, dtype=int)
            fpr, tpr = fpr[indexes], tpr[indexes]

        roc_data[class_name.lower()] = [
            {"fpr": float(x), "tpr": float(y)} for x, y in zip(fpr, tpr)
        ]
        roc_auc[class_name.lower()] = float(
            roc_auc_score(binary_labels, probabilities[:, class_id])
        )

    entry = MODELS[model_id]

    return {
        "model": model_id,
        "model_name": ARCHITECTURES[entry["architecture"]]["name"],
        "trained_on": entry["group"],
        "test_set": test_set,
        "dataset": TEST_SETS[test_set]["name"],
        "cross_domain": test_set != entry["group"],
        "test_samples": int(len(true_labels)),
        "accuracy": float(accuracy_score(true_labels, predictions)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "roc_auc": roc_auc,
        "roc": roc_data
    }


def get_evaluation(model_id: str, test_set: str, refresh: bool = False) -> dict:

    path = evaluation_cache_path(model_id, test_set)

    if not refresh:
        cached = read_json(path)
        if cached:
            return cached

    result = compute_evaluation(model_id, test_set)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)

    return result
