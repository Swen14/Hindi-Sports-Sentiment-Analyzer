import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

from sklearn.preprocessing import label_binarize


# ============================================================
# PATHS
# ============================================================

BACKEND_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_ROOT.parent

RESEARCH_DATASET = (
    BACKEND_ROOT
    / "dataset"
    / "Research_12000.csv"
)

ORIGINAL_TEST_DATA = (
    BACKEND_ROOT
    / "dataset"
    / "test.csv"
)

def resolve_model_path(primary_path: Path, fallback_path: Path | None = None) -> Path:
    if primary_path.exists():
        return primary_path
    if fallback_path is not None and fallback_path.exists():
        return fallback_path
    return primary_path


MODEL_PATHS = {
    "old_muril": resolve_model_path(
        PROJECT_ROOT
        / "model"
        / "muril_sentiment_model",
        PROJECT_ROOT
        / "research"
        / "muril"
        / "best_model"
    ),

    "research_muril":
        PROJECT_ROOT
        / "research"
        / "muril"
        / "best_model",

    "indicbert_v2":
        PROJECT_ROOT
        / "research"
        / "indicbert_v2"
        / "best_model",

    "xlm_roberta":
        PROJECT_ROOT
        / "research"
        / "xlm_roberta"
        / "best_model"
}


MODEL_NAMES = {
    "old_muril": "Original MuRIL",
    "research_muril": "Research MuRIL",
    "indicbert_v2": "IndicBERT v2",
    "xlm_roberta": "XLM-RoBERTa"
}


ID2LABEL = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CACHE
# ============================================================

evaluation_cache = {}


# ============================================================
# LOAD RESEARCH DATASET
# ============================================================

def load_research_test_data():

    df = pd.read_csv(RESEARCH_DATASET)

    test_df = df[
        df["split"] == "test"
    ].copy()

    if len(test_df) == 0:
        raise ValueError(
            "No test rows found in Research_12000.csv"
        )

    return (
        test_df["text"].astype(str).tolist(),
        test_df["label"].astype(int).to_numpy()
    )


# ============================================================
# LOAD ORIGINAL TEST DATA
# ============================================================

def load_original_test_data():

    # --------------------------------------------------------
    # Your original model uses the original dataset pipeline.
    #
    # We first try test.csv.
    # --------------------------------------------------------

    df = pd.read_csv(ORIGINAL_TEST_DATA)

    print("\nOriginal test columns:")
    print(df.columns.tolist())

    # Common column names used in your project
    if "sentence" in df.columns:
        text_column = "sentence"

    elif "text" in df.columns:
        text_column = "text"

    else:
        raise ValueError(
            "Could not find text column in test.csv"
        )

    if "label" in df.columns:
        label_column = "label"

    elif "sentiment" in df.columns:

        sentiment_map = {
            "negative": 0,
            "neutral": 1,
            "positive": 2,
            "Negative": 0,
            "Neutral": 1,
            "Positive": 2
        }

        df["label"] = (
            df["sentiment"]
            .map(sentiment_map)
        )

        label_column = "label"

    else:
        raise ValueError(
            "Could not find label column in test.csv"
        )

    return (
        df[text_column].astype(str).tolist(),
        df[label_column].astype(int).to_numpy()
    )


# ============================================================
# LOAD DATA FOR MODEL
# ============================================================

def load_test_data(model_name):

    if model_name == "old_muril":
        return load_original_test_data()

    return load_research_test_data()


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(model_name):

    if model_name not in MODEL_PATHS:
        raise ValueError(
            f"Invalid model: {model_name}"
        )

    # --------------------------------------------------------
    # Return cached result if already evaluated
    # --------------------------------------------------------

    if model_name in evaluation_cache:

        print(
            f"Returning cached evaluation for "
            f"{model_name}"
        )

        return evaluation_cache[model_name]

    model_path = MODEL_PATHS[model_name]

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model folder not found: {model_path}"
        )

    print("\n==============================================")
    print("MODEL EVALUATION")
    print("==============================================")

    print("Model:", MODEL_NAMES[model_name])
    print("Path:", model_path)
    print("Device:", device)

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path)
    )

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            str(model_path)
        )
    )

    model.to(device)
    model.eval()

    # --------------------------------------------------------
    # Load correct test dataset
    # --------------------------------------------------------

    texts, true_labels = load_test_data(
        model_name
    )

    print("Test samples:", len(texts))

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    all_predictions = []
    all_probabilities = []

    batch_size = 16

    for start in range(
        0,
        len(texts),
        batch_size
    ):

        batch_texts = texts[
            start:start + batch_size
        ]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = model(
                **inputs
            )

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1
            )

            predictions = torch.argmax(
                probabilities,
                dim=-1
            )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )

    predictions = np.array(
        all_predictions
    )

    probabilities = np.array(
        all_probabilities
    )

    true_labels = np.array(
        true_labels
    )

    # ========================================================
    # BASIC METRICS
    # ========================================================

    accuracy = accuracy_score(
        true_labels,
        predictions
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            predictions,
            average="macro",
            zero_division=0
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            predictions,
            average="weighted",
            zero_division=0
        )
    )

    # ========================================================
    # PER CLASS METRICS
    # ========================================================

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            true_labels,
            predictions,
            labels=[0, 1, 2],
            zero_division=0
        )
    )

    class_metrics = {}

    for class_id in range(3):

        class_metrics[
            ID2LABEL[class_id]
        ] = {
            "precision": round(
                float(precision[class_id]),
                4
            ),
            "recall": round(
                float(recall[class_id]),
                4
            ),
            "f1": round(
                float(f1[class_id]),
                4
            ),
            "support": int(
                support[class_id]
            )
        }

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        true_labels,
        predictions,
        labels=[0, 1, 2]
    )

    # ========================================================
    # ROC / AUC
    # ========================================================

    binary_labels = label_binarize(
        true_labels,
        classes=[0, 1, 2]
    )

    roc_data = {}

    for class_id in range(3):

        fpr, tpr, _ = roc_curve(
            binary_labels[:, class_id],
            probabilities[:, class_id]
        )

        class_auc = auc(
            fpr,
            tpr
        )

        roc_data[
            ID2LABEL[class_id]
        ] = {
            "fpr": [
                round(float(x), 5)
                for x in fpr
            ],
            "tpr": [
                round(float(x), 5)
                for x in tpr
            ],
            "auc": round(
                float(class_auc),
                4
            )
        }

    # ========================================================
    # RESULT
    # ========================================================

    result = {

        "model": model_name,

        "model_name":
            MODEL_NAMES[model_name],

        "dataset":
            (
                "Original Dataset"
                if model_name == "old_muril"
                else "Research_12000 Dataset"
            ),

        "test_samples":
            int(len(true_labels)),

        "accuracy":
            round(float(accuracy), 4),

        "macro_precision":
            round(float(macro_precision), 4),

        "macro_recall":
            round(float(macro_recall), 4),

        "macro_f1":
            round(float(macro_f1), 4),

        "weighted_precision":
            round(float(weighted_precision), 4),

        "weighted_recall":
            round(float(weighted_recall), 4),

        "weighted_f1":
            round(float(weighted_f1), 4),

        "classes":
            class_metrics,

        "confusion_matrix":
            cm.tolist(),

        "roc":
            roc_data
    }

    # --------------------------------------------------------
    # Cache
    # --------------------------------------------------------

    evaluation_cache[
        model_name
    ] = result

    # --------------------------------------------------------
    # Free memory
    # --------------------------------------------------------

    del model
    del tokenizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        f"\nEvaluation completed: "
        f"{MODEL_NAMES[model_name]}"
    )

    return result