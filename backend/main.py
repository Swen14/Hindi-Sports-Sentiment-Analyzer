from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

from .model_loader import (
    predict_sentiment,
    MODEL_PATHS,
    load_model
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

DATASET_DIR = (
    PROJECT_ROOT / "dataset"
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(

    title="Hindi Sports Sentiment Analyzer",

    description=(
        "Hindi / Hinglish Sports Sentiment Analyzer "
        "with selectable transformer models and "
        "model evaluation."
    ),

    version="4.0"

)


# ============================================================
# CORS
# ============================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]

)


# ============================================================
# REQUEST MODEL
# ============================================================

class SentimentRequest(BaseModel):

    text: str

    model: str = "old_muril"


# ============================================================
# MODEL INFORMATION
# ============================================================

MODEL_INFO = {

    "old_muril": {
        "name": "Original MuRIL",
        "dataset": "Original test dataset"
    },

    "research_muril": {
        "name": "Research MuRIL",
        "dataset": "Research_12000.csv"
    },

    "indicbert_v2": {
        "name": "IndicBERT v2",
        "dataset": "Research_12000.csv"
    },

    "xlm_roberta": {
        "name": "XLM-RoBERTa",
        "dataset": "Research_12000.csv"
    }

}


# ============================================================
# LABELS
# ============================================================

LABEL_NAMES = [
    "Negative",
    "Neutral",
    "Positive"
]


LABEL_MAP = {

    "negative": 0,
    "neutral": 1,
    "positive": 2

}


# ============================================================
# LOAD DATASET FOR EVALUATION
# ============================================================

def load_evaluation_dataset(model_name):

    # --------------------------------------------------------
    # ORIGINAL MuRIL
    # --------------------------------------------------------

    if model_name == "old_muril":

        test_path = (
            DATASET_DIR / "test.csv"
        )

        if not test_path.exists():

            raise FileNotFoundError(
                f"Original test dataset not found: "
                f"{test_path}"
            )

        df = pd.read_csv(
            test_path
        )

        dataset_name = "test.csv"


    # --------------------------------------------------------
    # THREE RESEARCH MODELS
    # --------------------------------------------------------

    else:

        research_path = (
            DATASET_DIR /
            "Research_12000.csv"
        )

        if not research_path.exists():

            raise FileNotFoundError(
                f"Research dataset not found: "
                f"{research_path}"
            )

        df = pd.read_csv(
            research_path
        )

        # Research dataset contains
        # train / validation / test

        if "split" not in df.columns:

            raise ValueError(
                "Research_12000.csv does not "
                "contain a 'split' column."
            )

        df = df[
            df["split"].astype(str).str.lower()
            == "test"
        ].copy()

        dataset_name = "Research_12000.csv (test split)"


    if len(df) == 0:

        raise ValueError(
            "No test samples found."
        )


    # --------------------------------------------------------
    # FIND TEXT COLUMN
    # --------------------------------------------------------

    text_column = None

    for column in [
        "text",
        "sentence",
        "comment"
    ]:

        if column in df.columns:

            text_column = column
            break


    if text_column is None:

        raise ValueError(
            "Could not find text column. "
            "Expected one of: text, sentence, comment."
        )


    # --------------------------------------------------------
    # FIND LABEL COLUMN
    # --------------------------------------------------------

    label_column = None

    for column in [
        "label",
        "sentiment"
    ]:

        if column in df.columns:

            label_column = column
            break


    if label_column is None:

        raise ValueError(
            "Could not find label column. "
            "Expected label or sentiment."
        )


    texts = (
        df[text_column]
        .fillna("")
        .astype(str)
        .tolist()
    )


    raw_labels = (
        df[label_column]
        .tolist()
    )


    labels = []


    for value in raw_labels:

        # Numeric labels

        if isinstance(
            value,
            (int, np.integer)
        ):

            labels.append(
                int(value)
            )

            continue


        if isinstance(
            value,
            (float, np.floating)
        ):

            labels.append(
                int(value)
            )

            continue


        # String labels

        value_string = (
            str(value)
            .strip()
            .lower()
        )


        if value_string in LABEL_MAP:

            labels.append(
                LABEL_MAP[value_string]
            )

        else:

            try:

                labels.append(
                    int(float(value_string))
                )

            except ValueError:

                raise ValueError(
                    f"Unknown sentiment label: "
                    f"{value}"
                )


    return (
        texts,
        np.array(labels),
        dataset_name
    )


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(model_name):

    if model_name not in MODEL_PATHS:

        raise ValueError(
            f"Invalid model: {model_name}"
        )


    # --------------------------------------------------------
    # LOAD TEST DATA
    # --------------------------------------------------------

    texts, true_labels, dataset_name = (
        load_evaluation_dataset(
            model_name
        )
    )


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    tokenizer, model = load_model(
        model_name
    )


    device = next(
        model.parameters()
    ).device


    all_probabilities = []


    # --------------------------------------------------------
    # BATCH PREDICTION
    # --------------------------------------------------------

    batch_size = 16


    model.eval()


    with torch.no_grad():

        for start in range(
            0,
            len(texts),
            batch_size
        ):

            batch_texts = texts[
                start:
                start + batch_size
            ]


            inputs = tokenizer(
                batch_texts,

                return_tensors="pt",

                padding=True,

                truncation=True,

                max_length=128
            )


            inputs = {
                key: value.to(device)

                for key, value
                in inputs.items()
            }


            outputs = model(
                **inputs
            )


            probabilities = torch.softmax(
                outputs.logits,
                dim=-1
            )


            all_probabilities.append(
                probabilities
                .cpu()
                .numpy()
            )


    probabilities = np.concatenate(
        all_probabilities,
        axis=0
    )


    predictions = np.argmax(
        probabilities,
        axis=1
    )


    # ========================================================
    # BASIC METRICS
    # ========================================================

    accuracy = accuracy_score(
        true_labels,
        predictions
    )


    precision, recall, f1, _ = (
        precision_recall_fscore_support(

            true_labels,

            predictions,

            average="macro",

            zero_division=0
        )
    )


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    report = classification_report(

        true_labels,

        predictions,

        labels=[
            0,
            1,
            2
        ],

        target_names=LABEL_NAMES,

        output_dict=True,

        zero_division=0
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(

        true_labels,

        predictions,

        labels=[
            0,
            1,
            2
        ]

    )


    # ========================================================
    # ROC
    # ========================================================

    roc_data = {}

    roc_auc = {}


    for class_id, class_name in enumerate(
        LABEL_NAMES
    ):

        binary_labels = (
            true_labels == class_id
        ).astype(int)


        # ROC curve

        fpr, tpr, _ = roc_curve(

            binary_labels,

            probabilities[:, class_id]
        )


        # AUC

        auc_value = roc_auc_score(

            binary_labels,

            probabilities[:, class_id]
        )


        # ----------------------------------------------------
        # Reduce number of points for frontend
        # ----------------------------------------------------

        max_points = 100


        if len(fpr) > max_points:

            indexes = np.linspace(

                0,

                len(fpr) - 1,

                max_points,

                dtype=int
            )

            fpr = fpr[indexes]

            tpr = tpr[indexes]


        roc_data[
            class_name.lower()
        ] = [

            {
                "fpr": float(x),
                "tpr": float(y)
            }

            for x, y
            in zip(fpr, tpr)

        ]


        roc_auc[
            class_name.lower()
        ] = float(
            auc_value
        )


    # ========================================================
    # RETURN
    # ========================================================

    return {

        "model": model_name,

        "model_name":
            MODEL_INFO[model_name]["name"],

        "dataset":
            dataset_name,

        "test_samples":
            int(len(true_labels)),

        "accuracy":
            float(accuracy),

        "macro_precision":
            float(precision),

        "macro_recall":
            float(recall),

        "macro_f1":
            float(f1),

        "classification_report":
            report,

        "confusion_matrix":
            cm.tolist(),

        "roc_auc":
            roc_auc,

        "roc":
            roc_data

    }


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {

        "message":
            "Hindi Sports Sentiment Analyzer API",

        "available_models":
            list(MODEL_PATHS.keys()),

        "default_model":
            "old_muril",

        "labels": {
            "0": "Negative",
            "1": "Neutral",
            "2": "Positive"
        }

    }


# ============================================================
# AVAILABLE MODELS
# ============================================================

@app.get("/models")
def models():

    return {

        "models": [

            {
                "id": "old_muril",
                "name": "Original MuRIL"
            },

            {
                "id": "research_muril",
                "name": "Research MuRIL"
            },

            {
                "id": "indicbert_v2",
                "name": "IndicBERT v2"
            },

            {
                "id": "xlm_roberta",
                "name": "XLM-RoBERTa"
            }

        ]

    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {

        "status":
            "running",

        "available_models":
            list(MODEL_PATHS.keys())

    }


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
def predict(
    request: SentimentRequest
):

    try:

        if not request.text.strip():

            raise HTTPException(

                status_code=400,

                detail="Text cannot be empty."

            )


        if request.model not in MODEL_PATHS:

            raise HTTPException(

                status_code=400,

                detail=(
                    f"Invalid model. "
                    f"Available models: "
                    f"{list(MODEL_PATHS.keys())}"
                )

            )


        result = predict_sentiment(

            request.text,

            request.model

        )


        return result


    except HTTPException:

        raise


    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=str(e)

        )


# ============================================================
# MODEL EVALUATION
# ============================================================

@app.get(
    "/evaluation/{model_name}"
)
def evaluation(
    model_name: str
):

    try:

        if model_name not in MODEL_PATHS:

            raise HTTPException(

                status_code=400,

                detail=(
                    f"Invalid model. "
                    f"Available models: "
                    f"{list(MODEL_PATHS.keys())}"
                )

            )


        result = evaluate_model(
            model_name
        )


        return result


    except HTTPException:

        raise


    except Exception as e:

        print(
            "\nEvaluation error:"
        )

        print(e)


        raise HTTPException(

            status_code=500,

            detail=str(e)

        )