"""
Single source of truth for the six models shown in the app:
three trained on synthetic (AI-generated) data and three trained on
real-world YouTube comments.
"""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIR = PROJECT_ROOT / "research"
RESULTS_DIR = RESEARCH_DIR / "results"
EVALUATION_CACHE_DIR = RESULTS_DIR / "evaluation"


# ============================================================
# TEST SETS
# ============================================================

TEST_SETS = {

    "synthetic": {
        "name": "Synthetic test set",
        "file": "dataset/final_test.csv",
        "description": (
            "Held-out split of NLP_Project_Final_Dataset.csv, "
            "template-generated Hindi sports sentences."
        )
    },

    "real": {
        "name": "Real-world test set",
        "file": "dataset/real_world/real_test.csv",
        "description": (
            "Held-out split of manually verified Hindi "
            "sports comments collected from YouTube."
        )
    }

}


# ============================================================
# TRAINING DATA GROUPS
# ============================================================

TRAINING_GROUPS = {

    "synthetic": {
        "title": "Trained on synthetic data",
        "dataset": "NLP_Project_Final_Dataset.csv",
        "description": (
            "10,000 AI-generated Hindi sports sentences built "
            "from repeated templates (cricket, football, "
            "kabaddi, badminton, F1). Balanced classes."
        ),
        "train_size": 8000,
        "validation_size": 1000,
        "test_size": 1000,
        "hyperparameters": {
            "epochs": 3,
            "learning_rate": 2e-5,
            "train_batch_size": 8,
            "eval_batch_size": 16,
            "weight_decay": 0.01,
            "max_length": 128,
            "seed": 42
        }
    },

    "real": {
        "title": "Trained on real-world data",
        "dataset": "dataset/real_world/real_world_dataset.csv",
        "description": (
            "3,720 real Hindi (Devanagari) comments from YouTube "
            "sports videos, labeled and manually verified. "
            "Imbalanced classes (mostly negative)."
        ),
        "train_size": 2976,
        "validation_size": 372,
        "test_size": 372,
        "hyperparameters": {
            "epochs": 5,
            "learning_rate": 2e-5,
            "train_batch_size": 8,
            "eval_batch_size": 16,
            "weight_decay": 0.01,
            "max_length": 128,
            "seed": 42
        }
    }

}


# ============================================================
# ARCHITECTURES
# ============================================================

ARCHITECTURES = {

    "muril": {
        "name": "MuRIL",
        "base_model": "google/muril-base-cased",
        "architecture": "BERT-base (12 layers, 768 hidden, 12 heads)",
        "pretraining": "Google, 17 Indian languages + transliterated text",
        "tokenizer": "WordPiece"
    },

    "indicbert_v2": {
        "name": "IndicBERT v2",
        "base_model": "ai4bharat/IndicBERTv2-MLM-only",
        "architecture": "BERT-base (12 layers, 768 hidden, 12 heads)",
        "pretraining": "AI4Bharat, IndicCorp v2 (24 Indian languages)",
        "tokenizer": "WordPiece"
    },

    "xlm_roberta": {
        "name": "XLM-RoBERTa",
        "base_model": "FacebookAI/xlm-roberta-base",
        "architecture": "RoBERTa-base (12 layers, 768 hidden, 12 heads)",
        "pretraining": "Meta AI, CommonCrawl in 100 languages",
        "tokenizer": "SentencePiece"
    }

}


# ============================================================
# MODELS
# ============================================================

MODELS = {

    "research_muril": {
        "architecture": "muril",
        "group": "synthetic",
        "path": RESEARCH_DIR / "muril" / "best_model"
    },

    "indicbert_v2": {
        "architecture": "indicbert_v2",
        "group": "synthetic",
        "path": RESEARCH_DIR / "indicbert_v2" / "best_model"
    },

    "xlm_roberta": {
        "architecture": "xlm_roberta",
        "group": "synthetic",
        "path": RESEARCH_DIR / "xlm_roberta" / "best_model"
    },

    "real_muril": {
        "architecture": "muril",
        "group": "real",
        "path": RESEARCH_DIR / "real_world" / "muril" / "best_model"
    },

    "real_indicbert_v2": {
        "architecture": "indicbert_v2",
        "group": "real",
        "path": RESEARCH_DIR / "real_world" / "indicbert_v2" / "best_model"
    },

    "real_xlm_roberta": {
        "architecture": "xlm_roberta",
        "group": "real",
        "path": RESEARCH_DIR / "real_world" / "xlm_roberta" / "best_model"
    }

}


def training_info_path(model_id: str) -> Path:
    return MODELS[model_id]["path"].parent / "training_info.json"


def evaluation_cache_path(model_id: str, test_set: str) -> Path:
    return EVALUATION_CACHE_DIR / f"{model_id}__{test_set}.json"


def read_json(path: Path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def model_summary(model_id: str) -> dict:
    """Everything the frontend needs to describe one model."""

    entry = MODELS[model_id]
    architecture = ARCHITECTURES[entry["architecture"]]
    group = TRAINING_GROUPS[entry["group"]]
    training_info = read_json(training_info_path(model_id)) or {}

    scores = {}
    for test_set in TEST_SETS:
        cached = read_json(evaluation_cache_path(model_id, test_set))
        if cached:
            scores[test_set] = {
                "accuracy": cached["accuracy"],
                "macro_f1": cached["macro_f1"]
            }

    return {
        "id": model_id,
        "name": architecture["name"],
        "group": entry["group"],
        "available": entry["path"].exists(),
        "architecture": architecture,
        "training": {
            "dataset": group["dataset"],
            "train_size": group["train_size"],
            "validation_size": group["validation_size"],
            "test_size": group["test_size"],
            "hyperparameters": group["hyperparameters"],
            "parameters": training_info.get("parameters"),
            "training_minutes": training_info.get("training_minutes"),
            "history": training_info.get("history", [])
        },
        "own_test_set": entry["group"],
        "scores": scores
    }
