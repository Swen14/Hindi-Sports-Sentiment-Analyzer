"""
Fine-tune MuRIL, IndicBERT v2 or XLM-RoBERTa on the REAL-WORLD dataset
(manually verified Hindi sports comments collected from YouTube).

Usage (from the project root):
    venv\\Scripts\\python research\\real_world\\train_real_world.py --model muril
    venv\\Scripts\\python research\\real_world\\train_real_world.py --model indicbert_v2
    venv\\Scripts\\python research\\real_world\\train_real_world.py --model xlm_roberta
    venv\\Scripts\\python research\\real_world\\train_real_world.py --model all
"""

import argparse
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)


# ============================================================
# 1. PATHS AND CONFIG
# ============================================================

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "dataset", "real_world")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "research", "results")

BASE_MODELS = {
    "muril": "google/muril-base-cased",
    "indicbert_v2": "ai4bharat/IndicBERTv2-MLM-only",
    "xlm_roberta": "FacebookAI/xlm-roberta-base"
}

DISPLAY_NAMES = {
    "muril": "MuRIL",
    "indicbert_v2": "IndicBERT v2",
    "xlm_roberta": "XLM-RoBERTa"
}

# Same settings as the synthetic-data models, except more epochs
# because the real-world dataset is smaller.
HYPERPARAMETERS = {
    "epochs": 5,
    "learning_rate": 2e-5,
    "train_batch_size": 8,
    "eval_batch_size": 16,
    "weight_decay": 0.01,
    "max_length": 128,
    "seed": 42
}

LABEL_NAMES = ["Negative", "Neutral", "Positive"]


# ============================================================
# 2. DATASET
# ============================================================

class SentimentDataset(Dataset):

    def __init__(self, dataframe, tokenizer):
        self.texts = dataframe["text"].astype(str).tolist()
        self.labels = dataframe["label"].astype(int).tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoding = self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=HYPERPARAMETERS["max_length"]
        )
        encoding["labels"] = self.labels[index]
        return encoding


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1
    }


# ============================================================
# 3. TRAIN ONE MODEL
# ============================================================

def train(model_key):

    print("\n==============================================")
    print(f"REAL-WORLD TRAINING: {DISPLAY_NAMES[model_key]}")
    print("==============================================")

    train_df = pd.read_csv(os.path.join(DATA_DIR, "real_train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "real_validation.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "real_test.csv"))

    print("Train / Validation / Test:", len(train_df), len(val_df), len(test_df))

    model_dir = os.path.join(HERE, model_key)
    output_dir = os.path.join(model_dir, "output")
    best_model_dir = os.path.join(model_dir, "best_model")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODELS[model_key])

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODELS[model_key],
        num_labels=3,
        id2label={0: "negative", 1: "neutral", 2: "positive"},
        label2id={"negative": 0, "neutral": 1, "positive": 2}
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=HYPERPARAMETERS["epochs"],
        per_device_train_batch_size=HYPERPARAMETERS["train_batch_size"],
        per_device_eval_batch_size=HYPERPARAMETERS["eval_batch_size"],
        learning_rate=HYPERPARAMETERS["learning_rate"],
        weight_decay=HYPERPARAMETERS["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=HYPERPARAMETERS["seed"]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=SentimentDataset(train_df, tokenizer),
        eval_dataset=SentimentDataset(val_df, tokenizer),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("WARNING: GPU not detected, training on CPU")

    start = time.time()
    trainer.train()
    training_minutes = (time.time() - start) / 60

    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    print("Best model saved to:", best_model_dir)

    # Validation history per epoch
    history = [
        {
            "epoch": round(entry["epoch"]),
            "val_macro_f1": round(entry["eval_macro_f1"], 4),
            "val_accuracy": round(entry["eval_accuracy"], 4),
            "val_loss": round(entry["eval_loss"], 4)
        }
        for entry in trainer.state.log_history
        if "eval_macro_f1" in entry
    ]

    # Test set evaluation
    prediction_output = trainer.predict(SentimentDataset(test_df, tokenizer))
    predictions = np.argmax(prediction_output.predictions, axis=-1)
    true_labels = prediction_output.label_ids

    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average="macro", zero_division=0
    )
    report = classification_report(
        true_labels, predictions, labels=[0, 1, 2],
        target_names=LABEL_NAMES, digits=4, zero_division=0
    )
    cm = confusion_matrix(true_labels, predictions, labels=[0, 1, 2])

    print(f"\nAccuracy        : {accuracy:.4f}")
    print(f"Macro Precision : {precision:.4f}")
    print(f"Macro Recall    : {recall:.4f}")
    print(f"Macro F1        : {f1:.4f}")
    print(report)
    print(cm)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(
        os.path.join(RESULTS_DIR, f"real_{model_key}_results.txt"),
        "w", encoding="utf-8"
    ) as f:
        f.write(f"{DISPLAY_NAMES[model_key]} Real-World Results\n")
        f.write("=" * 40 + "\n\n")
        f.write("Dataset: dataset/real_world (manually verified YouTube comments)\n")
        f.write(f"Base model: {BASE_MODELS[model_key]}\n")
        f.write(f"Train Size: {len(train_df)}\n")
        f.write(f"Validation Size: {len(val_df)}\n")
        f.write(f"Test Size: {len(test_df)}\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Macro Precision: {precision:.4f}\n")
        f.write(f"Macro Recall: {recall:.4f}\n")
        f.write(f"Macro F1: {f1:.4f}\n\n")
        f.write("Classification Report\n---------------------\n")
        f.write(report)
        f.write("\nConfusion Matrix\n----------------\n")
        f.write(str(cm))

    training_info = {
        "model_key": model_key,
        "base_model": BASE_MODELS[model_key],
        "train_size": len(train_df),
        "validation_size": len(val_df),
        "test_size": len(test_df),
        "hyperparameters": HYPERPARAMETERS,
        "training_minutes": round(training_minutes, 1),
        "best_val_macro_f1": max(h["val_macro_f1"] for h in history),
        "history": history,
        "test_accuracy": round(accuracy, 4),
        "test_macro_f1": round(f1, 4)
    }

    with open(
        os.path.join(model_dir, "training_info.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(training_info, f, indent=2)

    # Checkpoints (with optimizer state) are large and no longer needed
    shutil.rmtree(output_dir, ignore_errors=True)

    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 4. MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=list(BASE_MODELS) + ["all"],
        default="all"
    )
    args = parser.parse_args()

    keys = list(BASE_MODELS) if args.model == "all" else [args.model]

    for key in keys:
        train(key)

    print("\nREAL-WORLD TRAINING COMPLETED")
