"""
Robustness experiments for the paper:

  video_split      Re-split the real-world data by VIDEO (no video appears in more
                   than one split) and fine-tune the three models with 3 seeds.
  seeds_real       Fine-tune the three real-world models on the original
                   comment-level split with two extra seeds (seed 42 = the
                   models already in research/real_world/).
  seeds_synthetic  Fine-tune the three synthetic-data models with two extra seeds
                   (seed 42 = the models already in research/<model>/).
  existing         Evaluate the existing seed-42 synthetic-data models on the
                   video-level real-world test set.

Every run writes one JSON file to research/results/robustness/<experiment>/.
Runs whose JSON already exists are skipped, so the script can be resumed.
Model weights of the extra runs are NOT kept (only the scores).

Usage (from the project root):
    venv\\Scripts\\python research\\robustness\\run_robustness.py --experiment all
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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
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
REAL_DIR = os.path.join(PROJECT_ROOT, "dataset", "real_world")
SYN_DIR = os.path.join(PROJECT_ROOT, "dataset")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "research", "results", "robustness")
TMP_DIR = os.path.join(HERE, "_tmp_output")

BASE_MODELS = {
    "muril": "google/muril-base-cased",
    "indicbert_v2": "ai4bharat/IndicBERTv2-MLM-only",
    "xlm_roberta": "FacebookAI/xlm-roberta-base"
}

SEEDS = [42, 7, 123]

# Same hyper-parameters as the models in the paper
COMMON = {
    "learning_rate": 2e-5,
    "train_batch_size": 8,
    "eval_batch_size": 16,
    "weight_decay": 0.01,
    "max_length": 128
}
EPOCHS = {"real": 5, "synthetic": 3}

LABEL_NAMES = ["Negative", "Neutral", "Positive"]


# ============================================================
# 2. DATA
# ============================================================

def read(path):
    df = pd.read_csv(path)
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    return df


def real_comment_split():
    return {s: read(os.path.join(REAL_DIR, f"real_{s}.csv"))
            for s in ["train", "validation", "test"]}


def synthetic_split():
    return {s: read(os.path.join(SYN_DIR, f"final_{s}.csv"))
            for s in ["train", "validation", "test"]}


def real_video_split():
    """
    80/10/10 split in which all comments of a video stay in the same split.
    StratifiedGroupKFold (10 folds) keeps the class balance close to the
    full dataset: fold 0 = test, fold 1 = validation, folds 2-9 = train.
    """
    data = read(os.path.join(REAL_DIR, "real_world_dataset.csv"))
    folds = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    fold_of = np.zeros(len(data), dtype=int)
    for fold, (_, idx) in enumerate(folds.split(data, data["label"], data["video_id"])):
        fold_of[idx] = fold

    split = np.where(fold_of == 0, "test", np.where(fold_of == 1, "validation", "train"))
    data["video_split"] = split

    # Publish the assignment (IDs only, no comment text)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    data[["comment_id", "video_id", "label", "video_split"]].to_csv(
        os.path.join(RESULTS_DIR, "video_split_assignment.csv"), index=False
    )

    parts = {s: data[data["video_split"] == s].reset_index(drop=True)
             for s in ["train", "validation", "test"]}

    # Sanity check: no video in two splits
    videos = {s: set(df["video_id"]) for s, df in parts.items()}
    assert not (videos["train"] & videos["test"])
    assert not (videos["train"] & videos["validation"])
    assert not (videos["validation"] & videos["test"])
    return parts


class SentimentDataset(Dataset):

    def __init__(self, dataframe, tokenizer):
        self.texts = dataframe["text"].tolist()
        self.labels = dataframe["label"].tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoding = self.tokenizer(
            self.texts[index], truncation=True, max_length=COMMON["max_length"]
        )
        encoding["labels"] = self.labels[index]
        return encoding


# ============================================================
# 3. METRICS
# ============================================================

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
        "class_f1": {name: round(float(v), 4) for name, v in zip(LABEL_NAMES, class_f1)},
        "confusion_matrix": confusion_matrix(true_labels, predictions, labels=[0, 1, 2]).tolist()
    }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    _, _, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    return {"accuracy": accuracy_score(labels, predictions), "macro_f1": f1}


def evaluate(trainer, tokenizer, test_sets):
    results = {}
    for name, df in test_sets.items():
        output = trainer.predict(SentimentDataset(df, tokenizer))
        predictions = np.argmax(output.predictions, axis=-1)
        results[name] = scores(output.label_ids, predictions)
    return results


# ============================================================
# 4. ONE TRAINING RUN
# ============================================================

def run(experiment, model_key, seed, data, train_kind, test_sets):

    out_dir = os.path.join(RESULTS_DIR, experiment)
    out_file = os.path.join(out_dir, f"{model_key}_seed{seed}.json")
    if os.path.exists(out_file):
        print("skip (done):", out_file)
        return

    print(f"\n=== {experiment} | {model_key} | seed {seed} ===")
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODELS[model_key])
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODELS[model_key],
        num_labels=3,
        id2label={0: "negative", 1: "neutral", 2: "positive"},
        label2id={"negative": 0, "neutral": 1, "positive": 2}
    )

    run_dir = os.path.join(TMP_DIR, f"{experiment}_{model_key}_{seed}")

    args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=EPOCHS[train_kind],
        per_device_train_batch_size=COMMON["train_batch_size"],
        per_device_eval_batch_size=COMMON["eval_batch_size"],
        learning_rate=COMMON["learning_rate"],
        weight_decay=COMMON["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=seed
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=SentimentDataset(data["train"], tokenizer),
        eval_dataset=SentimentDataset(data["validation"], tokenizer),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )

    start = time.time()
    trainer.train()
    minutes = (time.time() - start) / 60

    history = [
        {"epoch": round(e["epoch"]), "val_macro_f1": round(e["eval_macro_f1"], 4)}
        for e in trainer.state.log_history if "eval_macro_f1" in e
    ]

    result = {
        "experiment": experiment,
        "model_key": model_key,
        "seed": seed,
        "trained_on": train_kind,
        "train_size": len(data["train"]),
        "validation_size": len(data["validation"]),
        "epochs": EPOCHS[train_kind],
        **COMMON,
        "training_minutes": round(minutes, 1),
        "history": history,
        "test": evaluate(trainer, tokenizer, test_sets)
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v["macro_f1"] for k, v in result["test"].items()}))

    del trainer, model
    shutil.rmtree(run_dir, ignore_errors=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 5. EXISTING SEED-42 SYNTHETIC MODELS ON THE VIDEO TEST SET
# ============================================================

def evaluate_existing_synthetic(video_test):
    out_dir = os.path.join(RESULTS_DIR, "existing")
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for model_key in BASE_MODELS:
        out_file = os.path.join(out_dir, f"{model_key}_synthetic_seed42_video_test.json")
        if os.path.exists(out_file):
            continue
        path = os.path.join(PROJECT_ROOT, "research", model_key, "best_model")
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()

        predictions = []
        texts = video_test["text"].tolist()
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                batch = tokenizer(texts[i:i + 32], truncation=True, max_length=128,
                                  padding=True, return_tensors="pt").to(device)
                predictions += model(**batch).logits.argmax(-1).tolist()

        result = {"model_key": model_key, "seed": 42, "trained_on": "synthetic",
                  "test": {"real_video": scores(video_test["label"].to_numpy(), np.array(predictions))}}
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(model_key, "synthetic seed42 on video test:", result["test"]["real_video"]["macro_f1"])

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================
# 6. MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=["video_split", "seeds_real", "seeds_synthetic", "existing", "all"],
        default="all"
    )
    args = parser.parse_args()
    todo = ["existing", "video_split", "seeds_real", "seeds_synthetic"] \
        if args.experiment == "all" else [args.experiment]

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    comment = real_comment_split()
    video = real_video_split()
    synthetic = synthetic_split()
    print("Video split sizes:", {k: len(v) for k, v in video.items()},
          "videos:", {k: v["video_id"].nunique() for k, v in video.items()})

    if "existing" in todo:
        evaluate_existing_synthetic(video["test"])

    if "video_split" in todo:
        for seed in SEEDS:
            for key in BASE_MODELS:
                run("video_split", key, seed, video, "real",
                    {"real_video": video["test"], "synthetic": synthetic["test"]})

    if "seeds_real" in todo:
        for seed in SEEDS[1:]:
            for key in BASE_MODELS:
                run("seeds_real", key, seed, comment, "real",
                    {"real_comment": comment["test"], "synthetic": synthetic["test"]})

    if "seeds_synthetic" in todo:
        for seed in SEEDS[1:]:
            for key in BASE_MODELS:
                run("seeds_synthetic", key, seed, synthetic, "synthetic",
                    {"synthetic": synthetic["test"], "real_comment": comment["test"],
                     "real_video": video["test"]})

    print("\nROBUSTNESS EXPERIMENTS COMPLETED")
