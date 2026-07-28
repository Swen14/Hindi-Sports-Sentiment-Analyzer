from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TOKENIZED_DATASET_PATH = (
    PROJECT_ROOT / "dataset" / "tokenized_dataset"
)

OUTPUT_MODEL_PATH = (
    PROJECT_ROOT / "model" / "muril_sentiment_model"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT / "model" / "checkpoints"
)

MODEL_NAME = "google/muril-base-cased"

NUM_LABELS = 3

ID_TO_LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

LABEL_TO_ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}


print("Checking hardware...")

if torch.cuda.is_available():
    print("GPU available:", torch.cuda.get_device_name(0))
else:
    print("GPU not available. Training will use CPU and may be very slow.")


print("\nLoading tokenized dataset...")

dataset = load_from_disk(
    str(TOKENIZED_DATASET_PATH)
)

print(dataset)


print("\nLoading MuRIL tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False
)


print("\nLoading MuRIL classification model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=ID_TO_LABEL,
    label2id=LABEL_TO_ID
)


data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


def compute_metrics(evaluation_prediction):
    logits, labels = evaluation_prediction

    predictions = np.argmax(
        logits,
        axis=-1
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="weighted",
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


CHECKPOINT_PATH.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_MODEL_PATH.mkdir(
    parents=True,
    exist_ok=True
)


training_args = TrainingArguments(
    output_dir=str(CHECKPOINT_PATH),

    learning_rate=2e-5,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    num_train_epochs=3,

    weight_decay=0.01,

    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=25,

    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,

    save_total_limit=2,

    report_to="none",

    fp16=torch.cuda.is_available(),

    seed=42
)


trainer = Trainer(
    model=model,
    args=training_args,

    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],

    processing_class=tokenizer,
    data_collator=data_collator,

    compute_metrics=compute_metrics
)


print("\nStarting MuRIL fine-tuning...")

trainer.train()


print("\nEvaluating on validation dataset...")

validation_results = trainer.evaluate(
    dataset["validation"]
)

print("\nValidation results:")

for metric, value in validation_results.items():
    print(f"{metric}: {value}")


print("\nEvaluating on test dataset...")

test_results = trainer.evaluate(
    dataset["test"]
)

print("\nTest results:")

for metric, value in test_results.items():
    print(f"{metric}: {value}")


print("\nSaving trained model...")

trainer.save_model(
    str(OUTPUT_MODEL_PATH)
)

tokenizer.save_pretrained(
    str(OUTPUT_MODEL_PATH)
)


print("\nTraining completed.")
print("Model saved to:")
print(OUTPUT_MODEL_PATH)