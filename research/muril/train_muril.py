import os
import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. PATHS
# ============================================================

DATA_PATH = "../../dataset/Research_12000.csv"

MODEL_NAME = "google/muril-base-cased"

OUTPUT_DIR = "./output"
BEST_MODEL_DIR = "./best_model"


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully")
print("Total rows:", len(df))

print("\nSplit counts:")
print(df["split"].value_counts())

print("\nSentiment counts:")
print(df["sentiment"].value_counts())


train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()
test_df = df[df["split"] == "test"].copy()


print("\nTrain:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))


# ============================================================
# 3. LOAD TOKENIZER
# ============================================================

print("\nLoading fresh MuRIL tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# 4. PYTORCH DATASET
# ============================================================

class SentimentDataset(Dataset):

    def __init__(self, dataframe, tokenizer):

        self.texts = dataframe["text"].tolist()
        self.labels = dataframe["label"].tolist()

        self.tokenizer = tokenizer


    def __len__(self):

        return len(self.texts)


    def __getitem__(self, index):

        text = str(self.texts[index])

        label = int(self.labels[index])

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=128
        )

        encoding["labels"] = label

        return encoding


train_dataset = SentimentDataset(
    train_df,
    tokenizer
)

val_dataset = SentimentDataset(
    val_df,
    tokenizer
)

test_dataset = SentimentDataset(
    test_df,
    tokenizer
)


# ============================================================
# 5. LOAD FRESH MuRIL MODEL
# ============================================================

print("\nLoading fresh MuRIL model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,

    num_labels=3,

    id2label={
        0: "negative",
        1: "neutral",
        2: "positive"
    },

    label2id={
        "negative": 0,
        "neutral": 1,
        "positive": 2
    }
)


# ============================================================
# 6. DATA COLLATOR
# ============================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# ============================================================
# 7. METRICS
# ============================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1
    }


# ============================================================
# 8. TRAINING SETTINGS
# ============================================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    num_train_epochs=3,

    per_device_train_batch_size=8,

    per_device_eval_batch_size=16,

    learning_rate=2e-5,

    weight_decay=0.01,

    eval_strategy="epoch",

    save_strategy="epoch",

    load_best_model_at_end=True,

    metric_for_best_model="macro_f1",

    greater_is_better=True,

    logging_steps=50,

    save_total_limit=2,

    fp16=torch.cuda.is_available(),

    report_to="none",

    seed=42
)


# ============================================================
# 9. TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    data_collator=data_collator,

    compute_metrics=compute_metrics
)


# ============================================================
# 10. DEVICE CHECK
# ============================================================

print("\n==============================================")
print("DEVICE INFORMATION")
print("==============================================")

if torch.cuda.is_available():

    print("GPU detected:")
    print(torch.cuda.get_device_name(0))

else:

    print("WARNING: GPU not detected")
    print("Training will use CPU")


# ============================================================
# 11. TRAIN
# ============================================================

print("\n==============================================")
print("STARTING MuRIL TRAINING")
print("==============================================")

trainer.train()


# ============================================================
# 12. SAVE BEST MODEL
# ============================================================

print("\nSaving best model...")

trainer.save_model(
    BEST_MODEL_DIR
)

tokenizer.save_pretrained(
    BEST_MODEL_DIR
)


# ============================================================
# 13. FINAL TEST
# ============================================================

print("\n==============================================")
print("FINAL TEST SET EVALUATION")
print("==============================================")

prediction_output = trainer.predict(
    test_dataset
)

predictions = np.argmax(
    prediction_output.predictions,
    axis=-1
)

true_labels = prediction_output.label_ids


accuracy = accuracy_score(
    true_labels,
    predictions
)


precision, recall, f1, _ = precision_recall_fscore_support(

    true_labels,

    predictions,

    average="macro",

    zero_division=0
)


print("\nFINAL RESULTS")
print("--------------------------------")

print(f"Accuracy        : {accuracy:.4f}")

print(
    f"Macro Precision : {precision:.4f}"
)

print(
    f"Macro Recall    : {recall:.4f}"
)

print(
    f"Macro F1        : {f1:.4f}"
)


# ============================================================
# 14. CLASSIFICATION REPORT
# ============================================================

print("\nCLASSIFICATION REPORT")
print("--------------------------------")

report = classification_report(

    true_labels,

    predictions,

    target_names=[
        "Negative",
        "Neutral",
        "Positive"
    ],

    digits=4
)

print(report)


# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

print("\nCONFUSION MATRIX")
print("--------------------------------")

cm = confusion_matrix(
    true_labels,
    predictions
)

print(cm)


# ============================================================
# 16. SAVE RESULTS
# ============================================================

results_dir = "../results"

os.makedirs(
    results_dir,
    exist_ok=True
)


results_path = os.path.join(
    results_dir,
    "muril_results.txt"
)


with open(
    results_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "MuRIL Research Results\n"
    )

    f.write(
        "======================\n\n"
    )

    f.write(
        f"Accuracy: {accuracy:.4f}\n"
    )

    f.write(
        f"Macro Precision: {precision:.4f}\n"
    )

    f.write(
        f"Macro Recall: {recall:.4f}\n"
    )

    f.write(
        f"Macro F1: {f1:.4f}\n\n"
    )

    f.write(
        "Classification Report\n"
    )

    f.write(
        "---------------------\n"
    )

    f.write(report)

    f.write(
        "\nConfusion Matrix\n"
    )

    f.write(
        "----------------\n"
    )

    f.write(
        str(cm)
    )


print("\nResults saved to:")
print(results_path)

print("\nMuRIL experiment completed.")