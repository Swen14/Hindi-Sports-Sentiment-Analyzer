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
# 1. PROJECT PATH
# ============================================================

PROJECT_ROOT = r"C:\Users\Swen\Desktop\NLP-Project"


# ============================================================
# 2. MODEL INFORMATION
# ============================================================

MODEL_CHOICE = "xlm_roberta"

MODEL_NAME = "FacebookAI/xlm-roberta-base"


# ============================================================
# 3. PATHS
# ============================================================

TRAIN_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "final_train.csv"
)

VAL_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "final_validation.csv"
)

TEST_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "final_test.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "research",
    "xlm_roberta",
    "output"
)

BEST_MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "research",
    "xlm_roberta",
    "best_model"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "research",
    "results"
)

RESULTS_PATH = os.path.join(
    RESULTS_DIR,
    "xlm_roberta_results.txt"
)


# ============================================================
# 4. CREATE DIRECTORIES
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 5. MODEL INFORMATION
# ============================================================

print("\n==============================================")
print("MODEL INFORMATION")
print("==============================================")

print("Selected Model  :", MODEL_CHOICE)
print("Base Checkpoint :", MODEL_NAME)

print("==============================================\n")


# ============================================================
# 6. LOAD FIXED DATASET SPLITS
# ============================================================

print("Loading fixed dataset splits...")

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)


print("\nDataset loaded successfully")

print("Train      :", len(train_df))
print("Validation :", len(val_df))
print("Test       :", len(test_df))


# ============================================================
# 7. VALIDATE DATASET
# ============================================================

required_columns = [
    "text",
    "sentiment",
    "label"
]

for dataframe_name, dataframe in [
    ("Train", train_df),
    ("Validation", val_df),
    ("Test", test_df)
]:

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:

        raise ValueError(
            f"{dataframe_name} dataset is missing columns: "
            f"{missing_columns}"
        )


print("\nRequired columns verified.")


# ============================================================
# 8. SHOW SENTIMENT DISTRIBUTION
# ============================================================

print("\nSentiment distribution:")

print("\nTrain:")
print(train_df["sentiment"].value_counts())

print("\nValidation:")
print(val_df["sentiment"].value_counts())

print("\nTest:")
print(test_df["sentiment"].value_counts())


# ============================================================
# 9. VERIFY LABELS
# ============================================================

expected_labels = {0, 1, 2}

for dataframe_name, dataframe in [
    ("Train", train_df),
    ("Validation", val_df),
    ("Test", test_df)
]:

    actual_labels = set(
        dataframe["label"].unique()
    )

    if not actual_labels.issubset(expected_labels):

        raise ValueError(
            f"Unexpected labels found in {dataframe_name}: "
            f"{actual_labels}"
        )


print("\nLabels verified: 0, 1, 2")


# ============================================================
# 10. LOAD TOKENIZER
# ============================================================

print("\nLoading fresh XLM-RoBERTa tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# 11. PYTORCH DATASET
# ============================================================

class SentimentDataset(Dataset):

    def __init__(
        self,
        dataframe,
        tokenizer
    ):

        self.texts = dataframe["text"].tolist()

        self.labels = dataframe["label"].tolist()

        self.tokenizer = tokenizer


    def __len__(self):

        return len(self.texts)


    def __getitem__(self, index):

        text = str(
            self.texts[index]
        )

        label = int(
            self.labels[index]
        )

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
# 12. LOAD FRESH XLM-ROBERTA MODEL
# ============================================================

print("\nLoading fresh XLM-RoBERTa model...")

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
# 13. DATA COLLATOR
# ============================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# ============================================================
# 14. METRICS
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

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="macro",
            zero_division=0
        )
    )

    return {

        "accuracy": accuracy,

        "macro_precision": precision,

        "macro_recall": recall,

        "macro_f1": f1
    }


# ============================================================
# 15. TRAINING SETTINGS
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
# 16. TRAINER
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
# 17. DEVICE INFORMATION
# ============================================================

print("\n==============================================")
print("DEVICE INFORMATION")
print("==============================================")

if torch.cuda.is_available():

    print("GPU detected:")
    print(torch.cuda.get_device_name(0))

else:

    print("WARNING: GPU not detected.")
    print("Training will use CPU.")


# ============================================================
# 18. START TRAINING
# ============================================================

print("\n==============================================")
print("STARTING XLM-ROBERTA TRAINING")
print("==============================================")

trainer.train()


# ============================================================
# 19. SAVE BEST MODEL
# ============================================================

print("\nSaving best XLM-RoBERTa model...")

trainer.save_model(
    BEST_MODEL_DIR
)

tokenizer.save_pretrained(
    BEST_MODEL_DIR
)


# ============================================================
# 20. FINAL TEST SET
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


# ============================================================
# 21. FINAL METRICS
# ============================================================

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


print("\nFINAL RESULTS")
print("--------------------------------")

print(
    f"Accuracy        : {accuracy:.4f}"
)

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
# 22. CLASSIFICATION REPORT
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
# 23. CONFUSION MATRIX
# ============================================================

print("\nCONFUSION MATRIX")
print("--------------------------------")

cm = confusion_matrix(
    true_labels,
    predictions
)

print(cm)


# ============================================================
# 24. SAVE RESULTS
# ============================================================

with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "XLM-RoBERTa Research Results\n"
    )

    f.write(
        "============================\n\n"
    )

    f.write(
        f"Model: {MODEL_NAME}\n"
    )

    f.write(
        f"Train Samples: {len(train_df)}\n"
    )

    f.write(
        f"Validation Samples: {len(val_df)}\n"
    )

    f.write(
        f"Test Samples: {len(test_df)}\n\n"
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


# ============================================================
# 25. COMPLETION
# ============================================================

print("\nResults saved to:")
print(RESULTS_PATH)

print("\nBest model saved to:")
print(BEST_MODEL_DIR)

print(
    "\nXLM-RoBERTa experiment completed."
)