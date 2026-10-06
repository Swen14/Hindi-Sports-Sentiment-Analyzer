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
# 1. MODEL INFORMATION
# ============================================================

MODEL_CHOICE = "indicbert_v2"

MODEL_NAME = "ai4bharat/IndicBERTv2-MLM-only"


# ============================================================
# 2. PATHS
# ============================================================

# Project root = NLP-Project
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATASET_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset"
)

TRAIN_PATH = os.path.join(
    DATASET_DIR,
    "final_train.csv"
)

VAL_PATH = os.path.join(
    DATASET_DIR,
    "final_validation.csv"
)

TEST_PATH = os.path.join(
    DATASET_DIR,
    "final_test.csv"
)

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "output"
)

BEST_MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "best_model"
)


# ============================================================
# 3. SHOW MODEL INFORMATION
# ============================================================

print("\n==============================================")
print("MODEL INFORMATION")
print("==============================================")

print("Selected Model  :", MODEL_CHOICE)
print("Base Checkpoint :", MODEL_NAME)

print("==============================================\n")


# ============================================================
# 4. LOAD FIXED DATASET SPLITS
# ============================================================

print("\n==============================================")
print("LOADING FIXED DATASET SPLITS")
print("==============================================")

print("\nTrain dataset :", TRAIN_PATH)
print("Validation dataset :", VAL_PATH)
print("Test dataset :", TEST_PATH)


train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)


print("\nDataset loaded successfully")

print("Train rows      :", len(train_df))
print("Validation rows :", len(val_df))
print("Test rows       :", len(test_df))


# ============================================================
# 5. CHECK DATASET
# ============================================================

print("\n==============================================")
print("DATASET INFORMATION")
print("==============================================")


print("\nTrain columns:")
print(train_df.columns.tolist())

print("\nTrain sentiment counts:")
print(train_df["sentiment"].value_counts())

print("\nTrain label counts:")
print(train_df["label"].value_counts())


print("\nValidation sentiment counts:")
print(val_df["sentiment"].value_counts())

print("\nValidation label counts:")
print(val_df["label"].value_counts())


print("\nTest sentiment counts:")
print(test_df["sentiment"].value_counts())

print("\nTest label counts:")
print(test_df["label"].value_counts())


# ============================================================
# 6. BASIC VALIDATION
# ============================================================

required_columns = ["text", "sentiment", "label"]

for column in required_columns:

    if column not in train_df.columns:
        raise ValueError(
            f"ERROR: Required column '{column}' "
            f"not found in training dataset."
        )

    if column not in val_df.columns:
        raise ValueError(
            f"ERROR: Required column '{column}' "
            f"not found in validation dataset."
        )

    if column not in test_df.columns:
        raise ValueError(
            f"ERROR: Required column '{column}' "
            f"not found in test dataset."
        )


# ============================================================
# 7. CLEAN DATA
# ============================================================

train_df["text"] = train_df["text"].astype(str).str.strip()
val_df["text"] = val_df["text"].astype(str).str.strip()
test_df["text"] = test_df["text"].astype(str).str.strip()


train_df = train_df.dropna(
    subset=["text", "label"]
).reset_index(drop=True)

val_df = val_df.dropna(
    subset=["text", "label"]
).reset_index(drop=True)

test_df = test_df.dropna(
    subset=["text", "label"]
).reset_index(drop=True)


print("\nRows after cleaning:")
print("Train      :", len(train_df))
print("Validation :", len(val_df))
print("Test       :", len(test_df))


# ============================================================
# 8. LOAD TOKENIZER
# ============================================================

print("\n==============================================")
print("LOADING FRESH INDICBERT V2 TOKENIZER")
print("==============================================")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# 9. PYTORCH DATASET
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


# ============================================================
# 10. CREATE DATASETS
# ============================================================

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
# 11. LOAD FRESH INDICBERT V2 MODEL
# ============================================================

print("\n==============================================")
print("LOADING FRESH INDICBERT V2 MODEL")
print("==============================================")

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
# 12. DATA COLLATOR
# ============================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# ============================================================
# 13. METRICS
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
# 14. TRAINING SETTINGS
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
# 15. TRAINER
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
# 16. DEVICE INFORMATION
# ============================================================

print("\n==============================================")
print("DEVICE INFORMATION")
print("==============================================")

if torch.cuda.is_available():

    print("GPU detected:")

    print(
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA memory:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3),
            2
        ),
        "GB"
    )

else:

    print("WARNING: GPU not detected.")

    print("Training will use CPU.")


# ============================================================
# 17. START TRAINING
# ============================================================

print("\n==============================================")
print("STARTING NEW INDICBERT V2 TRAINING")
print("==============================================")

print("\nTraining on:")
print("Train      :", len(train_df))
print("Validation :", len(val_df))
print("Test       :", len(test_df))

print("\nThis is a FRESH IndicBERT v2 model.")
print("It is NOT loading the old best_model.")


trainer.train()


# ============================================================
# 18. SAVE NEW BEST MODEL
# ============================================================

print("\n==============================================")
print("SAVING NEW BEST INDICBERT V2 MODEL")
print("==============================================")

trainer.save_model(
    BEST_MODEL_DIR
)

tokenizer.save_pretrained(
    BEST_MODEL_DIR
)

print("\nNew best model saved to:")

print(BEST_MODEL_DIR)


# ============================================================
# 19. FINAL TEST SET EVALUATION
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
# 20. FINAL METRICS
# ============================================================

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
# 21. CLASSIFICATION REPORT
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
# 22. CONFUSION MATRIX
# ============================================================

print("\nCONFUSION MATRIX")
print("--------------------------------")

cm = confusion_matrix(

    true_labels,

    predictions
)

print(cm)


# ============================================================
# 23. SAVE RESULTS
# ============================================================

results_dir = os.path.join(
    PROJECT_ROOT,
    "research",
    "results"
)

os.makedirs(
    results_dir,
    exist_ok=True
)


results_path = os.path.join(
    results_dir,
    "indicbert_v2_results.txt"
)


with open(

    results_path,

    "w",

    encoding="utf-8"

) as f:

    f.write(
        "IndicBERT v2 Research Results\n"
    )

    f.write(
        "=============================\n\n"
    )

    f.write(
        "Dataset: NLP_Project_Final_Dataset.csv\n"
    )

    f.write(
        f"Train Size: {len(train_df)}\n"
    )

    f.write(
        f"Validation Size: {len(val_df)}\n"
    )

    f.write(
        f"Test Size: {len(test_df)}\n\n"
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


# ============================================================
# 24. COMPLETED
# ============================================================

print("\n==============================================")
print("NEW INDICBERT V2 EXPERIMENT COMPLETED")
print("==============================================")