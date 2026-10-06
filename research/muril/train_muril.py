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

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. PATHS
# ============================================================

# Project root = NLP-Project
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "NLP_Project_Final_Dataset.csv"
)

MODEL_NAME = "google/muril-base-cased"

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "output"
)

BEST_MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "best_model"
)


# ============================================================
# 2. LOAD NEW DATASET
# ============================================================

print("\n==============================================")
print("LOADING NEW HINDI SPORTS DATASET")
print("==============================================")

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully")
print("Dataset path:", DATA_PATH)
print("Total rows:", len(df))

print("\nColumns:")
print(df.columns.tolist())

print("\nSentiment counts:")
print(df["sentiment"].value_counts())

print("\nSport counts:")
print(df["sport"].value_counts())


# ============================================================
# 3. LABEL MAPPING
# ============================================================

label_map = {
    "नकारात्मक": 0,
    "तटस्थ": 1,
    "सकारात्मक": 2
}

df["label"] = df["sentiment"].map(label_map)


# Check for invalid labels
if df["label"].isna().any():
    print("\nERROR: Unknown sentiment labels found:")
    print(df[df["label"].isna()]["sentiment"].unique())
    raise ValueError("Dataset contains sentiment labels not present in label_map.")


# ============================================================
# 4. CLEAN DATA
# ============================================================

df["text"] = df["text"].astype(str).str.strip()

df = df.dropna(subset=["text", "label"])

print("\nRows after cleaning:", len(df))


# ============================================================
# 5. REMOVE EXACT DUPLICATES
# ============================================================

before_duplicates = len(df)

df = df.drop_duplicates(
    subset=["text"]
).reset_index(drop=True)

removed_duplicates = before_duplicates - len(df)

print("Duplicate rows removed:", removed_duplicates)
print("Final dataset size:", len(df))


# ============================================================
# 6. CREATE FIXED 80/10/10 SPLIT
# ============================================================

print("\n==============================================")
print("CREATING FIXED TRAIN / VALIDATION / TEST SPLIT")
print("==============================================")


# First:
# 80% train
# 20% temporary

train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["label"],
    random_state=42
)


# Then:
# 10% validation
# 10% test

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)


# Reset indexes
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


print("\nSplit sizes:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))


print("\nTrain sentiment distribution:")
print(train_df["sentiment"].value_counts())

print("\nValidation sentiment distribution:")
print(val_df["sentiment"].value_counts())

print("\nTest sentiment distribution:")
print(test_df["sentiment"].value_counts())


# ============================================================
# 7. SAVE FIXED SPLITS
# ============================================================

dataset_dir = os.path.join(
    PROJECT_ROOT,
    "dataset"
)

train_split_path = os.path.join(
    dataset_dir,
    "final_train.csv"
)

val_split_path = os.path.join(
    dataset_dir,
    "final_validation.csv"
)

test_split_path = os.path.join(
    dataset_dir,
    "final_test.csv"
)


train_df.to_csv(
    train_split_path,
    index=False,
    encoding="utf-8-sig"
)

val_df.to_csv(
    val_split_path,
    index=False,
    encoding="utf-8-sig"
)

test_df.to_csv(
    test_split_path,
    index=False,
    encoding="utf-8-sig"
)


print("\nFixed datasets saved:")
print(train_split_path)
print(val_split_path)
print(test_split_path)


# ============================================================
# 8. LOAD TOKENIZER
# ============================================================

print("\n==============================================")
print("LOADING MuRIL TOKENIZER")
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
# 10. LOAD FRESH MuRIL MODEL
# ============================================================

print("\n==============================================")
print("LOADING FRESH MuRIL MODEL")
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
# 11. DATA COLLATOR
# ============================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# ============================================================
# 12. METRICS
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
# 13. TRAINING SETTINGS
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
# 14. TRAINER
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
# 15. DEVICE CHECK
# ============================================================

print("\n==============================================")
print("DEVICE INFORMATION")
print("==============================================")

if torch.cuda.is_available():

    print("GPU detected:")
    print(torch.cuda.get_device_name(0))

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

    print("WARNING: GPU not detected")
    print("Training will use CPU")


# ============================================================
# 16. TRAIN
# ============================================================

print("\n==============================================")
print("STARTING MuRIL TRAINING")
print("==============================================")

trainer.train()


# ============================================================
# 17. SAVE BEST MODEL
# ============================================================

print("\n==============================================")
print("SAVING BEST MuRIL MODEL")
print("==============================================")

trainer.save_model(
    BEST_MODEL_DIR
)

tokenizer.save_pretrained(
    BEST_MODEL_DIR
)

print("\nBest model saved to:")
print(BEST_MODEL_DIR)


# ============================================================
# 18. FINAL TEST SET EVALUATION
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
# 19. CLASSIFICATION REPORT
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
# 20. CONFUSION MATRIX
# ============================================================

print("\nCONFUSION MATRIX")
print("--------------------------------")

cm = confusion_matrix(
    true_labels,
    predictions
)

print(cm)


# ============================================================
# 21. SAVE RESULTS
# ============================================================

results_dir = os.path.join(
    os.path.dirname(__file__),
    "..",
    "results"
)

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
        "Dataset: NLP_Project_Final_Dataset.csv\n"
    )

    f.write(
        "Total Dataset Size: 10000\n"
    )

    f.write(
        "Train Size: 8000\n"
    )

    f.write(
        "Validation Size: 1000\n"
    )

    f.write(
        "Test Size: 1000\n\n"
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

print("\n==============================================")
print("MuRIL EXPERIMENT COMPLETED")
print("==============================================")