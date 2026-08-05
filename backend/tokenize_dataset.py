from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_FOLDER = PROJECT_ROOT / "dataset"
MODEL_FOLDER = PROJECT_ROOT / "model"

TRAIN_PATH = DATASET_FOLDER / "train.csv"
VALIDATION_PATH = DATASET_FOLDER / "validation.csv"
TEST_PATH = DATASET_FOLDER / "test.csv"

OUTPUT_FOLDER = DATASET_FOLDER / "tokenized_pytorch"

MODEL_NAME = "google/muril-base-cased"

MAX_LENGTH = 128


# ============================================================
# FIND LOCAL MURIL TOKENIZER
# ============================================================

def find_local_muril_folder():

    possible_folders = [
        MODEL_FOLDER / "muril",
        MODEL_FOLDER / "muril-base-cased",
        MODEL_FOLDER / "google-muril-base-cased",
        MODEL_FOLDER
    ]

    for folder in possible_folders:

        if not folder.exists():
            continue

        if (
            (folder / "tokenizer_config.json").exists()
            or (folder / "vocab.txt").exists()
            or (folder / "config.json").exists()
        ):
            return folder

    return None


local_model_path = find_local_muril_folder()


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading MuRIL tokenizer...")

if local_model_path:

    print("Loading tokenizer from local folder:")
    print(local_model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        local_model_path,
        local_files_only=True,
        use_fast=False
    )

else:

    print("Local MuRIL tokenizer not found.")
    print("Loading MuRIL from Hugging Face/cache...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=False
    )


# ============================================================
# LOAD CSV DATA
# ============================================================

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_PATH)
validation_df = pd.read_csv(VALIDATION_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Training rows:", len(train_df))
print("Validation rows:", len(validation_df))
print("Testing rows:", len(test_df))


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = {
    "sentence",
    "label"
}

for name, dataframe in {
    "train": train_df,
    "validation": validation_df,
    "test": test_df
}.items():

    if not required_columns.issubset(dataframe.columns):

        raise ValueError(
            f"{name}.csv must contain "
            f"'sentence' and 'label' columns.\n"
            f"Found: {dataframe.columns.tolist()}"
        )


# ============================================================
# TOKENIZE FUNCTION
# ============================================================

def tokenize_dataframe(dataframe):

    sentences = (
        dataframe["sentence"]
        .astype(str)
        .tolist()
    )

    labels = (
        dataframe["label"]
        .astype(int)
        .tolist()
    )

    print(
        f"Tokenizing {len(sentences)} sentences..."
    )

    tokenized = tokenizer(
        sentences,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

    dataset = {
        "input_ids": torch.tensor(
            tokenized["input_ids"],
            dtype=torch.long
        ),

        "attention_mask": torch.tensor(
            tokenized["attention_mask"],
            dtype=torch.long
        ),

        "labels": torch.tensor(
            labels,
            dtype=torch.long
        )
    }

    return dataset


# ============================================================
# TOKENIZE ALL DATASETS
# ============================================================

print("\n======================================")
print("TOKENIZATION STARTED")
print("======================================")

train_data = tokenize_dataframe(
    train_df
)

validation_data = tokenize_dataframe(
    validation_df
)

test_data = tokenize_dataframe(
    test_df
)


# ============================================================
# PRINT SHAPES
# ============================================================

print("\n======================================")
print("TOKENIZATION COMPLETED")
print("======================================")

print("\nTrain:")
print("input_ids:", train_data["input_ids"].shape)
print("attention_mask:", train_data["attention_mask"].shape)
print("labels:", train_data["labels"].shape)

print("\nValidation:")
print("input_ids:", validation_data["input_ids"].shape)
print("attention_mask:", validation_data["attention_mask"].shape)
print("labels:", validation_data["labels"].shape)

print("\nTest:")
print("input_ids:", test_data["input_ids"].shape)
print("attention_mask:", test_data["attention_mask"].shape)
print("labels:", test_data["labels"].shape)


# ============================================================
# SAVE TOKENIZED DATA
# ============================================================

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

torch.save(
    train_data,
    OUTPUT_FOLDER / "train.pt"
)

torch.save(
    validation_data,
    OUTPUT_FOLDER / "validation.pt"
)

torch.save(
    test_data,
    OUTPUT_FOLDER / "test.pt"
)


# ============================================================
# SAVE TOKENIZER
# ============================================================

tokenizer.save_pretrained(
    OUTPUT_FOLDER
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\nTokenized datasets saved successfully.")

print("\nLocation:")
print(OUTPUT_FOLDER)

print("\nFiles created:")

print(
    OUTPUT_FOLDER / "train.pt"
)

print(
    OUTPUT_FOLDER / "validation.pt"
)

print(
    OUTPUT_FOLDER / "test.pt"
)

print(
    OUTPUT_FOLDER / "tokenizer_config.json"
)

print("\nNo Hugging Face datasets/PyArrow dependency was used.")