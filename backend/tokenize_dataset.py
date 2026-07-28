from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "dataset" / "train.csv"
VALIDATION_PATH = PROJECT_ROOT / "dataset" / "validation.csv"
TEST_PATH = PROJECT_ROOT / "dataset" / "test.csv"

MODEL_FOLDER = PROJECT_ROOT / "model"

OUTPUT_FOLDER = PROJECT_ROOT / "dataset" / "tokenized_dataset"


def find_local_muril_folder():
    possible_folders = [
        MODEL_FOLDER / "muril",
        MODEL_FOLDER / "muril-base-cased",
        MODEL_FOLDER / "google-muril-base-cased",
        MODEL_FOLDER
    ]

    for folder in possible_folders:
        tokenizer_config = folder / "tokenizer_config.json"
        vocab_file = folder / "vocab.txt"
        config_file = folder / "config.json"

        if folder.exists() and (
            tokenizer_config.exists()
            or vocab_file.exists()
            or config_file.exists()
        ):
            return folder

    return None


local_model_path = find_local_muril_folder()

if local_model_path:
    print("Loading MuRIL tokenizer from local folder:")
    print(local_model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        local_model_path,
        local_files_only=True
    )
else:
    print("Local MuRIL folder not found.")
    print("Loading MuRIL tokenizer from Hugging Face cache or internet.")

    tokenizer = AutoTokenizer.from_pretrained(
        "google/muril-base-cased"
    )


print("\nLoading processed datasets...")

train_df = pd.read_csv(TRAIN_PATH)
validation_df = pd.read_csv(VALIDATION_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Train rows:", len(train_df))
print("Validation rows:", len(validation_df))
print("Test rows:", len(test_df))


required_columns = {"sentence", "label"}

for name, dataframe in {
    "train": train_df,
    "validation": validation_df,
    "test": test_df
}.items():
    if not required_columns.issubset(dataframe.columns):
        raise ValueError(
            f"{name}.csv must contain 'sentence' and 'label' columns. "
            f"Found: {dataframe.columns.tolist()}"
        )


train_dataset = Dataset.from_pandas(
    train_df[["sentence", "label"]],
    preserve_index=False
)

validation_dataset = Dataset.from_pandas(
    validation_df[["sentence", "label"]],
    preserve_index=False
)

test_dataset = Dataset.from_pandas(
    test_df[["sentence", "label"]],
    preserve_index=False
)


dataset_dict = DatasetDict({
    "train": train_dataset,
    "validation": validation_dataset,
    "test": test_dataset
})


def tokenize_batch(batch):
    return tokenizer(
        batch["sentence"],
        truncation=True,
        padding="max_length",
        max_length=128
    )


print("\nTokenizing datasets...")

tokenized_dataset = dataset_dict.map(
    tokenize_batch,
    batched=True
)

tokenized_dataset = tokenized_dataset.remove_columns(
    ["sentence"]
)

tokenized_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

print("\nTokenization completed.")

print("\nTrain dataset:")
print(tokenized_dataset["train"])

print("\nValidation dataset:")
print(tokenized_dataset["validation"])

print("\nTest dataset:")
print(tokenized_dataset["test"])


OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

tokenized_dataset.save_to_disk(
    str(OUTPUT_FOLDER)
)

print("\nTokenized dataset saved to:")
print(OUTPUT_FOLDER)