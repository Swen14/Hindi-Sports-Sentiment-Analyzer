import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_FOLDER = PROJECT_ROOT / "dataset"

SENTIHIN_PATH = DATASET_FOLDER / "hindi_sentiment_analysis.csv"

sports_files = list(DATASET_FOLDER.glob("sports_dataset*.csv"))

if not sports_files:
    raise FileNotFoundError(
        f"No sports dataset CSV found inside: {DATASET_FOLDER}"
    )

SPORTS_PATH = sports_files[0]

COMBINED_PATH = DATASET_FOLDER / "combined_dataset.csv"
TRAIN_PATH = DATASET_FOLDER / "train.csv"
VALIDATION_PATH = DATASET_FOLDER / "validation.csv"
TEST_PATH = DATASET_FOLDER / "test.csv"


def clean_hindi_text(text):
    text = str(text)

    text = unicodedata.normalize("NFC", text)

    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"@\w+", " ", text)

    text = text.replace("#", "")

    text = re.sub(
        r"[^\u0900-\u097FA-Za-z0-9\s।!?]",
        " ",
        text
    )

    text = re.sub(r"\s+", " ", text).strip()

    return text


def standardize_label(label):
    label = str(label).strip().lower()

    label_mapping = {
        "positive": "positive",
        "pos": "positive",
        "सकारात्मक": "positive",
        "पॉजिटिव": "positive",
        "2": "positive",

        "negative": "negative",
        "neg": "negative",
        "नकारात्मक": "negative",
        "नेगेटिव": "negative",
        "0": "negative",

        "neutral": "neutral",
        "neu": "neutral",
        "तटस्थ": "neutral",
        "न्यूट्रल": "neutral",
        "1": "neutral"
    }

    return label_mapping.get(label, label)


def preprocess_dataset(df, source_name):
    required_columns = {"sentence", "sentiment"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"{source_name} dataset must contain "
            f"'sentence' and 'sentiment' columns. "
            f"Found columns: {df.columns.tolist()}"
        )

    df = df[["sentence", "sentiment"]].copy()

    print(f"\n--- {source_name}: Before preprocessing ---")
    print("Rows:", len(df))

    print("\nMissing values:")
    print(df.isnull().sum())

    print(
        "\nDuplicate sentences:",
        df.duplicated(subset=["sentence"]).sum()
    )

    df = df.dropna(subset=["sentence", "sentiment"])

    df["sentence"] = df["sentence"].apply(clean_hindi_text)

    df["sentiment"] = df["sentiment"].apply(
        standardize_label
    )

    df = df[df["sentence"].str.len() > 0]

    valid_labels = {
        "positive",
        "negative",
        "neutral"
    }

    invalid_labels = df[
        ~df["sentiment"].isin(valid_labels)
    ]["sentiment"].unique()

    if len(invalid_labels) > 0:
        print(
            f"\nInvalid labels removed from {source_name}:",
            invalid_labels
        )

    df = df[df["sentiment"].isin(valid_labels)]

    df = df.drop_duplicates(
        subset=["sentence"],
        keep="first"
    )

    df["source"] = source_name

    print(f"\n--- {source_name}: After preprocessing ---")
    print("Rows:", len(df))

    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts())

    return df.reset_index(drop=True)


print("Loading datasets...")

print("SentiHin dataset found:", SENTIHIN_PATH.name)
print("Sports dataset found:", SPORTS_PATH.name)

sentihin_df = pd.read_csv(SENTIHIN_PATH)
sports_df = pd.read_csv(SPORTS_PATH)

print("\nSentiHin shape:", sentihin_df.shape)
print("Sports dataset shape:", sports_df.shape)

sentihin_processed = preprocess_dataset(
    sentihin_df,
    "sentihin"
)

sports_processed = preprocess_dataset(
    sports_df,
    "sports"
)

combined_df = pd.concat(
    [sentihin_processed, sports_processed],
    ignore_index=True
)

print(
    "\nRows before cross-dataset duplicate removal:",
    len(combined_df)
)

combined_df = combined_df.drop_duplicates(
    subset=["sentence"],
    keep="last"
)

combined_df = combined_df.reset_index(drop=True)

label_mapping = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

combined_df["label"] = combined_df["sentiment"].map(
    label_mapping
)

if combined_df["label"].isnull().any():
    unmapped_labels = combined_df.loc[
        combined_df["label"].isnull(),
        "sentiment"
    ].unique()

    raise ValueError(
        f"Some labels could not be encoded: {unmapped_labels}"
    )

combined_df["label"] = combined_df["label"].astype(int)

print("\n--- Final combined dataset ---")
print("Total rows:", len(combined_df))

print("\nSentiment distribution:")
print(combined_df["sentiment"].value_counts())

print("\nSource distribution:")
print(combined_df["source"].value_counts())

print("\nLabel mapping:")
print(label_mapping)

combined_df.to_csv(
    COMBINED_PATH,
    index=False,
    encoding="utf-8-sig"
)

train_df, temp_df = train_test_split(
    combined_df,
    test_size=0.20,
    random_state=42,
    stratify=combined_df["label"]
)

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)

train_df.to_csv(
    TRAIN_PATH,
    index=False,
    encoding="utf-8-sig"
)

validation_df.to_csv(
    VALIDATION_PATH,
    index=False,
    encoding="utf-8-sig"
)

test_df.to_csv(
    TEST_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("\n--- Dataset split completed ---")
print("Training rows:", len(train_df))
print("Validation rows:", len(validation_df))
print("Testing rows:", len(test_df))

print("\nFiles created:")
print(COMBINED_PATH)
print(TRAIN_PATH)
print(VALIDATION_PATH)
print(TEST_PATH)