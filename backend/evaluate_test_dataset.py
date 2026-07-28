from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model" / "muril_sentiment_model"
TEST_PATH = PROJECT_ROOT / "dataset" / "test.csv"
OUTPUT_PATH = PROJECT_ROOT / "dataset" / "test_predictions.csv"

ID_TO_LABEL = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

class SentimentDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=64):
        self.texts = dataframe["sentence"].astype(str).tolist()
        self.labels = dataframe["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "text": self.texts[idx]
        }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

test_df = pd.read_csv(TEST_PATH)

print("Test rows:", len(test_df))
print("Columns:", test_df.columns.tolist())

if "sentence" not in test_df.columns or "label" not in test_df.columns:
    raise ValueError("test.csv must contain 'sentence' and 'label' columns.")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=False,
    local_files_only=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model.to(device)
model.eval()

dataset = SentimentDataset(test_df, tokenizer)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False
)

true_labels = []
predicted_labels = []
confidences = []
texts = []

with torch.inference_mode():
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        probabilities = torch.softmax(outputs.logits, dim=-1)

        predictions = probabilities.argmax(dim=-1)

        confidence = probabilities.max(dim=-1).values

        true_labels.extend(labels.cpu().tolist())
        predicted_labels.extend(predictions.cpu().tolist())
        confidences.extend(confidence.cpu().tolist())
        texts.extend(batch["text"])

accuracy = accuracy_score(true_labels, predicted_labels)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:\n")
print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=["Negative", "Neutral", "Positive"],
        zero_division=0
    )
)

print("\nConfusion Matrix:\n")
print(
    confusion_matrix(
        true_labels,
        predicted_labels
    )
)

results = pd.DataFrame({
    "sentence": texts,
    "actual": [ID_TO_LABEL[x] for x in true_labels],
    "predicted": [ID_TO_LABEL[x] for x in predicted_labels],
    "confidence": [round(x * 100, 2) for x in confidences],
    "correct": [a == b for a, b in zip(true_labels, predicted_labels)]
})

results.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("\nResults saved to:")
print(OUTPUT_PATH)