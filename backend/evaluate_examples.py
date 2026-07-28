from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model" / "muril_sentiment_model"

ID_TO_LABEL = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

LABEL_TO_ID = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2
}


test_examples = [
    {
        "text": "भारत ने शानदार प्रदर्शन करते हुए मैच जीत लिया।",
        "expected": "Positive"
    },
    {
        "text": "खिलाड़ी ने बेहतरीन पारी खेली।",
        "expected": "Positive"
    },
    {
        "text": "टीम की जीत से प्रशंसक बहुत खुश हैं।",
        "expected": "Positive"
    },
    {
        "text": "भारत ने फाइनल में शानदार वापसी की।",
        "expected": "Positive"
    },
    {
        "text": "गेंदबाज ने शानदार प्रदर्शन किया।",
        "expected": "Positive"
    },

    {
        "text": "टीम की खराब बल्लेबाजी के कारण हार हुई।",
        "expected": "Negative"
    },
    {
        "text": "खिलाड़ी का प्रदर्शन बेहद निराशाजनक रहा।",
        "expected": "Negative"
    },
    {
        "text": "भारत लगातार तीसरा मैच हार गया।",
        "expected": "Negative"
    },
    {
        "text": "टीम ने आसान मौके गंवा दिए।",
        "expected": "Negative"
    },
    {
        "text": "चोट के कारण खिलाड़ी टूर्नामेंट से बाहर हो गया।",
        "expected": "Negative"
    },

    {
        "text": "मैच रविवार शाम सात बजे शुरू होगा।",
        "expected": "Neutral"
    },
    {
        "text": "भारत और ऑस्ट्रेलिया के बीच मैच मुंबई में खेला जाएगा।",
        "expected": "Neutral"
    },
    {
        "text": "टीम की घोषणा कल की जाएगी।",
        "expected": "Neutral"
    },
    {
        "text": "टूर्नामेंट में कुल आठ टीमें भाग लेंगी।",
        "expected": "Neutral"
    },
    {
        "text": "मैच का प्रसारण शाम छह बजे से होगा।",
        "expected": "Neutral"
    }
]


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

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


def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=False,
        max_length=64
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )

    predicted_id = probabilities.argmax(
        dim=-1
    ).item()

    confidence = probabilities[
        0,
        predicted_id
    ].item()

    return (
        ID_TO_LABEL[predicted_id],
        round(confidence * 100, 2)
    )


rows = []
actual_labels = []
predicted_labels = []

for example in test_examples:
    prediction, confidence = predict(example["text"])

    actual_labels.append(
        LABEL_TO_ID[example["expected"]]
    )

    predicted_labels.append(
        LABEL_TO_ID[prediction]
    )

    rows.append({
        "text": example["text"],
        "expected": example["expected"],
        "predicted": prediction,
        "confidence": confidence,
        "correct": example["expected"] == prediction
    })


results_df = pd.DataFrame(rows)

print("\nDetailed results:\n")
print(results_df.to_string(index=False))

accuracy = accuracy_score(
    actual_labels,
    predicted_labels
)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification report:\n")
print(
    classification_report(
        actual_labels,
        predicted_labels,
        target_names=[
            "Negative",
            "Neutral",
            "Positive"
        ],
        zero_division=0
    )
)

print("\nConfusion matrix:\n")
print(
    confusion_matrix(
        actual_labels,
        predicted_labels
    )
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "manual_evaluation_results.csv"
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("\nResults saved to:")
print(OUTPUT_PATH)