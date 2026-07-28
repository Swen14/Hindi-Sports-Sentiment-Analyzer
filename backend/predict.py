from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model" / "muril_sentiment_model"

ID_TO_LABEL = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=False
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)
model.eval()


def predict_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )

    predicted_id = torch.argmax(
        probabilities,
        dim=-1
    ).item()

    confidence = probabilities[0][predicted_id].item()

    return {
        "text": text,
        "sentiment": ID_TO_LABEL[predicted_id],
        "confidence": round(confidence * 100, 2)
    }


test_sentences = [
    "भारत ने शानदार जीत हासिल की।",
    "टीम का प्रदर्शन बेहद खराब रहा।",
    "मैच शाम सात बजे शुरू होगा।",
    "खिलाड़ी ने बेहतरीन प्रदर्शन किया।",
    "टीम लगातार मैच हार रही है।"
]


print("Model loaded successfully.")
print("Device:", device)
print()

for sentence in test_sentences:
    result = predict_sentiment(sentence)

    print("Text:", result["text"])
    print("Sentiment:", result["sentiment"])
    print("Confidence:", result["confidence"], "%")
    print("-" * 50)