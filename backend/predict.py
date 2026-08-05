from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "muril_sentiment_model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.to(device)
model.eval()

print("Model loaded successfully.")
print("Device:", device)

label_map = {
    int(key): value
    for key, value in model.config.id2label.items()
}

while True:
    text = input("\nEnter text for sentiment analysis (or type exit): ").strip()

    if text.lower() == "exit":
        print("Program stopped.")
        break

    if not text:
        print("Please enter some text.")
        continue

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)

        predicted_id = int(
            torch.argmax(probabilities, dim=1).item()
        )

        confidence = float(
            probabilities[0][predicted_id].item()
        )

    sentiment = label_map.get(
        predicted_id,
        f"Unknown-{predicted_id}",
    )

    print("\nText:", text)
    print("Sentiment:", sentiment)
    print("Confidence:", round(confidence * 100, 2), "%")