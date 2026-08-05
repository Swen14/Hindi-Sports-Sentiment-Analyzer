from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


app = FastAPI(
    title="Hindi Sports Sentiment API",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "muril_sentiment_model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()


label_map = {
    int(key): value.capitalize()
    for key, value in model.config.id2label.items()
}


class SentimentRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {
        "message": "Hindi Sports Sentiment API is running",
        "device": str(device),
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }


@app.post("/predict")
def predict_sentiment(request: SentimentRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Please enter a Hindi sports sentence.",
        )

    try:
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

            probabilities = torch.softmax(
                outputs.logits,
                dim=1,
            )

            predicted_id = int(
                torch.argmax(
                    probabilities,
                    dim=1,
                ).item()
            )

            confidence = float(
                probabilities[0][predicted_id].item()
            )

        sentiment = label_map.get(
            predicted_id,
            f"Unknown-{predicted_id}",
        )

        return {
            "text": text,
            "sentiment": sentiment,
            "confidence": round(confidence * 100, 2),
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}",
        )