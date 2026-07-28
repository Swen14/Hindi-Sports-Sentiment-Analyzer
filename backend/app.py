import time
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model" / "muril_sentiment_model"

ID_TO_LABEL = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

torch.set_num_threads(4)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

app = FastAPI(
    title="Hindi Sports Sentiment API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


class SentimentRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=500
    )


class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    processing_time: float


print("Loading trained MuRIL model...")

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

print("Model loaded successfully.")
print("Device:", device)


def predict_sentiment(text: str):
    cleaned_text = text.strip()

    if not cleaned_text:
        raise ValueError("Text cannot be empty.")

    start_time = time.perf_counter()

    inputs = tokenizer(
        cleaned_text,
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

    processing_time = time.perf_counter() - start_time

    print(
        f"Prediction completed in "
        f"{processing_time:.2f} seconds"
    )

    return {
        "text": cleaned_text,
        "sentiment": ID_TO_LABEL[predicted_id],
        "confidence": round(confidence * 100, 2),
        "processing_time": round(processing_time, 2)
    }


@app.on_event("startup")
def warm_up_model():
    print("Warming up model...")

    try:
        predict_sentiment(
            "भारत ने शानदार जीत हासिल की।"
        )
        print("Model warm-up completed.")
    except Exception as error:
        print("Warm-up failed:", error)


@app.get("/")
def root():
    return {
        "message": "Hindi Sports Sentiment API is running."
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "device": str(device),
        "model_loaded": True
    }


@app.post(
    "/predict",
    response_model=SentimentResponse
)
def predict(request: SentimentRequest):
    try:
        return predict_sentiment(request.text)

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        print("Prediction error:", error)

        raise HTTPException(
            status_code=500,
            detail="Prediction failed."
        )