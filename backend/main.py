from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .benchmark import get_evaluation
from .model_loader import predict_sentiment, MODEL_PATHS
from .model_registry import (
    MODELS,
    TEST_SETS,
    TRAINING_GROUPS,
    model_summary
)


DEFAULT_MODEL = "real_muril"


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Hindi Sports Sentiment Analyzer",
    description=(
        "Hindi sports sentiment analysis with six transformer models: "
        "MuRIL, IndicBERT v2 and XLM-RoBERTa, each trained once on "
        "synthetic data and once on real-world comments."
    ),
    version="5.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


class SentimentRequest(BaseModel):
    text: str
    model: str = DEFAULT_MODEL


def check_model(model_id: str):
    if model_id not in MODEL_PATHS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model. Available models: {list(MODELS)}"
        )


# ============================================================
# INFO
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Hindi Sports Sentiment Analyzer API",
        "available_models": list(MODELS),
        "default_model": DEFAULT_MODEL,
        "labels": {"0": "Negative", "1": "Neutral", "2": "Positive"}
    }


@app.get("/health")
def health():
    return {
        "status": "running",
        "available_models": [
            model_id for model_id, entry in MODELS.items()
            if entry["path"].exists()
        ]
    }


@app.get("/models")
def models():
    return {
        "default_model": DEFAULT_MODEL,
        "groups": TRAINING_GROUPS,
        "test_sets": TEST_SETS,
        "models": [model_summary(model_id) for model_id in MODELS]
    }


@app.get("/models/{model_id}")
def model_details(model_id: str):
    if model_id not in MODELS:
        check_model(model_id)
    return model_summary(model_id)


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
def predict(request: SentimentRequest):

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    check_model(request.model)

    try:
        return predict_sentiment(request.text, request.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# EVALUATION
# ============================================================

@app.get("/evaluation/{model_id}")
def evaluation(model_id: str, test_set: str = "own", refresh: bool = False):
    """
    test_set: "own" (the test split of the model's training data),
              "synthetic" or "real".
    Results are cached in research/results/evaluation/.
    """

    if model_id not in MODELS:
        check_model(model_id)

    if test_set == "own":
        test_set = MODELS[model_id]["group"]

    if test_set not in TEST_SETS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid test set. Use one of: own, {', '.join(TEST_SETS)}"
        )

    try:
        return get_evaluation(model_id, test_set, refresh=refresh)
    except Exception as e:
        print("\nEvaluation error:", e)
        raise HTTPException(status_code=500, detail=str(e))
