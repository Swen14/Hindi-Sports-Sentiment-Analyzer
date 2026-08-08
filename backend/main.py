from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from .model_loader import (
    predict_sentiment,
    MODEL_PATHS
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(

    title="Hindi Sports Sentiment Analyzer",

    description=(
        "Hindi / Hinglish Sports Sentiment Analyzer "
        "with selectable transformer models."
    ),

    version="3.0"
)


# ============================================================
# CORS
# ============================================================

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


# ============================================================
# REQUEST MODEL
# ============================================================

class SentimentRequest(BaseModel):

    text: str

    model: str = "old_muril"


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {

        "message":
            "Hindi Sports Sentiment Analyzer API",

        "available_models":
            list(MODEL_PATHS.keys()),

        "default_model":
            "old_muril",

        "labels": {

            "0": "Negative",

            "1": "Neutral",

            "2": "Positive"
        }
    }


# ============================================================
# AVAILABLE MODELS
# ============================================================

@app.get("/models")
def models():

    return {

        "models": [

            {
                "id": "old_muril",
                "name": "Original MuRIL"
            },

            {
                "id": "research_muril",
                "name": "Research MuRIL"
            },

            {
                "id": "indicbert_v2",
                "name": "IndicBERT v2"
            },

            {
                "id": "xlm_roberta",
                "name": "XLM-RoBERTa"
            }

        ]
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {

        "status":
            "running",

        "available_models":
            list(MODEL_PATHS.keys())
    }


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
def predict(
    request: SentimentRequest
):

    try:

        if not request.text.strip():

            raise HTTPException(

                status_code=400,

                detail="Text cannot be empty."
            )


        if request.model not in MODEL_PATHS:

            raise HTTPException(

                status_code=400,

                detail=(
                    f"Invalid model. "
                    f"Available models: "
                    f"{list(MODEL_PATHS.keys())}"
                )
            )


        result = predict_sentiment(

            request.text,

            request.model
        )


        return result


    except HTTPException:

        raise


    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=str(e)
        )