from pathlib import Path
import gc

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


MODEL_PATHS = {

    "old_muril":
        PROJECT_ROOT
        / "model"
        / "muril_sentiment_model",

    "research_muril":
        PROJECT_ROOT
        / "research"
        / "muril"
        / "best_model",

    "indicbert_v2":
        PROJECT_ROOT
        / "research"
        / "indicbert_v2"
        / "best_model",

    "xlm_roberta":
        PROJECT_ROOT
        / "research"
        / "xlm_roberta"
        / "best_model"
}


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


ID2LABEL = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}


current_model_name = None
current_model = None
current_tokenizer = None


def unload_current_model():

    global current_model_name
    global current_model
    global current_tokenizer

    if current_model is not None:

        print(
            f"\nUnloading model: {current_model_name}"
        )

        del current_model

        current_model = None


    if current_tokenizer is not None:

        del current_tokenizer

        current_tokenizer = None


    current_model_name = None


    gc.collect()


    if torch.cuda.is_available():

        torch.cuda.empty_cache()


def load_model(model_name: str):

    global current_model_name
    global current_model
    global current_tokenizer


    model_name = model_name.lower()


    if model_name not in MODEL_PATHS:

        raise ValueError(
            f"Invalid model '{model_name}'. "
            f"Available models: {list(MODEL_PATHS.keys())}"
        )


    # Already loaded
    if current_model_name == model_name:

        return (
            current_tokenizer,
            current_model
        )


    # Unload previous model
    unload_current_model()


    model_path = MODEL_PATHS[
        model_name
    ]


    if not model_path.exists():

        raise FileNotFoundError(
            f"Model folder not found: {model_path}"
        )


    print("\n==============================================")
    print("LOADING MODEL")
    print("==============================================")

    print(
        "Selected Model :",
        model_name
    )

    print(
        "Model Path     :",
        model_path
    )

    print(
        "Device         :",
        device
    )


    if torch.cuda.is_available():

        print(
            "GPU            :",
            torch.cuda.get_device_name(0)
        )


    print("==============================================\n")


    current_tokenizer = AutoTokenizer.from_pretrained(
        str(model_path)
    )


    current_model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path)
    )


    current_model.to(
        device
    )


    current_model.eval()


    current_model_name = model_name


    print(
        f"{model_name} loaded successfully.\n"
    )


    return (
        current_tokenizer,
        current_model
    )


def predict_sentiment(
    text: str,
    model_name: str
):

    if not text or not text.strip():

        raise ValueError(
            "Input text cannot be empty."
        )


    tokenizer, model = load_model(
        model_name
    )


    inputs = tokenizer(

        text,

        return_tensors="pt",

        truncation=True,

        padding=True,

        max_length=128
    )


    inputs = {

        key: value.to(device)

        for key, value
        in inputs.items()
    }


    with torch.no_grad():

        outputs = model(
            **inputs
        )


        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )


    predicted_id = torch.argmax(
        probabilities,
        dim=-1
    ).item()


    confidence = probabilities[
        0,
        predicted_id
    ].item()


    sentiment = ID2LABEL[
        predicted_id
    ]


    return {

        "sentiment":
            sentiment,

        "confidence":
            round(
                confidence * 100,
                2
            ),

        "model":
            model_name
    }