"""
Evaluate all six models on BOTH test sets (synthetic and real-world)
and cache the results for the frontend.

Also writes training_info.json for each model (parameter count,
validation history, training time).

Usage (from the project root):
    venv\\Scripts\\python research\\evaluate_all.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.benchmark import count_parameters, get_evaluation  # noqa: E402
from backend.model_registry import (  # noqa: E402
    MODELS,
    TEST_SETS,
    training_info_path,
    read_json
)


def synthetic_training_info(model_id: str) -> dict:
    """Rebuild training history from the saved Trainer checkpoint."""

    checkpoints = sorted(
        (MODELS[model_id]["path"].parent / "output").glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1])
    )
    if not checkpoints:
        return {}

    state = read_json(checkpoints[-1] / "trainer_state.json") or {}
    log = state.get("log_history", [])

    history = [
        {
            "epoch": round(entry["epoch"]),
            "val_macro_f1": round(entry["eval_macro_f1"], 4),
            "val_accuracy": round(entry["eval_accuracy"], 4),
            "val_loss": round(entry["eval_loss"], 4)
        }
        for entry in log
        if "eval_macro_f1" in entry
    ]

    runtime = next(
        (entry["train_runtime"] for entry in log if "train_runtime" in entry),
        None
    )

    return {
        "history": history,
        "training_minutes": round(runtime / 60, 1) if runtime else None
    }


def main():

    summary = []

    for model_id, entry in MODELS.items():

        if not entry["path"].exists():
            print(f"SKIP {model_id}: model folder not found")
            continue

        print(f"\n=== {model_id} ===")

        info_path = training_info_path(model_id)
        info = read_json(info_path) or {}

        if entry["group"] == "synthetic" and not info.get("history"):
            info.update(synthetic_training_info(model_id))

        info["parameters"] = count_parameters(model_id)

        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

        for test_set in TEST_SETS:
            result = get_evaluation(model_id, test_set, refresh=True)
            print(
                f"{test_set:>9} test | "
                f"accuracy {result['accuracy']:.4f} | "
                f"macro F1 {result['macro_f1']:.4f}"
            )
            summary.append(
                (model_id, test_set, result["accuracy"], result["macro_f1"])
            )

    print("\nSUMMARY")
    print(f"{'model':<20}{'test set':<12}{'accuracy':>10}{'macro F1':>10}")
    for model_id, test_set, accuracy, f1 in summary:
        print(f"{model_id:<20}{test_set:<12}{accuracy:>10.4f}{f1:>10.4f}")


if __name__ == "__main__":
    main()
