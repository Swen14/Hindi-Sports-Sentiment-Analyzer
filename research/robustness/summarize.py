"""
Combine the robustness runs into mean +/- standard deviation tables.

Seed 42 of the comment-level split comes from the models used in the paper
(research/results/evaluation/*.json); seeds 7 and 123 come from
research/results/robustness/. Writes summary.json and summary.md next to the runs.

Usage (from the project root):
    venv\\Scripts\\python research\\robustness\\summarize.py
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
EVAL_DIR = os.path.join(PROJECT_ROOT, "research", "results", "evaluation")
ROB_DIR = os.path.join(PROJECT_ROOT, "research", "results", "robustness")
BASELINE = os.path.join(PROJECT_ROOT, "research", "results", "baseline", "tfidf_lr_results.json")

MODELS = ["muril", "indicbert_v2", "xlm_roberta"]
NAMES = {"muril": "MuRIL", "indicbert_v2": "IndicBERT v2", "xlm_roberta": "XLM-RoBERTa"}
SYN_ID = {"muril": "research_muril", "indicbert_v2": "indicbert_v2", "xlm_roberta": "xlm_roberta"}
SEEDS = [42, 7, 123]


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def from_evaluation(model_id, test_set):
    """Seed-42 scores from the cached evaluation used in the paper."""
    d = load(os.path.join(EVAL_DIR, f"{model_id}__{test_set}.json"))
    report = d["classification_report"]
    return {
        "accuracy": d["accuracy"],
        "macro_f1": d["macro_f1"],
        "class_f1": {c: report[c]["f1-score"] for c in ["Negative", "Neutral", "Positive"]}
    }


def collect():
    runs = {}  # (setting, model) -> {test_set: [scores per seed]}

    def add(setting, model, test_set, score):
        runs.setdefault((setting, model), {}).setdefault(test_set, []).append(score)

    for m in MODELS:
        # Real-world models, comment-level split
        add("real_comment", m, "real_comment", from_evaluation(f"real_{m}", "real"))
        add("real_comment", m, "synthetic", from_evaluation(f"real_{m}", "synthetic"))
        for seed in SEEDS[1:]:
            r = load(os.path.join(ROB_DIR, "seeds_real", f"{m}_seed{seed}.json"))
            for t in ["real_comment", "synthetic"]:
                add("real_comment", m, t, r["test"][t])

        # Real-world models, video-level split
        for seed in SEEDS:
            r = load(os.path.join(ROB_DIR, "video_split", f"{m}_seed{seed}.json"))
            for t in ["real_video", "synthetic"]:
                add("real_video", m, t, r["test"][t])

        # Synthetic-data models
        add("synthetic", m, "synthetic", from_evaluation(SYN_ID[m], "synthetic"))
        add("synthetic", m, "real_comment", from_evaluation(SYN_ID[m], "real"))
        add("synthetic", m, "real_video",
            load(os.path.join(ROB_DIR, "existing", f"{m}_synthetic_seed42_video_test.json"))["test"]["real_video"])
        for seed in SEEDS[1:]:
            r = load(os.path.join(ROB_DIR, "seeds_synthetic", f"{m}_seed{seed}.json"))
            for t in ["synthetic", "real_comment", "real_video"]:
                add("synthetic", m, t, r["test"][t])
    return runs


def stats(values):
    a = np.array(values, dtype=float) * 100
    return {"mean": round(float(a.mean()), 2), "std": round(float(a.std(ddof=1)), 2),
            "runs": [round(float(v), 2) for v in a]}


def main():
    runs = collect()
    summary = {}
    for (setting, model), tests in runs.items():
        for test_set, score_list in tests.items():
            summary.setdefault(setting, {}).setdefault(model, {})[test_set] = {
                "macro_f1": stats([s["macro_f1"] for s in score_list]),
                "accuracy": stats([s["accuracy"] for s in score_list]),
                "neutral_f1": stats([s["class_f1"]["Neutral"] for s in score_list]),
                "seeds": len(score_list)
            }

    baseline = load(BASELINE)
    summary["tfidf_lr"] = baseline

    with open(os.path.join(ROB_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    def cell(setting, model, test_set, metric="macro_f1"):
        s = summary[setting][model].get(test_set)
        return f"{s[metric]['mean']:.2f} ± {s[metric]['std']:.2f}" if s else "–"

    def base(train, test):
        t = baseline[train]["test"].get(test)
        return f"{t['macro_f1'] * 100:.2f}" if t else "–"

    lines = ["# Robustness results (macro F1 %, mean ± std over seeds 42, 7, 123)", "",
             "| Model | Trained on | Split | Synthetic test | Real test (comment split) | Real test (video split) |",
             "|---|---|---|---|---|---|"]
    for m in MODELS:
        lines.append(f"| {NAMES[m]} | Synthetic | – | {cell('synthetic', m, 'synthetic')} | "
                     f"{cell('synthetic', m, 'real_comment')} | {cell('synthetic', m, 'real_video')} |")
    for m in MODELS:
        lines.append(f"| {NAMES[m]} | Real | comment | {cell('real_comment', m, 'synthetic')} | "
                     f"{cell('real_comment', m, 'real_comment')} | – |")
    for m in MODELS:
        lines.append(f"| {NAMES[m]} | Real | video | {cell('real_video', m, 'synthetic')} | – | "
                     f"{cell('real_video', m, 'real_video')} |")
    lines.append(f"| TF-IDF + LR | Synthetic | – | {base('synthetic', 'synthetic')} | "
                 f"{base('synthetic', 'real_comment')} | {base('synthetic', 'real_video')} |")
    lines.append(f"| TF-IDF + LR | Real | comment | {base('real_comment', 'synthetic')} | "
                 f"{base('real_comment', 'real_comment')} | – |")
    lines.append(f"| TF-IDF + LR | Real | video | {base('real_video', 'synthetic')} | – | "
                 f"{base('real_video', 'real_video')} |")

    lines += ["", "## Accuracy (%) on the real-world test set", "",
              "| Model | Synthetic-trained (comment test) | Real-trained, comment split | Real-trained, video split |",
              "|---|---|---|---|"]
    for m in MODELS:
        lines.append(f"| {NAMES[m]} | {cell('synthetic', m, 'real_comment', 'accuracy')} | "
                     f"{cell('real_comment', m, 'real_comment', 'accuracy')} | "
                     f"{cell('real_video', m, 'real_video', 'accuracy')} |")

    lines += ["", "## Neutral-class F1 (%) on the real-world test set", "",
              "| Model | Real-trained, comment split | Real-trained, video split |", "|---|---|---|"]
    for m in MODELS:
        lines.append(f"| {NAMES[m]} | {cell('real_comment', m, 'real_comment', 'neutral_f1')} | "
                     f"{cell('real_video', m, 'real_video', 'neutral_f1')} |")

    with open(os.path.join(ROB_DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
