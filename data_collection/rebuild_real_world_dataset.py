"""
Rebuild the real-world dataset (with comment text) from
dataset/real_world/real_world_labels.csv.

The public repository only stores comment IDs and labels, because
YouTube's terms do not allow republishing comment text in bulk.
This script downloads the text again through the official
YouTube Data API v3 and recreates:

    dataset/real_world/real_world_dataset.csv
    dataset/real_world/real_train.csv
    dataset/real_world/real_validation.csv
    dataset/real_world/real_test.csv

Usage (PowerShell, from the project root):
    $env:YT_API_KEY = "PASTE_YOUR_KEY_HERE"
    python data_collection\\rebuild_real_world_dataset.py

Comments that were deleted since collection are skipped, so the
rebuilt splits can be slightly smaller than the original ones.
Quota: 1 unit per 50 comments (about 75 units in total).
"""

import os
import sys
from pathlib import Path

import pandas as pd
from googleapiclient.discovery import build


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "dataset" / "real_world"


def main():

    api_key = os.environ.get("YT_API_KEY", "").strip()
    if not api_key:
        sys.exit("Set the YT_API_KEY environment variable first.")

    labels = pd.read_csv(DATA_DIR / "real_world_labels.csv")
    youtube = build("youtube", "v3", developerKey=api_key)

    texts = {}
    ids = labels["comment_id"].astype(str).tolist()

    for start in range(0, len(ids), 50):
        response = youtube.comments().list(
            part="snippet",
            id=",".join(ids[start:start + 50]),
            textFormat="plainText",
            maxResults=50
        ).execute()

        for item in response.get("items", []):
            texts[item["id"]] = item["snippet"]["textDisplay"]

        print(f"{min(start + 50, len(ids))}/{len(ids)} comments checked")

    labels["text"] = labels["comment_id"].map(texts)
    found = labels.dropna(subset=["text"])

    print(f"Recovered {len(found)} of {len(labels)} comments")

    columns = ["comment_id", "video_id", "topic", "text", "sentiment", "label"]
    found[columns].to_csv(
        DATA_DIR / "real_world_dataset.csv", index=False, encoding="utf-8-sig"
    )

    for split in ["train", "validation", "test"]:
        found[found["split"] == split][columns].to_csv(
            DATA_DIR / f"real_{split}.csv", index=False, encoding="utf-8-sig"
        )


if __name__ == "__main__":
    main()
