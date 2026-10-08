"""
Turn raw_comments.csv into 3 separate labeling sheets (one per annotator).

- Samples N comments, keeping a balance of Devanagari and Romanized
- Shuffles them (same order for everyone, so ids line up)
- Annotator sheets contain ONLY id + text (no script/video info -> no bias)
- master_key.csv keeps the hidden metadata (comment_id, video_id, script)

Usage (PowerShell, inside data_collection):
    python make_labeling_sheets.py            # default 1500 comments
    python make_labeling_sheets.py 1200       # or choose a number
"""

import csv
import random
import sys

INPUT = "raw_comments.csv"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
ANNOTATORS = ["A", "B", "C"]
SEED = 42

with open(INPUT, encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

dev = [r for r in rows if r["script"] == "devanagari"]
rom = [r for r in rows if r["script"] == "romanized"]

rng = random.Random(SEED)
rng.shuffle(dev)
rng.shuffle(rom)

# Aim for 50/50 scripts; if one side is short, fill from the other.
half = N // 2
take_dev = min(len(dev), half)
take_rom = min(len(rom), N - take_dev)
take_dev = min(len(dev), N - take_rom)
sample = dev[:take_dev] + rom[:take_rom]
rng.shuffle(sample)

with open("master_key.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["id", "comment_id", "video_id", "query", "script", "text"])
    for i, r in enumerate(sample, 1):
        w.writerow([i, r["comment_id"], r["video_id"], r["query"],
                    r["script"], r["text"]])

for a in ANNOTATORS:
    with open(f"annotator_{a}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "text", "label", "notes"])
        for i, r in enumerate(sample, 1):
            w.writerow([i, r["text"], "", ""])

print(f"Available : {len(dev)} Devanagari, {len(rom)} Romanized")
print(f"Sampled   : {len(sample)}  ({take_dev} Devanagari, {take_rom} Romanized)")
print("Created   : master_key.csv, " + ", ".join(f"annotator_{a}.csv" for a in ANNOTATORS))
print("Label values allowed: positive / neutral / negative / skip")
