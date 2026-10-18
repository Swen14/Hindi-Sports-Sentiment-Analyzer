"""
Step-by-step demo of the preprocessing used for the real-world dataset.

Shows, for a few example comments, what every step does:
  1. noise removal (links, @mentions, extra spaces)       - data_collection/collect.py: clean()
  2. language filter (>= 60% Devanagari, no Marathi)      - data_collection/collect.py: script_of()
  3. length filter (at least 3 words)                      - data_collection/collect.py
  4. label to id (Negative 0, Neutral 1, Positive 2)
  5. subword tokenization (max 128 tokens)                 - each model's own tokenizer

Usage (from the project root):
    venv\\Scripts\\python research\\preprocessing_demo.py
"""

import re
import sys

from transformers import AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")

MARATHI_WORDS = set("""आहे नाही आणि मी आम्ही तुम्ही मला तुला माझा माझी तुझा आपण
त्याला त्याचा त्याची झाला झाली केला केली बघतो खूप""".split())

LABEL_IDS = {"Negative": 0, "Neutral": 1, "Positive": 2}

# Same functions as data_collection/collect.py
def clean(text):
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\S+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def devanagari_share(text):
    dev = len(re.findall(r"[ऀ-ॿ]", text))
    lat = len(re.findall(r"[A-Za-z]", text))
    return dev / (dev + lat) if dev + lat else 0.0


def is_marathi(text):
    words = re.findall(r"[ऀ-ॿ]+", text)
    return "ळ" in text or sum(w in MARATHI_WORDS for w in words) >= 2


EXAMPLES = [
    ("@cricketfan वाह क्या बैटिंग है,    फिर से जीरो पर आउट 😂 https://youtu.be/xyz", "Negative"),
    ("रोहित शर्मा तो खुद ही एक ब्रँड है ❤🔥", "Positive"),
    ("aaj ki batting ekdum bekaar thi yaar", "Negative"),
    ("आज खूप छान खेळ झाला, मला आवडला", "Positive"),
    ("शानदार 👏", "Positive"),
]


def main():
    tokenizer = AutoTokenizer.from_pretrained("research/real_world/indicbert_v2/best_model")

    for i, (raw, label) in enumerate(EXAMPLES, 1):
        print("=" * 70)
        print(f"Example {i}")
        print(f"  Raw comment        : {raw}")

        text = clean(raw)
        print(f"  1. Noise removed   : {text}")

        share = devanagari_share(text)
        print(f"  2. Devanagari share: {share:.0%}", end="")
        if share < 0.6:
            print("  -> REMOVED (not Hindi in Devanagari script)")
            continue
        if is_marathi(text):
            print("  -> REMOVED (Marathi)")
            continue
        print("  -> kept")

        n_words = len(text.split())
        print(f"  3. Word count      : {n_words}", end="")
        if n_words < 3:
            print("  -> REMOVED (shorter than 3 words)")
            continue
        print("  -> kept")

        print(f"  4. Label           : {label} -> id {LABEL_IDS[label]}")

        tokens = tokenizer.tokenize(text)
        ids = tokenizer(text, truncation=True, max_length=128)["input_ids"]
        print(f"  5. Tokens          : {tokens}")
        print(f"     Token ids       : {ids}")

    print("=" * 70)
    print("Emojis are kept on purpose: they often carry the sentiment (😂 = mocking, ❤ = praise).")


if __name__ == "__main__":
    main()
