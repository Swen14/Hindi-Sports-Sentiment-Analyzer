"""
Collect real Hindi / Hinglish sports comments from YouTube using the
official YouTube Data API v3 (allowed by YouTube's terms, unlike HTML scraping).

What it does
------------
1. Searches sports videos using QUERIES below (wins AND losses, so you get
   positive AND negative comments).
2. Downloads top-level comments from each video (max MAX_PER_VIDEO per video).
3. Keeps only Hindi comments: Devanagari OR Romanized Hindi.
4. Cleans (removes links/@mentions), removes duplicates and spam-length text.
5. Saves NO usernames - only comment text, comment_id, video_id, date.

Output: raw_comments.csv  (in the folder you run the script from)

Usage (PowerShell, inside the data_collection folder):
    $env:YT_API_KEY = "PASTE_YOUR_KEY_HERE"
    python collect_youtube_comments.py

API quota: each search costs 100 units, each comment page costs 1 unit.
The free daily quota is 10,000 units, so this script uses ~4,000-5,000.
"""

import csv
import os
import re
import sys
import time

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ------------------------------------------------------------------
# SETTINGS - edit these if you want
# ------------------------------------------------------------------

# Mix of wins, losses, controversies -> balanced sentiment.
# Each query costs 100 quota units. 40 queries = 4,000 units.
QUERIES = [
    # Cricket - wins
    "india vs pakistan highlights hindi",
    "india vs australia highlights hindi",
    "india win match highlights hindi",
    "virat kohli century highlights",
    "rohit sharma batting highlights hindi",
    "jasprit bumrah wickets highlights",
    "ipl final highlights hindi",
    "world cup final india highlights hindi",
    # Cricket - losses / criticism
    "india lost match highlights hindi",
    "india haar gaya match reaction",
    "india batting collapse highlights",
    "ipl worst bowling highlights",
    "team india flop show reaction",
    "world cup 2023 final india loss reaction",
    "dropped catch fielding mistakes india",
    # Cricket - neutral / analysis / news
    "india squad announcement hindi",
    "playing 11 prediction hindi",
    "match preview hindi cricket",
    "pitch report hindi",
    "cricket news hindi today",
    # Football
    "football highlights hindi commentary",
    "isl highlights hindi",
    "india football team highlights",
    "messi ronaldo hindi",
    "fifa world cup highlights hindi",
    # Kabaddi
    "pro kabaddi highlights hindi",
    "kabaddi match highlights",
    "pkl best raid hindi",
    # Badminton
    "pv sindhu match highlights",
    "badminton highlights india hindi",
    "lakshya sen match highlights",
    # Formula 1
    "f1 race highlights hindi",
    "formula 1 hindi",
    # Hockey / Olympics / others
    "india hockey highlights hindi",
    "neeraj chopra javelin reaction",
    "olympics india medal reaction hindi",
    "wrestling india olympics reaction",
    # Reactions / fan talk (lots of slang)
    "fans reaction india match hindi",
    "cricket roast hindi",
    "cricket memes hindi reaction",
]

VIDEOS_PER_QUERY = 5        # top videos per search
MAX_PER_VIDEO = 50          # cap so one video doesn't dominate
MIN_WORDS = 3
MAX_WORDS = 100
OUTPUT_FILE = "raw_comments.csv"

# ------------------------------------------------------------------
# HINDI DETECTION
# ------------------------------------------------------------------

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")

# Very common Romanized-Hindi words. A comment needs >= 2 of these
# to count as Romanized Hindi (filters out pure English).
ROMAN_HINDI_WORDS = {
    # (English look-alikes such as "the", "to", "hi", "me" are deliberately
    #  excluded so pure-English comments don't pass the filter)
    "hai", "hain", "tha", "thi", "hoga", "hogi", "nahi", "nhi",
    "kya", "kyu", "kyun", "kaise", "kab", "kaun", "koi", "kuch", "sab",
    "ne", "ki", "ka", "ke", "ko", "se", "mein", "pe", "aur",
    "bhai", "yaar", "bhaiya", "ji", "bas", "abhi", "aaj", "kal",
    "bahut", "bohot", "bht", "ekdum", "sirf", "bhi", "toh",
    "mat", "accha", "acha", "achha", "badiya", "badhiya", "bekar", "bekaar",
    "khela", "khel", "khelna", "maar", "mara", "maara", "diya", "dia",
    "gaya", "gaye", "gayi", "jeet", "jeeta", "jeete", "haar", "haara",
    "hara", "kar", "karo", "kiya", "raha", "rahe", "rahi", "wala", "wali",
    "isko", "usko", "iska", "uska", "apna", "apne", "hum", "humara",
    "tum", "aap", "mera", "meri", "dekh", "dekho", "samajh", "pata",
    "sharam", "zabardast", "kamaal", "shandaar", "faltu", "ghatiya",
}

URL = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"@\S+")
HTML = re.compile(r"<[^>]+>")
SPACES = re.compile(r"\s+")


def clean(text: str) -> str:
    text = HTML.sub(" ", text)
    text = URL.sub(" ", text)
    text = MENTION.sub(" ", text)
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&amp;", "&")
    return SPACES.sub(" ", text).strip()


def detect_script(text: str):
    """Return 'devanagari', 'romanized', or None (not Hindi)."""
    dev = len(DEVANAGARI.findall(text))
    lat = len(LATIN.findall(text))
    if dev >= 5 and dev >= lat:
        return "devanagari"
    words = re.findall(r"[a-z]+", text.lower())
    hits = sum(1 for w in words if w in ROMAN_HINDI_WORDS)
    if hits >= 2 and dev == 0:
        return "romanized"
    return None


def normalize_for_dedup(text: str) -> str:
    return re.sub(r"[^\wऀ-ॿ]+", "", text.lower())


# ------------------------------------------------------------------
# YOUTUBE API
# ------------------------------------------------------------------

def search_videos(youtube, query):
    resp = youtube.search().list(
        q=query,
        part="id",
        type="video",
        maxResults=VIDEOS_PER_QUERY,
        regionCode="IN",
        relevanceLanguage="hi",
        order="relevance",
    ).execute()
    return [item["id"]["videoId"] for item in resp.get("items", [])]


def fetch_comments(youtube, video_id, limit):
    comments, token = [], None
    while len(comments) < limit * 4:   # fetch extra, most get filtered out
        try:
            resp = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                order="relevance",
                textFormat="plainText",
                pageToken=token,
            ).execute()
        except HttpError as e:
            # comments disabled or video unavailable -> skip video
            print(f"   skip {video_id}: {e.resp.status}")
            return comments
        for item in resp.get("items", []):
            s = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "comment_id": item["snippet"]["topLevelComment"]["id"],
                "video_id": video_id,
                "published": s.get("publishedAt", "")[:10],
                "text": s.get("textOriginal") or s.get("textDisplay", ""),
            })
        token = resp.get("nextPageToken")
        if not token:
            break
    return comments


def main():
    api_key = os.environ.get("YT_API_KEY")
    if not api_key:
        sys.exit("ERROR: set your key first:  $env:YT_API_KEY = \"...\"")

    youtube = build("youtube", "v3", developerKey=api_key)

    seen_videos, seen_text = set(), set()
    rows = []

    for qi, query in enumerate(QUERIES, 1):
        print(f"[{qi}/{len(QUERIES)}] {query}")
        try:
            video_ids = search_videos(youtube, query)
        except HttpError as e:
            if e.resp.status == 403:
                print("Quota finished for today. Saving what we have.")
                break
            raise

        for vid in video_ids:
            if vid in seen_videos:
                continue
            seen_videos.add(vid)

            kept = 0
            for c in fetch_comments(youtube, vid, MAX_PER_VIDEO):
                if kept >= MAX_PER_VIDEO:
                    break
                text = clean(c["text"])
                n_words = len(text.split())
                if not (MIN_WORDS <= n_words <= MAX_WORDS):
                    continue
                script = detect_script(text)
                if script is None:
                    continue
                key = normalize_for_dedup(text)
                if key in seen_text:
                    continue
                seen_text.add(key)

                rows.append({
                    "comment_id": c["comment_id"],
                    "video_id": vid,
                    "query": query,
                    "published": c["published"],
                    "script": script,
                    "text": text,
                })
                kept += 1
            print(f"   {vid}: kept {kept}")
        time.sleep(0.2)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["comment_id", "video_id", "query",
                           "published", "script", "text"]
        )
        writer.writeheader()
        writer.writerows(rows)

    dev = sum(r["script"] == "devanagari" for r in rows)
    print("\nDONE")
    print(f"Videos used     : {len(seen_videos)}")
    print(f"Comments saved  : {len(rows)}")
    print(f"  Devanagari    : {dev}")
    print(f"  Romanized     : {len(rows) - dev}")
    print(f"File            : {os.path.abspath(OUTPUT_FILE)}")


if __name__ == "__main__":
    main()
