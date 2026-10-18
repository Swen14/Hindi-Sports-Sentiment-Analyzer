"""Step 1: Find Hindi SPORTS videos on YouTube automatically and download their comments.
Put your YouTube API key in api_key.txt, then run:  python collect.py
Run it again on later days to get more (it skips videos it already used).
"""
import re, sys, time
from datetime import date
from pathlib import Path
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

HERE = Path(__file__).parent
OUT = HERE / "hindi_comments_raw.csv"

# Sports-only search topics. Wins, losses, controversies -> positive AND negative comments.
QUERIES = [
    "क्रिकेट हाइलाइट्स हिंदी", "भारत बनाम पाकिस्तान क्रिकेट", "टीम इंडिया हार",
    "टीम इंडिया जीत", "आईपीएल हाइलाइट्स हिंदी", "विराट कोहली", "रोहित शर्मा",
    "एमएस धोनी", "शुभमन गिल", "क्रिकेट विश्लेषण हिंदी", "वर्ल्ड कप क्रिकेट हिंदी",
    "एशिया कप हिंदी", "प्रो कबड्डी", "फुटबॉल हिंदी", "नीरज चोपड़ा",
    "ओलंपिक भारत हिंदी", "हॉकी इंडिया", "विनेश फोगाट कुश्ती", "पीवी सिंधु बैडमिंटन",
    "खेल समाचार हिंदी",
]
VIDEOS_PER_QUERY = 10
MAX_COMMENTS_PER_VIDEO = 1000
# Different ordering each day so repeat runs find new videos
ORDER = ["relevance", "viewCount", "date"][date.today().toordinal() % 3]

# Common Hindi words typed in English letters, used to spot Hinglish
HINGLISH_WORDS = set("""hai hain nahi nhi nahin kya kyu kyun bhai bhi aur ka ki ke ko se mein me
mai tha thi the ho hota hoga raha rahi rahe kar karo karna yaar yr bahut bohot bht acha accha achha
sab kuch koi kab jab tab abhi ab wala wali wale apna apne hum tum aap ye yeh wo woh hi toh jo
dekho dekh khel khela match jeet haar gaya gayi diya liya bas ekdum mast bekar bakwas""".split())

sys.stdout.reconfigure(encoding="utf-8")

key_file = HERE / "api_key.txt"
api_key = key_file.read_text(encoding="utf-8").strip() if key_file.exists() else ""
if not api_key or api_key.startswith("PASTE"):
    sys.exit("Open api_key.txt and paste your YouTube API key into it first.")

yt = build("youtube", "v3", developerKey=api_key)


MARATHI_WORDS = set("""आहे नाही आणि मी आम्ही तुम्ही मला तुला माझा माझी तुझा आपण
त्याला त्याचा त्याची झाला झाली केला केली बघतो खूप""".split())


def script_of(text):
    """devanagari = Hindi letters, marathi, hinglish = Hindi in English letters, other = English etc."""
    dev = len(re.findall(r"[ऀ-ॿ]", text))
    lat = len(re.findall(r"[A-Za-z]", text))
    if dev > 0 and dev / (dev + lat) >= 0.6:
        words = re.findall(r"[ऀ-ॿ]+", text)
        if "ळ" in text or sum(w in MARATHI_WORDS for w in words) >= 2:
            return "marathi"
        return "devanagari"
    words = re.findall(r"[a-z]+", text.lower())
    if words and sum(w in HINGLISH_WORDS for w in words) >= max(2, len(words) * 0.2):
        return "hinglish"
    return "other"


def clean(text):
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\S+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def find_videos(query):
    resp = yt.search().list(
        part="id", q=query, type="video", maxResults=VIDEOS_PER_QUERY,
        relevanceLanguage="hi", regionCode="IN", order=ORDER,
        videoCategoryId="17").execute()   # 17 = YouTube "Sports" category
    return [item["id"]["videoId"] for item in resp["items"]]


def fetch_comments(video_id, topic):
    rows, token = [], None
    while len(rows) < MAX_COMMENTS_PER_VIDEO:
        try:
            resp = yt.commentThreads().list(
                part="snippet", videoId=video_id, maxResults=100,
                pageToken=token, textFormat="plainText").execute()
        except HttpError as e:
            if e.status_code == 403 and "quota" in str(e).lower():
                raise
            print(f"    skipped (comments off or error {e.status_code})")
            break
        for item in resp["items"]:
            s = item["snippet"]["topLevelComment"]["snippet"]
            rows.append({"comment_id": item["id"], "video_id": video_id, "topic": topic,
                         "text": s["textDisplay"], "likes": s["likeCount"]})
        token = resp.get("nextPageToken")
        if not token:
            break
        time.sleep(0.2)
    return rows


old = None
seen_videos = set()
if OUT.exists():
    old = pd.read_csv(OUT, encoding="utf-8-sig", dtype=str).fillna("")
    seen_videos = set(old["video_id"])

videos = {}   # video_id -> search topic
for q in QUERIES:
    try:
        found = find_videos(q)
    except HttpError as e:
        if "quota" in str(e).lower():
            print("Daily YouTube quota used up - saving what we have. Run again tomorrow.")
            break
        sys.exit(f"API error: {e}\nCheck that your key is correct and YouTube Data API v3 is enabled.")
    new = [v for v in found if v not in seen_videos and v not in videos]
    print(f"Search '{q}': {len(new)} new videos")
    for v in new:
        videos[v] = q

all_rows = []
try:
    for i, (vid, topic) in enumerate(videos.items(), 1):
        print(f"[{i}/{len(videos)}] video {vid}")
        all_rows += fetch_comments(vid, topic)
except HttpError:
    print("Daily YouTube quota used up - saving what we have. Run again tomorrow.")

df = pd.DataFrame(all_rows, columns=["comment_id", "video_id", "topic", "text", "likes"])
total = len(df)
df["text"] = df["text"].apply(clean)
df["script"] = df["text"].apply(script_of)
df = df[df["script"] == "devanagari"]   # Hindi only
df = df[df["text"].str.split().str.len() >= 3]
df = df.drop_duplicates(subset="text")
df["label"] = ""

if old is not None:
    df = df[~df["text"].isin(old["text"])]
    df = pd.concat([old, df.astype(str)], ignore_index=True)

df.to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"\nDownloaded {total} comments this run.")
print("Total in dataset:")
print(df["script"].value_counts().to_string())
print("Next: python label.py")
