import json
import csv
import spacy
from pathlib import Path
from collections import Counter
from tqdm import tqdm

PROCESSED_DIR = Path(__file__).parent.parent / "training" / "data" / "processed"
VOCAB_PATH = PROCESSED_DIR / "vocab.json"
TOKENS_PATH = PROCESSED_DIR / "filtered_tokens.txt"
METADATA_CSV = PROCESSED_DIR / "vocab_metadata.csv"
METADATA_JSON = PROCESSED_DIR / "vocab_metadata.json"

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# load vocab
with open(VOCAB_PATH) as f:
    word_to_idx = json.load(f)
words = list(word_to_idx.keys())

# count actual frequencies from corpus
print("Counting frequencies...")
counts = Counter()
with open(TOKENS_PATH, encoding="utf-8") as f:
    for line in tqdm(f, desc="Reading corpus"):
        counts.update(line.strip().split())

# tag POS for each vocab word
print("Tagging POS...")
metadata = {}
for doc in tqdm(nlp.pipe(words, batch_size=1000), total=len(words), desc="Tagging"):
    w = doc[0].text
    metadata[w] = {
        "freq_rank": word_to_idx[w],
        "frequency": counts.get(w, 0),
        "pos": doc[0].pos_
    }

# save json
with open(METADATA_JSON, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
print(f"Saved {METADATA_JSON}")

# save csv
with open(METADATA_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["word", "freq_rank", "frequency", "pos"])
    writer.writeheader()
    for word, meta in sorted(metadata.items(), key=lambda x: x[1]["freq_rank"]):
        writer.writerow({"word": word, **meta})
print(f"Saved {METADATA_CSV}")
print(f"Total words: {len(metadata):,}")