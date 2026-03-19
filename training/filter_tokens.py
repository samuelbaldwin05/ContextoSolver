import nltk
from nltk.corpus import words as nltk_words
from pathlib import Path
from tqdm import tqdm

nltk.download('words', quiet=True)

PROCESSED_DIR = Path(__file__).parent / "data" / "processed"
TOKENS_PATH = PROCESSED_DIR / "tokens.txt"
FILTERED_PATH = PROCESSED_DIR / "filtered_tokens.txt"


def build_english_set() -> set[str]:
    """Load NLTK words corpus into a lowercase set for O(1) lookup."""
    return set(w.lower() for w in nltk_words.words())


def filter_tokens(tokens_path: Path, filtered_path: Path, english: set[str]) -> None:
    """Stream tokens.txt, filter to English words only, write to filtered_tokens.txt."""
    total = sum(1 for _ in open(tokens_path, encoding="utf-8"))
    kept_total = 0
    dropped_total = 0

    with open(tokens_path, encoding="utf-8") as f_in, \
         open(filtered_path, "w", encoding="utf-8") as f_out, \
         tqdm(total=total, unit=" articles", desc="Filtering") as bar:

        for line in f_in:
            tokens = line.strip().split()
            filtered = [t for t in tokens if t in english]
            if filtered:
                f_out.write(" ".join(filtered) + "\n")
            kept_total += len(filtered)
            dropped_total += len(tokens) - len(filtered)
            bar.update(1)

    print(f"Kept {kept_total:,} tokens, dropped {dropped_total:,} tokens")


if __name__ == "__main__":
    if FILTERED_PATH.exists():
        print(f"filtered_tokens.txt already exists at {FILTERED_PATH}, skipping.")
    else:
        english = build_english_set()
        print(f"English wordlist size: {len(english):,}")
        filter_tokens(TOKENS_PATH, FILTERED_PATH, english)