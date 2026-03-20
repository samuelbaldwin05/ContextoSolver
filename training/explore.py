import json
import sys
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent / "data" / "processed"
VOCAB_PATH = PROCESSED_DIR / "vocab.json"
VECTORS_PATH = PROCESSED_DIR / "vectors.npy"
TOP_N = 10


def load(vocab_path: Path, vectors_path: Path) -> tuple[dict[str, int], dict[int, str], np.ndarray]:
    """Load vocab and vectors from disk."""
    with open(vocab_path, encoding="utf-8") as f:
        word_to_idx = json.load(f)
    idx_to_word = {i: w for w, i in word_to_idx.items()}
    vectors = np.load(vectors_path)
    return word_to_idx, idx_to_word, vectors


def get_vector(word: str, word_to_idx: dict[str, int], vectors: np.ndarray) -> np.ndarray | None:
    """Return the vector for a word, or None if not in vocab."""
    if word not in word_to_idx:
        print(f"  '{word}' not in vocab")
        return None
    return vectors[word_to_idx[word]]


def nearest(vec: np.ndarray, vectors: np.ndarray, idx_to_word: dict[int, str],
            exclude: set[str], n: int = TOP_N) -> list[tuple[str, float]]:
    """Return the n nearest words to a vector by cosine similarity, excluding given words."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normed = vectors / (norms + 1e-8)
    query = vec / (np.linalg.norm(vec) + 1e-8)
    sims = normed @ query
    top = np.argsort(-sims)
    results = []
    for i in top:
        word = idx_to_word[i]
        if word not in exclude:
            results.append((word, float(sims[i])))
        if len(results) >= n:
            break
    return results


def parse_expression(tokens: list[str]) -> tuple[list[str], list[str]] | None:
    """Parse a word arithmetic expression into positive and negative word lists.

    Supports expressions like: cat + dog - tree
    Returns (positive_words, negative_words) or None on parse error.
    """
    positive, negative = [], []
    current_sign = 1
    for token in tokens:
        if token == "+":
            current_sign = 1
        elif token == "-":
            current_sign = -1
        else:
            if current_sign == 1:
                positive.append(token)
            else:
                negative.append(token)
            current_sign = 1
    if not positive and not negative:
        return None
    return positive, negative


def main() -> None:
    """Parse CLI args and run nearest neighbor or word arithmetic query."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python training/test.py <word>")
        print("  python training/test.py <word> + <word> - <word>")
        sys.exit(1)

    word_to_idx, idx_to_word, vectors = load(VOCAB_PATH, VECTORS_PATH)

    tokens = sys.argv[1:]

    if len(tokens) == 1:
        word = tokens[0].lower()
        vec = get_vector(word, word_to_idx, vectors)
        if vec is None:
            sys.exit(1)
        results = nearest(vec, vectors, idx_to_word, exclude={word})
        print(f"\nNearest to '{word}':")
        for w, sim in results:
            print(f"  {w:<20} {sim:.4f}")

    else:
        parsed = parse_expression([t.lower() for t in tokens])
        if parsed is None:
            print("Could not parse expression.")
            sys.exit(1)

        positive, negative = parsed
        result_vec = np.zeros(vectors.shape[1])
        all_words = set()

        for word in positive:
            vec = get_vector(word, word_to_idx, vectors)
            if vec is None:
                sys.exit(1)
            result_vec += vec
            all_words.add(word)

        for word in negative:
            vec = get_vector(word, word_to_idx, vectors)
            if vec is None:
                sys.exit(1)
            result_vec -= vec
            all_words.add(word)

        expr = " ".join(tokens)
        results = nearest(result_vec, vectors, idx_to_word, exclude=all_words)
        print(f"\nNearest to '{expr}':")
        for w, sim in results:
            print(f"  {w:<20} {sim:.4f}")


if __name__ == "__main__":
    main()