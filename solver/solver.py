import json
import random
import numpy as np
from pathlib import Path
from strategy import next_candidate_idx
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords', quiet=True)

PROCESSED_DIR = Path(__file__).parent.parent / "training" / "data" / "processed"
SOLVER_DIR = Path(__file__).parent
VOCAB_PATH = PROCESSED_DIR / "vocab.json"
VECTORS_PATH = PROCESSED_DIR / "vectors.npy"
SEEDS_PATH = SOLVER_DIR / "seed_words.json"


class Solver:
    """Orchestrates the Contexto guessing strategy.

    Maintains guess history and ranks, switches from seed exploration
    to equation-driven exploitation as ranks improve.
    """

    def __init__(self):
        self.word_to_idx, self.idx_to_word, self.vectors, self.normed = self._load()
        self.seeds = self._load_seeds()
        self.guesses: list[str] = []
        self.ranks: list[int] = []
        self.guessed_indices: set[int] = set()
        self._remaining_seeds = random.sample(self.seeds, len(self.seeds))

    def _load(self, top_n: int = 20000):
        with open(VOCAB_PATH) as f:
            word_to_idx = json.load(f)
        
        # load frequency-sorted vocab — gensim sorts by frequency
        # so the first N words are the most common
        word_to_idx = {w: i for i, w in enumerate(list(word_to_idx.keys())[:top_n])}
        idx_to_word = {i: w for w, i in word_to_idx.items()}
        
        vectors = np.load(VECTORS_PATH)[:top_n]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normed = vectors / (norms + 1e-8)
        return word_to_idx, idx_to_word, vectors, normed

    def _load_seeds(self) -> list[str]:
        """Load precomputed seed words."""
        if not SEEDS_PATH.exists():
            raise FileNotFoundError(
                f"Seed words not found at {SEEDS_PATH}. Run compute_seeds.py first."
            )
        with open(SEEDS_PATH, encoding="utf-8") as f:
            return json.load(f)

    def record(self, word: str, rank: int) -> None:
        """Record a guess and its rank from Contexto."""
        self.guesses.append(word)
        self.ranks.append(rank)
        if word in self.word_to_idx:
            self.guessed_indices.add(self.word_to_idx[word])

    def next_guess(self) -> str:
        """Return the next word to guess.

        Uses a random seed word if no guesses yet or seeds remain,
        otherwise uses the strategy equation.
        """
        if not self.guesses:
            return self._remaining_seeds.pop()

        if self._remaining_seeds:
            candidate = self._remaining_seeds.pop()
            if candidate not in self.guesses:
                return candidate

        idx = next_candidate_idx(
            self.vectors,
            self.normed,
            self.guessed_indices,
            self.ranks,
        )
        return self.idx_to_word[idx]

    @property
    def best_guess(self) -> tuple[str, int] | None:
        """Return the current best (word, rank) pair."""
        if not self.guesses:
            return None
        best_idx = int(np.argmin(self.ranks))
        return self.guesses[best_idx], self.ranks[best_idx]

    def reset(self) -> None:
        """Reset solver state for a new game."""
        self.guesses = []
        self.ranks = []
        self.guessed_indices = set()
        self._remaining_seeds = random.sample(self.seeds, len(self.seeds))

    def state(self) -> list[dict]:
        """Return current guess history as a list of dicts for display."""
        return [
            {"word": w, "rank": r}
            for w, r in sorted(zip(self.guesses, self.ranks), key=lambda x: x[1])
        ]