import json
import numpy as np
from pathlib import Path
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm

PROCESSED_DIR = Path(__file__).parent / "data" / "processed"
TOKENS_PATH = PROCESSED_DIR / "filtered_tokens.txt"
VOCAB_PATH = PROCESSED_DIR / "vocab.json"
VECTORS_PATH = PROCESSED_DIR / "vectors.npy"

EMBED_DIM = 300
WINDOW_SIZE = 5
NEGATIVE_SAMPLES = 15
MIN_COUNT = 3
WORKERS = 4
EPOCHS = 5


class EpochProgress(CallbackAny2Vec):
    """Callback to track training progress per epoch."""

    def __init__(self, epochs: int):
        self.epochs = epochs
        self.epoch = 0
        self.bar = tqdm(total=epochs, unit=" epochs", desc="Training")

    def on_epoch_end(self, model: Word2Vec) -> None:
        """Update progress bar with current loss at end of each epoch."""
        loss = model.get_latest_training_loss()
        self.bar.set_postfix(loss=f"{loss:.2f}")
        self.bar.update(1)
        self.epoch += 1
        if self.epoch == self.epochs:
            self.bar.close()


def load_articles(path: Path, limit: int = None) -> list[list[str]]:
    """Load tokens.txt into a list of tokenized articles, with optional limit for testing."""
    articles = []
    with open(path, encoding="utf-8") as f, \
         tqdm(unit=" articles", desc="Loading articles") as bar:
        for line in f:
            tokens = line.strip().split()
            if tokens:
                articles.append(tokens)
                bar.update(1)
            if limit and len(articles) >= limit:
                break
    return articles


def train(articles: list[list[str]], epochs: int) -> Word2Vec:
    """Train a skip-gram Word2Vec model on the given articles."""
    print(f"Training on {len(articles):,} articles...")
    model = Word2Vec(
        sentences=articles,
        vector_size=EMBED_DIM,
        window=WINDOW_SIZE,
        min_count=MIN_COUNT,
        sg=1,
        negative=NEGATIVE_SAMPLES,
        workers=WORKERS,
        epochs=epochs,
        compute_loss=True,
        callbacks=[EpochProgress(epochs)],
    )
    return model


def save_outputs(model: Word2Vec, vocab_path: Path, vectors_path: Path) -> None:
    """Save trained vectors as .npy and vocab as word-to-index JSON."""
    words = list(model.wv.index_to_key)
    vectors = model.wv.vectors
    word_to_idx = {w: i for i, w in enumerate(words)}

    np.save(vectors_path, vectors)
    print(f"Vectors saved to {vectors_path} — shape: {vectors.shape}")

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(word_to_idx, f)
    print(f"Vocab saved to {vocab_path} — {len(word_to_idx):,} words")


def main(test_mode: bool = False) -> None:
    """Load articles, train Word2Vec, and save outputs."""
    limit = 5000 if test_mode else None
    epochs = 10 if test_mode else EPOCHS

    articles = load_articles(TOKENS_PATH, limit=limit)
    print(f"Loaded {len(articles):,} articles")

    model = train(articles, epochs)
    save_outputs(model, VOCAB_PATH, VECTORS_PATH)


if __name__ == "__main__":
    import sys
    test_mode = "--test" in sys.argv
    main(test_mode=test_mode)