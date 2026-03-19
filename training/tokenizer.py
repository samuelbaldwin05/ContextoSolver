import re
import spacy
from pathlib import Path
from tqdm import tqdm

PROCESSED_DIR = Path(__file__).parent / "data" / "processed"
CORPUS_PATH = PROCESSED_DIR / "corpus.txt"
TOKENS_PATH = PROCESSED_DIR / "tokens.txt"

ONLY_LETTERS = re.compile(r'^[a-z]+$')
WRITE_BUFFER_SIZE = 1000


def load_model() -> spacy.Language:
    """Load the spaCy English model with unused pipeline components disabled."""
    try:
        return spacy.load("en_core_web_sm", disable=["parser", "ner", "tok2vec", "tagger"])
    except OSError:
        raise OSError(
            "spaCy model not found. Run: python -m spacy download en_core_web_sm"
        )


def process_token(token: spacy.tokens.Token) -> str | None:
    """Filter and lemmatize a single token using rule-based lemmatization.

    Returns the lemmatized token if it contains only ASCII letters, else None.
    Punctuation and non-alphabetic tokens are discarded.
    """
    if token.is_punct:
        return None
    lemma = token.lemma_.lower()
    if not ONLY_LETTERS.match(lemma):
        return None
    return lemma


def process_doc(doc: spacy.tokens.Doc) -> list[str]:
    """Filter, lemmatize, and validate all tokens in a doc, returning a list of tokens."""
    return [t for token in doc if (t := process_token(token)) is not None]


def tokenize(corpus_path: Path, tokens_path: Path) -> None:
    """Read corpus.txt, batch process with spaCy pipe, and write tokens to tokens.txt."""
    nlp = load_model()

    with open(corpus_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    buffer = []
    total_tokens = 0

    with open(tokens_path, "w", encoding="utf-8") as f_out, \
         tqdm(total=len(lines), unit=" articles", desc="Tokenizing") as bar:

        for doc in nlp.pipe(lines, batch_size=5000):
            tokens = process_doc(doc)
            if tokens:
                buffer.append(" ".join(tokens))
                total_tokens += len(tokens)

            if len(buffer) >= WRITE_BUFFER_SIZE:
                f_out.write("\n".join(buffer) + "\n")
                buffer.clear()

            bar.set_postfix(tokens=f"{total_tokens:,}")
            bar.update(1)

        if buffer:
            f_out.write("\n".join(buffer) + "\n")

    print(f"Done. Tokens written to {tokens_path}")


if __name__ == "__main__":
    if TOKENS_PATH.exists():
        print(f"tokens.txt already exists at {TOKENS_PATH}, skipping.")
    else:
        tokenize(CORPUS_PATH, TOKENS_PATH)