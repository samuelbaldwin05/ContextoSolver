# ContextoSolver
Custom Word2Vec implementation to help solve context game.


## Data

The training pipeline uses the [Simple English Wikipedia](https://simple.wikipedia.org) dump as its corpus.

### Option 1: Automated download

Run the download script from the project root:
```bash
python training/download.py
```

This fetches the latest dump from Wikimedia and saves it to `training/data/raw/`.

### Option 2: Manual download

1. Go to https://dumps.wikimedia.org/simplewiki/latest/
2. Download `simplewiki-latest-pages-articles.xml.bz2`
3. Place it in `training/data/raw/`

Either way, the rest of the pipeline expects the `.xml.bz2` file to be present in `training/data/raw/` before running `preprocess.py`.