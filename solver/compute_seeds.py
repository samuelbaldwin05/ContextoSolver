import json
import numpy as np
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from nltk.corpus import stopwords
import nltk
from tqdm import tqdm

nltk.download('stopwords', quiet=True)

PROCESSED_DIR = Path(__file__).parent.parent / "training" / "data" / "processed"
SOLVER_DIR = Path(__file__).parent
VOCAB_PATH = PROCESSED_DIR / "vocab.json"
VECTORS_PATH = PROCESSED_DIR / "vectors.npy"
ANSWERS_PATH = SOLVER_DIR / "answers.json"
SEEDS_PATH = SOLVER_DIR / "seed_words.json"

N_CLUSTERS = 20
SEEDS_PER_CLUSTER = 3
LAMBDA = 0.3  # 0 = pure centrality, 1 = pure frequency


def load(vocab_path, vectors_path):
    """Load vocab and vectors from disk."""
    with open(vocab_path, encoding="utf-8") as f:
        word_to_idx = json.load(f)
    idx_to_word = {i: w for w, i in word_to_idx.items()}
    vectors = np.load(vectors_path)
    return word_to_idx, idx_to_word, vectors


def normalize(vectors):
    """L2 normalize vectors for cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / (norms + 1e-8)


def filter_vocab(word_to_idx, answers, min_len=3):
    """Remove stop words, short words, and known answers."""
    stop = set(stopwords.words('english'))
    answer_set = set(answers)
    return {
        w: i for w, i in word_to_idx.items()
        if w not in stop
        and w not in answer_set
        and len(w) >= min_len
    }


def score_candidate(freq_rank, centroid_sim, lambda_=LAMBDA):
    """Score a candidate by frequency and centrality tradeoff.
    
    Lower freq_rank = more common = better.
    Higher centroid_sim = more central = better.
    """
    freq_score = 1.0 / (freq_rank + 1)
    return lambda_ * freq_score + (1 - lambda_) * centroid_sim


def compute_seeds(vectors, word_to_idx, filtered_vocab, answers, n_clusters, seeds_per_cluster):
    """Cluster answer vectors, find best non-answer probe word per cluster."""
    with open(ANSWERS_PATH) as f:
        answer_list = json.load(f)

    # get vectors for known answers that are in our vocab
    answer_vecs = []
    for w in answer_list:
        if w in word_to_idx:
            answer_vecs.append(vectors[word_to_idx[w]])

    answer_vecs = np.array(answer_vecs)
    normed_answers = normalize(answer_vecs)

    print(f"Clustering {len(answer_vecs)} answer vectors into {n_clusters} clusters...")
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=512,
        n_init=10,
        verbose=0,
    )
    kmeans.fit(normed_answers)

    # build filtered candidate pool
    filtered_indices = list(filtered_vocab.values())
    filtered_words = list(filtered_vocab.keys())
    filtered_vecs = normalize(vectors[filtered_indices])

    seed_words = []
    seen = set()

    for cluster_idx, centroid in enumerate(tqdm(kmeans.cluster_centers_, desc="Finding seeds")):
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        sims = filtered_vecs @ centroid

        # score each candidate
        scores = []
        for pos, (word, sim) in enumerate(zip(filtered_words, sims)):
            if word in seen:
                continue
            freq_rank = filtered_vocab[word]  # gensim sorts by frequency
            s = score_candidate(freq_rank, float(sim))
            scores.append((word, s))

        # take top seeds_per_cluster
        scores.sort(key=lambda x: -x[1])
        for word, _ in scores[:seeds_per_cluster]:
            if word not in seen:
                seed_words.append(word)
                seen.add(word)

    return seed_words


if __name__ == "__main__":
    word_to_idx, idx_to_word, vectors = load(VOCAB_PATH, VECTORS_PATH)

    with open(ANSWERS_PATH) as f:
        answers = json.load(f)

    filtered_vocab = filter_vocab(word_to_idx, answers)
    print(f"Filtered vocab: {len(filtered_vocab):,} candidates")

    seeds = compute_seeds(vectors, word_to_idx, filtered_vocab, answers, N_CLUSTERS, SEEDS_PER_CLUSTER)

    print(f"\n{len(seeds)} seed words:")
    for i, w in enumerate(seeds):
        print(f"  {i+1:>3}. {w}")

    with open(SEEDS_PATH, "w", encoding="utf-8") as f:
        json.dump(seeds, f, indent=2)
    print(f"\nSaved to {SEEDS_PATH}")