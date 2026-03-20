import numpy as np

# controls sharpness of pivot from exploration to exploitation
# higher k = sharper pivot when good ranks are found
K = 5.0


def rank_to_weight(rank: int, r_max: float, r_mean: float) -> float:
    """Convert a rank to a signed weight using inverse rank scaled by tanh.

    Positive for guesses better than average, negative for worse than average.
    No hard thresholds — smooth and continuous.
    """
    return (1.0 / rank) * np.tanh((r_max - rank) / (r_mean + 1e-8))


def compute_weights(ranks: list[int]) -> np.ndarray:
    """Compute signed weights for all guesses based on their ranks."""
    r_max = float(max(ranks))
    r_mean = float(np.mean(ranks))
    return np.array([rank_to_weight(r, r_max, r_mean) for r in ranks])


def compute_centroid(vecs: np.ndarray, ranks: list[int]) -> np.ndarray:
    """Compute weighted centroid of guessed vectors.

    Good ranks pull the centroid toward them, bad ranks push it away.
    Returns a normalized vector representing the estimated target direction.
    """
    weights = compute_weights(ranks)
    centroid = weights @ vecs
    norm = np.linalg.norm(centroid)
    return centroid / (norm + 1e-8)


def compute_t(ranks: list[int]) -> float:
    """Compute exploration-exploitation parameter t from current ranks.

    t is the mean inverse rank — grows naturally as better ranks accumulate.
    No guess count needed, purely rank-driven.
    """
    return float(np.mean([1.0 / r for r in ranks]))


def compute_alpha(t: float, k: float = K) -> float:
    """Map t to alpha in [0, 1] via sigmoid.

    alpha close to 0 = explore (maximize distance from guesses)
    alpha close to 1 = exploit (maximize similarity to centroid)
    """
    return float(1.0 / (1.0 + np.exp(-k * t)))


def score_candidate(
    candidate_vec: np.ndarray,
    centroid: np.ndarray,
    guessed_vecs: np.ndarray,
    alpha: float,
) -> float:
    """Score a candidate word vector given current search state.

    Combines exploitation (similarity to centroid) and exploration
    (dissimilarity to all previously guessed words).
    """
    exploit = float(np.dot(candidate_vec, centroid))
    novelty = -float(np.max(guessed_vecs @ candidate_vec))
    return alpha * exploit + (1 - alpha) * novelty


def score_all_candidates(
    vectors: np.ndarray,
    guessed_indices: set[int],
    centroid: np.ndarray,
    guessed_vecs: np.ndarray,
    alpha: float,
    normed_vectors: np.ndarray,
) -> np.ndarray:
    """Score all candidate vectors at once using vectorized operations.

    Returns an array of scores, with -inf for already-guessed words.
    """
    exploit = normed_vectors @ centroid                          # (vocab_size,)
    novelty = -np.max(normed_vectors @ guessed_vecs.T, axis=1)  # (vocab_size,)
    scores = alpha * exploit + (1 - alpha) * novelty

    # mask out already guessed words
    for idx in guessed_indices:
        scores[idx] = -np.inf

    return scores


def next_candidate_idx(
    vectors: np.ndarray,
    normed_vectors: np.ndarray,
    guessed_indices: set[int],
    ranks: list[int],
) -> int:
    """Given current guesses and ranks, return the index of the next best candidate."""
    guessed_vecs = normed_vectors[list(guessed_indices)]
    centroid = compute_centroid(guessed_vecs, ranks)
    t = compute_t(ranks)
    alpha = compute_alpha(t)
    scores = score_all_candidates(
        vectors, guessed_indices, centroid, guessed_vecs, alpha, normed_vectors
    )
    return int(np.argmax(scores))