import numpy as np
from sklearn.neighbors import NearestNeighbors


def levina_bickel_mle(X, k=20, normalize=True):
    """
    Levina-Bickel Maximum Likelihood Estimator (MLE)
    for intrinsic dimensionality.

    Parameters
    ----------
    X : np.ndarray
        Shape (n_samples, n_features)
    k : int
        Number of nearest neighbors (>= 5 recommended)
    normalize : bool
        Whether to L2-normalize embeddings

    Returns
    -------
    float
        Estimated intrinsic dimension
    """

    if normalize:
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

    n_samples = X.shape[0]

    if k >= n_samples:
        raise ValueError("k must be smaller than number of samples")

    # Fit nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(X)
    distances, _ = nbrs.kneighbors(X)

    # Remove self-distance (first column is zero)
    distances = distances[:, 1:]

    # Avoid zero distances
    distances[distances == 0] = 1e-10

    # Levina-Bickel formula
    log_ratios = np.log(distances[:, -1][:, None] / distances[:, :-1])
    local_dims = (k - 1) / np.sum(log_ratios, axis=1)

    # Remove invalid values
    local_dims = local_dims[np.isfinite(local_dims)]

    return np.mean(local_dims)


def levina_bickel_multi_k(X, k_values=(10, 15, 20, 25), normalize=True):
    """
    Multi-k averaged Levina-Bickel estimator.
    More stable than single-k.

    Returns average intrinsic dimension.
    """

    estimates = []

    for k in k_values:
        d = levina_bickel_mle(X, k=k, normalize=normalize)
        estimates.append(d)

    return np.mean(estimates), estimates


# ------------------------------------------------
# Example usage
# ------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)

    # Simulated embedding matrix
    X = np.random.randn(2000, 768)

    # Single k
    id_estimate = levina_bickel_mle(X, k=20)
    print("Intrinsic Dimension (k=20):", round(id_estimate, 2))

    # Multi-k (recommended)
    avg_id, per_k = levina_bickel_multi_k(X)
    print("Intrinsic Dimension (multi-k avg):", round(avg_id, 2))
    print("Per-k estimates:", [round(v, 2) for v in per_k])