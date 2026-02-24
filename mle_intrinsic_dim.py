import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def mle_intrinsic_dimension(X, k=20):
    """
    Levina–Bickel MLE intrinsic dimensionality estimator.

    Parameters:
    -----------
    X : np.ndarray
        Shape (n_samples, n_features)
        Embedding matrix
    k : int
        Number of nearest neighbors

    Returns:
    --------
    float
        Estimated intrinsic dimension
    """

    if k >= len(X):
        raise ValueError("k must be smaller than number of samples")

    # Fit nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X)
    distances, _ = nbrs.kneighbors(X)

    # Remove self-distance (0)
    distances = distances[:, 1:]

    mle_values = []

    for i in range(len(X)):
        dists = distances[i]
        T_k = dists[-1]

        # Avoid log(0)
        if T_k == 0:
            continue

        logs = np.log(T_k / dists[:-1])
        mle = (k - 1) / np.sum(logs)

        mle_values.append(mle)

    return np.mean(mle_values)


if __name__ == "__main__":
    # Example usage with random data
    np.random.seed(42)
    X = np.random.rand(1000, 768)

    id_estimate = mle_intrinsic_dimension(X, k=20)

    print("Estimated Intrinsic Dimension:", round(id_estimate, 2))