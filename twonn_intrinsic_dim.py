import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression


def twonn_intrinsic_dimension(X):
    """
    TwoNN intrinsic dimension estimator.

    Parameters
    ----------
    X : np.ndarray
        Shape (n_samples, n_features)
        Embedding matrix

    Returns
    -------
    float
        Estimated intrinsic dimension
    """

    n = X.shape[0]

    # Normalize (important for embeddings)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    # Find 2 nearest neighbors (exclude self)
    nn = NearestNeighbors(n_neighbors=3, metric="euclidean")
    nn.fit(X)
    distances, _ = nn.kneighbors(X)

    # distances[:,0] is 0 (self)
    r1 = distances[:, 1]
    r2 = distances[:, 2]

    # Avoid division by zero
    valid = r1 > 0
    mu = r2[valid] / r1[valid]

    # Sort mu
    mu_sorted = np.sort(mu)

    # Empirical CDF
    F = np.arange(1, len(mu_sorted) + 1) / len(mu_sorted)
    F = np.clip(F, 1e-10, 1 - 1e-10)

    # Linear regression:
    # y = -log(1 - F)
    # x = log(mu)
    x = np.log(mu_sorted).reshape(-1, 1)
    y = -np.log(1 - F)

    reg = LinearRegression(fit_intercept=False)
    reg.fit(x, y)

    d_estimate = reg.coef_[0]

    return d_estimate


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    np.random.seed(42)

    # Simulated embedding matrix
    X = np.random.randn(2000, 768)

    id_estimate = twonn_intrinsic_dimension(X)

    print("Estimated Intrinsic Dimension (TwoNN):", round(id_estimate, 2))