import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA


def twonn_intrinsic_dimension(X):
    """
    TwoNN intrinsic dimension estimator.
    
    Uses ratio of distances to 1st and 2nd nearest neighbors.
    """
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    nn = NearestNeighbors(n_neighbors=3, metric="euclidean")
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    
    r1 = distances[:, 1]
    r2 = distances[:, 2]
    valid = r1 > 0
    mu = r2[valid] / r1[valid]
    mu_sorted = np.sort(mu)
    
    F = np.arange(1, len(mu_sorted) + 1) / len(mu_sorted)
    F = np.clip(F, 1e-10, 1 - 1e-10)
    
    x = np.log(mu_sorted).reshape(-1, 1)
    y = -np.log(1 - F)
    reg = LinearRegression(fit_intercept=False)
    reg.fit(x, y)
    
    return reg.coef_[0]


def levina_bickel_mle(X, k=20, normalize=True):
    """
    Levina-Bickel Maximum Likelihood Estimator.
    """
    if normalize:
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
    
    n_samples = X.shape[0]
    if k >= n_samples:
        raise ValueError("k must be smaller than number of samples")
    
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(X)
    distances, _ = nbrs.kneighbors(X)
    distances = distances[:, 1:]
    distances[distances == 0] = 1e-10
    
    log_ratios = np.log(distances[:, -1][:, None] / distances[:, :-1])
    local_dims = (k - 1) / np.sum(log_ratios, axis=1)
    local_dims = local_dims[np.isfinite(local_dims)]
    
    return np.mean(local_dims)


def levina_bickel_multi_k(X, k_values=(10, 15, 20, 25), normalize=True):
    """
    Multi-k averaged Levina-Bickel estimator.
    """
    n_samples = X.shape[0]
    # Filter k_values to be valid for this sample size
    valid_k_values = [k for k in k_values if k < n_samples]
    if not valid_k_values:
        valid_k_values = [n_samples - 1]
    
    estimates = [levina_bickel_mle(X, k=k, normalize=normalize) for k in valid_k_values]
    return np.mean(estimates), estimates


def mle_intrinsic_dimension(X, k=20):
    """
    MLE intrinsic dimensionality estimator (Levina-Bickel variant).
    """
    if k >= len(X):
        raise ValueError("k must be smaller than number of samples")
    
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X)
    distances, _ = nbrs.kneighbors(X)
    distances = distances[:, 1:]
    
    mle_values = []
    for i in range(len(X)):
        dists = distances[i]
        T_k = dists[-1]
        if T_k == 0:
            continue
        logs = np.log(T_k / dists[:-1])
        mle = (k - 1) / np.sum(logs)
        mle_values.append(mle)
    
    return np.mean(mle_values)


def pca_intrinsic_dimension(X, variance_threshold=0.95):
    """
    PCA-based intrinsic dimension estimator.
    """
    pca = PCA()
    pca.fit(X)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    return np.argmax(cumulative_variance >= variance_threshold) + 1


def combined_intrinsic_dimension(X, k=20, use_pca=True, normalize=True):
    """
    Combined intrinsic dimension estimator using multiple methods.
    
    Returns a dictionary with estimates from all algorithms and a weighted average.
    """
    # TwoNN
    id_twonn = twonn_intrinsic_dimension(X)
    
    # Levina-Bickel single k
    id_lb_single = levina_bickel_mle(X, k=k, normalize=normalize)
    
    # Levina-Bickel multi-k
    id_lb_multi, lb_per_k = levina_bickel_multi_k(X, k_values=(10, 15, 20, 25), normalize=normalize)
    
    # MLE
    id_mle = mle_intrinsic_dimension(X, k=k)
    
    # PCA (optional)
    if use_pca:
        id_pca = pca_intrinsic_dimension(X)
    
    # Weighted average (excluding PCA which has different interpretation)
    # Give more weight to multi-k which is more stable
    weights = {'twonn': 1.0, 'lb_single': 1.0, 'lb_multi': 2.0, 'mle': 1.0}
    total_weight = sum(weights.values())
    
    weighted_sum = (
        weights['twonn'] * id_twonn +
        weights['lb_single'] * id_lb_single +
        weights['lb_multi'] * id_lb_multi +
        weights['mle'] * id_mle
    )
    
    id_combined = weighted_sum / total_weight
    
    results = {
        'twonn': id_twonn,
        'levina_bickel_single_k': id_lb_single,
        'levina_bickel_multi_k': id_lb_multi,
        'levina_bickel_per_k': lb_per_k,
        'mle': id_mle,
        'combined': id_combined
    }
    
    if use_pca:
        results['pca'] = id_pca
    
    return results


def print_intrinsic_dimension_report(results, k=20):
    """
    Print a formatted report of intrinsic dimension estimates.
    """
    print("\n" + "=" * 60)
    print("INTRINSIC DIMENSION ESTIMATION REPORT")
    print("=" * 60)
    print(f"\nParameters: k={k}")
    print("-" * 60)
    
    print(f"\nSingle-Method Estimates:")
    print(f"  TwoNN:                  {results['twonn']:8.2f}")
    print(f"  Levina-Bickel (k={k:2d}):       {results['levina_bickel_single_k']:8.2f}")
    print(f"  MLE (k={k:2d}):               {results['mle']:8.2f}")
    
    if 'pca' in results:
        print(f"  PCA (95% variance):     {results['pca']:8d}")
    
    print(f"\nMulti-k Levina-Bickel:")
    for i, k_val in enumerate([10, 15, 20, 25]):
        if i < len(results['levina_bickel_per_k']):
            print(f"  k={k_val:2d}: {results['levina_bickel_per_k'][i]:8.2f}")
        else:
            print(f"  k={k_val:2d}: N/A (too large for sample size)")
    print(f"  Average:                {results['levina_bickel_multi_k']:8.2f}")
    
    print("-" * 60)
    print(f"\nCombined Score (weighted average):")
    print(f"  Combined ID:            {results['combined']:8.2f}")
    print("=" * 60 + "\n")


# -------------------------
# PDF Loading & Embedding
# -------------------------
import os
from pypdf import PdfReader
from tqdm import tqdm
import requests
import random

OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"


def load_pdfs(folder_path, max_pdfs=500):
    """Load PDF files from a folder."""
    texts = []
    files = os.listdir(folder_path)[:max_pdfs]
    for file in tqdm(files, desc="Loading PDFs"):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(folder_path, file))
            text = ""
            for page in reader.pages:
                if page.extract_text():
                    text += page.extract_text()
            texts.append(text)
    return texts


def chunk_text(text, chunk_size=800, overlap=100):
    """Simple text chunking with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def embed_text(text):
    """Embed text using Ollama."""
    response = requests.post(
        OLLAMA_URL,
        json={"model": EMBED_MODEL, "prompt": text}
    )
    return response.json()["embedding"]


def build_embedding_matrix(texts, max_chunks=2000):
    """Build embedding matrix from texts."""
    all_chunks = []
    for text in texts:
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
    
    sampled = random.sample(all_chunks, min(len(all_chunks), max_chunks))
    
    embeddings = []
    for chunk in tqdm(sampled, desc="Embedding chunks"):
        emb = embed_text(chunk)
        embeddings.append(emb)
    
    return np.array(embeddings)


def intrinsic_dim_from_pdfs(folder_path, max_pdfs=500, max_chunks=2000, k=None, use_pca=True):
    """
    Full pipeline: Load PDFs -> Embed -> Calculate intrinsic dimension.
    """
    print(f"\nLoading PDFs from: {folder_path}")
    texts = load_pdfs(folder_path, max_pdfs)
    print(f"Loaded {len(texts)} PDFs")
    
    print("\nBuilding embedding matrix...")
    embeddings = build_embedding_matrix(texts, max_chunks)
    print(f"Embedding matrix shape: {embeddings.shape}")
    
    # Auto-adjust k based on sample size
    n_samples = embeddings.shape[0]
    if k is None:
        k = min(20, n_samples - 1)
    elif k >= n_samples:
        print(f"\nWarning: k={k} is too large for {n_samples} samples. Adjusting to k={n_samples-1}")
        k = n_samples - 1
    
    print(f"\nCalculating intrinsic dimension (k={k})...")
    results = combined_intrinsic_dimension(embeddings, k=k, use_pca=use_pca)
    print_intrinsic_dimension_report(results, k=k)
    
    return results, embeddings


if __name__ == "__main__":
    folder = "/Users/csp/datasets/random_resume_minimal"
    results, embeddings = intrinsic_dim_from_pdfs(folder, max_pdfs=500, max_chunks=2000)
    
    print("\nFinal Combined Intrinsic Dimension Score:")
    print(f"  Combined ID: {results['combined']:.2f}")
