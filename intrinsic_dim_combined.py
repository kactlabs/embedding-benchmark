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
import time
from pypdf import PdfReader
from tqdm import tqdm
import requests
import random

OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

# Model-specific context limits (in characters)
# Ollama models typically have 8192 token context, ~4 chars per token
MODEL_CONTEXT_LIMITS = {
    "nomic-embed-text": 3000,   # ~750 tokens
    "mxbai-embed-large": 2000,  # ~500 tokens
}


def load_pdfs(folder_path, max_pdfs=500):
    """Load PDF files from a folder."""
    texts = []
    files = os.listdir(folder_path)[:max_pdfs]
    for file in tqdm(files, desc="Loading PDFs"):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(folder_path, file))
            text = ""
            pages_count = 0
            for page in reader.pages:
                if page.extract_text():
                    text += page.extract_text()
                    pages_count += 1
            word_count = len(text.split()) if text else 0
            print(f"  {file}: {pages_count} pages, {word_count} words")
            texts.append(text)
    return texts


def chunk_text(text, chunk_size=800, overlap=100, model=None):
    """Simple text chunking with overlap."""
    # Adjust chunk size based on model context limit
    if model:
        max_len = MODEL_CONTEXT_LIMITS.get(model, 3000)
        if chunk_size > max_len:
            chunk_size = max_len
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def embed_text(text, model=EMBED_MODEL, max_retries=3):
    """Embed text using Ollama."""
    # Truncate text if it exceeds model's context limit
    max_len = MODEL_CONTEXT_LIMITS.get(model, 3000)
    if len(text) > max_len:
        text = text[:max_len]
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={"model": model, "prompt": text},
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            
            # Try different possible response keys
            if "embedding" in data:
                return data["embedding"]
            elif "embeddings" in data:
                return data["embeddings"][0]
            else:
                print(f"Unexpected response from {model}: {data.keys()}")
                raise KeyError(f"Could not find embedding in response. Keys: {data.keys()}")
        except (requests.exceptions.RequestException, KeyError) as e:
            if attempt == max_retries - 1:
                print(f"\nFailed to embed chunk after {max_retries} attempts: {e}")
                return None
            print(f"\nAttempt {attempt + 1} failed, retrying...")
            time.sleep(1)
    
    return None


def build_embedding_matrix(texts, max_chunks=2000, model=EMBED_MODEL):
    """Build embedding matrix from texts."""
    all_chunks = []
    for text in texts:
        chunks = chunk_text(text, model=model)
        all_chunks.extend(chunks)
    
    sampled = random.sample(all_chunks, min(len(all_chunks), max_chunks))
    
    embeddings = []
    skipped = 0
    for chunk in tqdm(sampled, desc=f"Embedding chunks ({model})"):
        emb = embed_text(chunk, model=model)
        if emb is not None:
            embeddings.append(emb)
        else:
            skipped += 1
    
    if skipped > 0:
        print(f"\nSkipped {skipped} chunks due to embedding errors")
    
    return np.array(embeddings)


def intrinsic_dim_from_pdfs(folder_path, max_pdfs=500, max_chunks=2000, k=None, use_pca=True, model=EMBED_MODEL):
    """
    Full pipeline: Load PDFs -> Embed -> Calculate intrinsic dimension.
    """
    print(f"\nLoading PDFs from: {folder_path}")
    texts = load_pdfs(folder_path, max_pdfs)
    print(f"Loaded {len(texts)} PDFs")
    
    print("\nBuilding embedding matrix...")
    embeddings = build_embedding_matrix(texts, max_chunks, model=model)
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
    folder = "/Users/csp/datasets/5_pdf"
    
    # Test multiple embedding models for comparison
    # Add more models as needed
    MODELS = [
        ("nomic-embed-text", "768-dim"),
        ("mxbai-embed-large", "1024-dim"),
    ]
    
    print("=" * 60)
    print("EMBEDDING DIMENSIONALITY COMPARISON")
    print("=" * 60)
    print(f"\nFolder: {folder}")
    print(f"Models: {', '.join([m[0] for m in MODELS])}")
    
    all_results = {}
    
    for model_name, dim_label in MODELS:
        print(f"\n{'='*60}")
        print(f"Testing: {model_name} ({dim_label})")
        print(f"{'='*60}")
        
        results, embeddings = intrinsic_dim_from_pdfs(folder, max_pdfs=500, max_chunks=500, model=model_name)
        all_results[model_name] = {
            'dim_label': dim_label,
            'id_combined': results['combined'],
            'id_twonn': results['twonn'],
            'id_lb_multi': results['levina_bickel_multi_k'],
            'id_mle': results['mle'],
            'shape': embeddings.shape
        }
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: COMPARISON ACROSS EMBEDDING MODELS")
    print("=" * 70)
    print(f"\n{'Model':<25} {'Dim':<10} {'ID (Combined)':<15} {'ID (TwoNN)':<12} {'ID (Multi-k)':<12}")
    print("-" * 70)
    
    for model_name, data in all_results.items():
        print(f"{model_name:<25} {data['dim_label']:<10} {data['id_combined']:<15.2f} {data['id_twonn']:<12.2f} {data['id_lb_multi']:<12.2f}")
    
    print("-" * 70)
    
    # Recommendation
    print("\nRECOMMENDATION:")
    print("-" * 50)
    
    best_model = min(all_results.items(), key=lambda x: x[1]['id_combined'])
    print(f"Lowest ID score: {best_model[0]} (ID = {best_model[1]['id_combined']:.2f})")
    print(f"\nInterpretation:")
    print(f"  - Lower ID = data lies on simpler manifold")
    print(f"  - If ID << embedding_dim, smaller model may work")
    print(f"  - If ID ≈ embedding_dim, current size is well utilized")
    
    # Check if smaller model would suffice
    for model_name, data in all_results.items():
        dim = int(data['dim_label'].replace('-dim', ''))
        id_score = data['id_combined']
        ratio = id_score / dim
        if ratio < 0.05:
            print(f"\n  {model_name}: ID ({id_score:.1f}) is only {ratio*100:.1f}% of dim ({dim})")
            print(f"    → Consider smaller embedding model")
        elif ratio > 0.7:
            print(f"\n  {model_name}: ID ({id_score:.1f}) is {ratio*100:.1f}% of dim ({dim})")
            print(f"    → Current dim is well utilized, larger dim may help")
