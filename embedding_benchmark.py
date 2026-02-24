import os
import time
import numpy as np
import faiss
import requests
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression

OLLAMA_URL = "http://localhost:11434/api/embeddings"

MODELS = ["nomic-embed-text", "mxbai-embed-large"]

# -------------------------
# CONFIG
# -------------------------
DOCUMENTS = [
    "Employees must submit leave requests 14 days in advance.",
    "All leave requests must be submitted two weeks prior.",
    "Revenue for Q3 increased by 12% year-over-year.",
    "Revenue for Q4 increased by 8% year-over-year.",
    "The compliance department reviews all submissions."
]

QUERIES = [
    ("How many days before leave must be submitted?", 0),
    ("What was Q3 revenue increase?", 2),
    ("Who reviews submissions?", 4)
]


# -------------------------
# OLLAMA EMBEDDING
# -------------------------
def embed_text(model, text):
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": text}
    )
    return response.json()["embedding"]


def embed_corpus(model, texts):
    embeddings = []
    for t in tqdm(texts, desc=f"Embedding with {model}"):
        embeddings.append(embed_text(model, t))
    return np.array(embeddings).astype("float32")


# -------------------------
# FAISS SEARCH
# -------------------------
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index


def evaluate(model):
    print(f"\n===== Benchmarking {model} =====")

    # Embed documents
    doc_embeddings = embed_corpus(model, DOCUMENTS)

    # Build FAISS
    index = build_faiss_index(doc_embeddings)

    recall_at_5 = 0
    mrr_total = 0
    latencies = []

    for query, correct_idx in QUERIES:
        start = time.time()

        q_emb = np.array([embed_text(model, query)]).astype("float32")
        faiss.normalize_L2(q_emb)

        D, I = index.search(q_emb, 5)

        latency = time.time() - start
        latencies.append(latency)

        # Recall@5
        if correct_idx in I[0]:
            recall_at_5 += 1

        # MRR
        if correct_idx in I[0]:
            rank = list(I[0]).index(correct_idx) + 1
            mrr_total += 1.0 / rank

    recall_at_5 /= len(QUERIES)
    mrr = mrr_total / len(QUERIES)

    print(f"Recall@5: {recall_at_5:.2f}")
    print(f"MRR: {mrr:.2f}")
    print(f"Avg latency: {np.mean(latencies):.4f} sec")

    # Intrinsic dimension
    id_estimate = twonn_intrinsic_dimension(doc_embeddings)
    print(f"Intrinsic Dimension (TwoNN): {id_estimate:.2f}")


# -------------------------
# TwoNN ID ESTIMATOR
# -------------------------
def twonn_intrinsic_dimension(X):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    nn = NearestNeighbors(n_neighbors=3, metric="euclidean")
    nn.fit(X)
    distances, _ = nn.kneighbors(X)

    r1 = distances[:, 1]
    r2 = distances[:, 2]

    mu = r2 / r1
    mu_sorted = np.sort(mu)

    F = np.arange(1, len(mu_sorted) + 1) / len(mu_sorted)

    x = np.log(mu_sorted).reshape(-1, 1)
    y = -np.log(1 - F)

    reg = LinearRegression(fit_intercept=False)
    reg.fit(x, y)

    return reg.coef_[0]


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    for model in MODELS:
        evaluate(model)