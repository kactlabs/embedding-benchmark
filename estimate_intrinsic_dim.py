import os
import numpy as np
from sklearn.decomposition import PCA
from pypdf import PdfReader
from tqdm import tqdm
import requests
import random

OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

# -------------------------
# 1️⃣ Load PDFs
# -------------------------
def load_pdfs(folder_path, max_pdfs=500):
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


# -------------------------
# 2️⃣ Simple Chunking
# -------------------------
def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


# -------------------------
# 3️⃣ Ollama Embedding
# -------------------------
def embed_text(text):
    response = requests.post(
        OLLAMA_URL,
        json={"model": EMBED_MODEL, "prompt": text}
    )
    return response.json()["embedding"]


# -------------------------
# 4️⃣ Build Embedding Matrix
# -------------------------
def build_embedding_matrix(texts, max_chunks=2000):
    all_chunks = []
    for text in texts:
        chunks = chunk_text(text)
        all_chunks.extend(chunks)

    # Sample chunks to limit computation
    sampled = random.sample(all_chunks, min(len(all_chunks), max_chunks))

    embeddings = []
    for chunk in tqdm(sampled, desc="Embedding chunks"):
        emb = embed_text(chunk)
        embeddings.append(emb)

    return np.array(embeddings)


# -------------------------
# 5️⃣ PCA Intrinsic Dimension
# -------------------------
def estimate_intrinsic_dim(embeddings, variance_threshold=0.95):
    pca = PCA()
    pca.fit(embeddings)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    intrinsic_dim = np.argmax(cumulative_variance >= variance_threshold) + 1

    return intrinsic_dim, cumulative_variance


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    folder = "your_pdf_folder"

    texts = load_pdfs(folder)
    embeddings = build_embedding_matrix(texts)

    intrinsic_dim, cumulative_variance = estimate_intrinsic_dim(embeddings)

    print("\nEstimated Intrinsic Dimension (95% variance):", intrinsic_dim)