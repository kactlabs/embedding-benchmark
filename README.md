# embedding-benchmark

A benchmarking script for evaluating embedding models using Ollama, FAISS, and scikit-learn.

## Features

- Embed documents and queries using Ollama models (e.g., nomic-embed-text, mxbai-embed-large)
- FAISS-based similarity search with Recall@5 and MRR metrics
- Intrinsic dimension estimation using TwoNN algorithm

## Requirements

See `requirements.txt` for dependencies.

## Usage

1. Start Ollama locally:
```bash
ollama serve
```

2. Run the benchmark:
```bash
python embedding_benchmark.py
```

## Models

Currently supports benchmarking multiple embedding models. Add or modify models in the `MODELS` list.


## Screenshots
![1771930396117](image/README/1771930396117.png)![1771930397350](image/README/1771930397350.png)