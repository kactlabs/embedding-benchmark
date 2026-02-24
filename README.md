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
### py estimate_intrinsic_dim.py
![1771933466528](image/README/1771933466528.png)

### py embedding_benchmark.py
![1771933513453](image/README/1771933513453.png)

### py levina_bickel_id.py
![1771933644855](image/README/1771933644855.png)

### py mle_intrinsic_dim.py
![1771933673302](image/README/1771933673302.png)

### py twonn_intrinsic_dim.py
![1771933850450](image/README/1771933850450.png)

![1771930396117](image/README/1771930396117.png)![1771930397350](image/README/1771930397350.png)