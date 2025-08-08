# app/embedder.py
import os
import torch
from sentence_transformers import SentenceTransformer, util

# Super important for 512MB instances
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_THREADS", "1")

_ST_MODEL = None

def _get_model():
    global _ST_MODEL
    if _ST_MODEL is None:
        # Use a very small model by default (fits in 512MB once loaded)
        name = os.getenv("ST_MODEL_NAME", "sentence-transformers/paraphrase-MiniLM-L3-v2")
        # Force CPU, don’t try to use CUDA
        _ST_MODEL = SentenceTransformer(name, device="cpu")
    return _ST_MODEL

def embed_texts(texts):
    model = _get_model()
    with torch.no_grad():
        return model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)

def semantic_search(query, chunks, chunk_embeddings, top_k=5):
    model = _get_model()
    with torch.no_grad():
        q = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
        scores = util.cos_sim(q, chunk_embeddings)[0]
        top_k = min(top_k, len(chunks))
        top = torch.topk(scores, k=top_k)
        return [chunks[i] for i in top.indices]
