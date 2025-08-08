from sentence_transformers import SentenceTransformer, util
import torch

# Load a lightweight embedding model
_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    """
    Embed a list of text chunks into vectors.
    """
    return _model.encode(texts, convert_to_tensor=True)

def semantic_search(query, chunks, chunk_embeddings, top_k=5):
    """
    Given a query, returns the top_k most similar chunks.
    """
    query_embedding = _model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, chunk_embeddings)[0]
    top_results = torch.topk(scores, k=top_k)
    return [chunks[idx] for idx in top_results.indices]
