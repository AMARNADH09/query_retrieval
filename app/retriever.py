
import faiss, numpy as np
from typing import List, Tuple

def build_faiss(embs: np.ndarray):
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine if normalized
    index.add(embs.astype('float32'))
    return index

def search(index, qvec, top_k=6) -> Tuple[List[int], List[float]]:
    D, I = index.search(qvec.reshape(1, -1).astype('float32'), top_k)
    return I[0].tolist(), D[0].tolist()

def mmr(query_vec, doc_vecs, cand_idx, lambda_mult=0.7, k=5):
    selected = []
    cand = cand_idx[:]
    while cand and len(selected) < k:
        best, best_score = None, -1e9
        for i in cand:
            rel = float(np.dot(query_vec, doc_vecs[i]))
            div = 0.0 if not selected else max(float(np.dot(doc_vecs[i], doc_vecs[j])) for j in selected)
            score = lambda_mult*rel - (1.0 - lambda_mult)*div
            if score > best_score:
                best, best_score = i, score
        selected.append(best)
        cand.remove(best)
    return selected
