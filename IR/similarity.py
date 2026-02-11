import math

def cosine_similarity(vec1: dict, vec2: dict) -> float:
    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0

    for term in vec1:
        w1 = vec1[term]
        w2 = vec2.get(term, 0)

        dot += w1 * w2
        norm1 += w1 * w1
        norm2 += w2 * w2

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (math.sqrt(norm1) * math.sqrt(norm2))

def rank_documents(query_vec: dict, doc_vectors: dict, scheme: str, top_k=10):
    scores = []

    for url, vectors in doc_vectors.items():
        sim = cosine_similarity(vectors[scheme], query_vec[scheme])
        scores.append((url, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]
