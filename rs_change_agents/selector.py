from __future__ import annotations
import numpy as np

class CaptionSelector:
    """
    Embedding-based candidate filtering.
    The paper states embedding/cosine-based filtering but does not publish an exact selection formula.
    This implementation uses a reproducible MMR-like rule: candidate-to-centroid representativeness
    balanced against redundancy with already selected captions.
    """
    def __init__(self, model_name: str, mmr_lambda: float = 0.70, device: str | None = None):
        self.model_name = model_name
        self.mmr_lambda = float(mmr_lambda)
        self.device = device
        self.model = None
        if model_name and model_name.lower() != "none":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, device=device)

    def select(self, captions: list[str], top_k: int = 1) -> tuple[list[str], dict]:
        # exact string deduplication first
        unique = list(dict.fromkeys(c.strip() for c in captions if c and c.strip()))
        if not unique:
            return [], {"indices": [], "scores": []}
        top_k = min(top_k, len(unique))
        if self.model is None or len(unique) == 1:
            return unique[:top_k], {"indices": list(range(top_k)), "scores": [1.0] * top_k}

        emb = self.model.encode(unique, normalize_embeddings=True, convert_to_numpy=True)
        centroid = emb.mean(axis=0)
        centroid /= (np.linalg.norm(centroid) + 1e-12)
        relevance = emb @ centroid
        sim = emb @ emb.T

        selected = []
        mmr_scores = []
        remaining = set(range(len(unique)))
        while remaining and len(selected) < top_k:
            best_i, best_s = None, -1e9
            for i in remaining:
                redundancy = max((sim[i, j] for j in selected), default=0.0)
                score = self.mmr_lambda * relevance[i] - (1 - self.mmr_lambda) * redundancy
                if score > best_s:
                    best_i, best_s = i, float(score)
            selected.append(best_i)
            mmr_scores.append(best_s)
            remaining.remove(best_i)
        return [unique[i] for i in selected], {"indices": selected, "scores": mmr_scores}
