from __future__ import annotations
import numpy as np
from .prompts import judge_prompt
from .utils import extract_json

class LLMJudge:
    def __init__(self, backend, max_new_tokens: int = 220):
        self.backend = backend
        self.max_new_tokens = max_new_tokens

    def score(self, task_type: str, output_text: str, reference: str | None = None) -> dict:
        raw = self.backend.generate(
            judge_prompt(task_type, output_text, reference), image_paths=None,
            temperature=0.0, max_new_tokens=self.max_new_tokens
        )
        parsed = extract_json(raw)
        parsed["raw"] = raw
        return parsed

class EmbeddingEvaluator:
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def pair_similarity(self, refs: list[str], hyps: list[str]) -> dict:
        if len(refs) != len(hyps):
            raise ValueError("refs and hyps must have same length")
        e1 = self.model.encode(refs, normalize_embeddings=True, convert_to_numpy=True)
        e2 = self.model.encode(hyps, normalize_embeddings=True, convert_to_numpy=True)
        sims = np.sum(e1 * e2, axis=1)
        return {"mean": float(sims.mean()), "std": float(sims.std()), "values": sims.tolist()}
