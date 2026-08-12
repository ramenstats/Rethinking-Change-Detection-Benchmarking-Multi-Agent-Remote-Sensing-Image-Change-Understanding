from __future__ import annotations
from .base import VLBackend

class MockBackend(VLBackend):
    def generate(self, prompt: str, image_paths: list[str] | None = None,
                 temperature: float = 0.2, max_new_tokens: int = 256) -> str:
        lower = prompt.lower()
        if "strict json" in lower and "route" in lower:
            return '{"route":"both"}'
        if "strict json" in lower and "score" in lower:
            return '{"score":7.5,"informativeness":7.5,"quality":7.5,"specificity":7.0,"reason":"mock judge"}'
        if "question:" in lower:
            return "Mock VQA answer: the visible change should be verified with a real VLM backend."
        return "Mock caption: semantic changes should be generated with a real VLM backend."
