from __future__ import annotations
from ..prompts import vqa_prompt

class RSVQAAgent:
    def __init__(self, backend, cfg: dict):
        self.backend = backend
        self.cfg = cfg

    def answer(self, pre_path: str, post_path: str, question: str,
               bridge_captions: list[str] | None = None, background: str | None = None) -> str:
        context = bridge_captions if self.cfg.get("use_knowledge_bridge", True) else None
        prompt = vqa_prompt(question, caption_context=context, background=background)
        return self.backend.generate(
            prompt, image_paths=[pre_path, post_path], temperature=0.15,
            max_new_tokens=int(self.cfg.get("max_new_tokens", 320))
        )
