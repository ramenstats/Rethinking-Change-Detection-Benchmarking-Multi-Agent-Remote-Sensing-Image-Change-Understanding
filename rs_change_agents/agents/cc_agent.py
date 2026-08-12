from __future__ import annotations
from ..prompts import CAPTION_STRATEGIES, caption_prompt

class RSCCAgent:
    def __init__(self, backend, selector, cfg: dict, model_cfg: dict):
        self.backend = backend
        self.selector = selector
        self.cfg = cfg
        self.model_cfg = model_cfg

    def run(self, pre_path: str, post_path: str, mask_path: str | None = None,
            weak_caption: str | None = None, background: str | None = None) -> dict:
        n = int(self.cfg.get("num_candidates", 5))
        temps = self.cfg.get("temperatures", [0.2, 0.4, 0.6, 0.8, 1.0])
        strategies = (CAPTION_STRATEGIES * ((n + len(CAPTION_STRATEGIES) - 1) // len(CAPTION_STRATEGIES)))[:n]
        image_paths = [pre_path, post_path]
        use_mask = bool(mask_path and self.cfg.get("use_mask_if_available", True))
        if use_mask:
            image_paths.append(mask_path)

        candidates = []
        for i, strategy in enumerate(strategies):
            p = caption_prompt(strategy, weak_caption=weak_caption, has_mask=use_mask, background=background)
            temp = float(temps[i % len(temps)])
            c = self.backend.generate(
                p, image_paths=image_paths, temperature=temp,
                max_new_tokens=int(self.model_cfg.get("max_new_tokens", 320))
            )
            candidates.append(c)

        selected, sel_meta = self.selector.select(candidates, top_k=int(self.cfg.get("keep_top_k", 2)))
        return {"candidates": candidates, "selected": selected, "selection": sel_meta}
