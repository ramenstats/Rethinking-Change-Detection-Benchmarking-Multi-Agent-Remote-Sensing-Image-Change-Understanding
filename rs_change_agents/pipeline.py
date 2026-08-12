from __future__ import annotations
from .backends.factory import build_backend
from .selector import CaptionSelector
from .agents.cc_agent import RSCCAgent
from .agents.vqa_agent import RSVQAAgent
from .agents.main_agent import MainAgent


def build_system(cfg: dict):
    backend = build_backend(cfg["model"])
    selector = CaptionSelector(
        cfg.get("selector", {}).get("embedding_model", "Qwen/Qwen3-Embedding-0.6B"),
        cfg.get("selector", {}).get("mmr_lambda", 0.70),
        cfg.get("selector", {}).get("device", None),
    )
    cc = RSCCAgent(backend, selector, cfg.get("caption_agent", {}), cfg.get("model", {}))
    vqa = RSVQAAgent(backend, cfg.get("vqa_agent", {}))
    main = MainAgent(backend, cc, vqa, cfg.get("planner", {}))
    return backend, main
