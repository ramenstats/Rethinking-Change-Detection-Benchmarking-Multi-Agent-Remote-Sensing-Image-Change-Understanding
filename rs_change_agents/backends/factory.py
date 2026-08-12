from __future__ import annotations
from .base import VLBackend

def build_backend(cfg: dict) -> VLBackend:
    kind = cfg.get("backend", "qwen_transformers")
    if kind == "qwen_transformers":
        from .qwen_transformers import QwenTransformersBackend
        return QwenTransformersBackend(cfg)
    if kind == "openai_compatible":
        from .openai_compatible import OpenAICompatibleVLBackend
        return OpenAICompatibleVLBackend(cfg)
    if kind == "mock":
        from .mock import MockBackend
        return MockBackend()
    raise ValueError(f"Unknown backend: {kind}")
