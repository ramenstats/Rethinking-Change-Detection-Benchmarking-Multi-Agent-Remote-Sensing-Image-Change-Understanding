from __future__ import annotations
from abc import ABC, abstractmethod

class VLBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, image_paths: list[str] | None = None,
                 temperature: float = 0.2, max_new_tokens: int = 256) -> str:
        raise NotImplementedError
