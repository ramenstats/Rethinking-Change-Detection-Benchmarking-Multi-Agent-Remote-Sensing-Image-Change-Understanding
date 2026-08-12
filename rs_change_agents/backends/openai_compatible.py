from __future__ import annotations
import base64, mimetypes
from pathlib import Path
from openai import OpenAI
from .base import VLBackend


def _data_url(path: str) -> str:
    p = Path(path)
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    raw = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{raw}"

class OpenAICompatibleVLBackend(VLBackend):
    """For vLLM/SGLang-style OpenAI-compatible /v1 endpoints that support image_url content."""
    def __init__(self, cfg: dict):
        self.model_name = cfg["name"]
        self.client = OpenAI(
            base_url=cfg.get("base_url", "http://localhost:8000/v1"),
            api_key=cfg.get("api_key", "EMPTY"),
        )

    def generate(self, prompt: str, image_paths: list[str] | None = None,
                 temperature: float = 0.2, max_new_tokens: int = 256) -> str:
        content = [{"type": "text", "text": prompt}]
        for p in image_paths or []:
            content.append({"type": "image_url", "image_url": {"url": _data_url(p)}})
        rsp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        return (rsp.choices[0].message.content or "").strip()
