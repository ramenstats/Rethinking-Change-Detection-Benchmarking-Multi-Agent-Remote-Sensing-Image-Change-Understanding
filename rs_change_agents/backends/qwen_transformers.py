from __future__ import annotations
from pathlib import Path
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from .base import VLBackend

class QwenTransformersBackend(VLBackend):
    def __init__(self, cfg: dict):
        self.model_name = cfg.get("name", "Qwen/Qwen2.5-VL-7B-Instruct")
        kwargs = {
            "torch_dtype": cfg.get("torch_dtype", "auto"),
            "device_map": cfg.get("device_map", "auto"),
        }
        attn = cfg.get("attn_implementation")
        if attn:
            kwargs["attn_implementation"] = attn
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_name, **kwargs)
        proc_kwargs = {}
        if cfg.get("min_pixels") is not None:
            proc_kwargs["min_pixels"] = int(cfg["min_pixels"])
        if cfg.get("max_pixels") is not None:
            proc_kwargs["max_pixels"] = int(cfg["max_pixels"])
        self.processor = AutoProcessor.from_pretrained(self.model_name, **proc_kwargs)

    def generate(self, prompt: str, image_paths: list[str] | None = None,
                 temperature: float = 0.2, max_new_tokens: int = 256) -> str:
        content = []
        for p in image_paths or []:
            uri = Path(p).expanduser().resolve().as_uri()
            content.append({"type": "image", "image": uri})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        )
        # device_map='auto' may shard the model; placing inputs on the first parameter device is safe for common single-GPU use.
        device = next(self.model.parameters()).device
        inputs = inputs.to(device)
        do_sample = temperature > 0
        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
        if do_sample:
            gen_kwargs.update(temperature=max(temperature, 1e-5), top_p=0.9)
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output.strip()
