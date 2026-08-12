#!/usr/bin/env python
from __future__ import annotations
import argparse
from rs_change_agents.utils import load_yaml, read_jsonl, append_jsonl
from rs_change_agents.backends.factory import build_backend
from rs_change_agents.evaluator import LLMJudge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen7b.yaml")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--task", choices=["caption", "vqa"], default="caption")
    ap.add_argument("--output", default="judge_scores.jsonl")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    backend = build_backend(cfg["model"])
    judge = LLMJudge(backend, cfg.get("judge", {}).get("max_new_tokens", 220))
    import os
    if os.path.exists(args.output): os.remove(args.output)
    for r in read_jsonl(args.predictions):
        if args.task == "caption":
            for text in r.get("caption", {}).get("selected", [])[:1]:
                append_jsonl(args.output, {"id": r.get("id"), "task": "caption", "text": text, "judge": judge.score("change caption", text)})
        else:
            for qa in r.get("vqa", []):
                append_jsonl(args.output, {"id": r.get("id"), "task": "vqa", "question": qa["question"], "text": qa["answer"], "judge": judge.score("VQA answer", qa["answer"])})
    print(f"Saved judge scores to {args.output}")

if __name__ == "__main__":
    main()
