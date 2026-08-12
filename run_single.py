#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
from rs_change_agents.utils import load_yaml, seed_everything, write_json
from rs_change_agents.pipeline import build_system


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen7b.yaml")
    ap.add_argument("--pre", required=True)
    ap.add_argument("--post", required=True)
    ap.add_argument("--mask", default=None)
    ap.add_argument("--task", choices=["caption", "vqa", "both"], default="both")
    ap.add_argument("--request", default="Describe and explain the semantic changes between these two remote sensing images.")
    ap.add_argument("--question", action="append", default=[])
    ap.add_argument("--weak-caption", default=None)
    ap.add_argument("--background", default=None)
    ap.add_argument("--output", default="single_result.json")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    _, main_agent = build_system(cfg)
    sample = {
        "id": Path(args.pre).stem,
        "pre_path": str(Path(args.pre).resolve()),
        "post_path": str(Path(args.post).resolve()),
        "mask_path": str(Path(args.mask).resolve()) if args.mask else None,
        "weak_caption": args.weak_caption,
    }
    result = main_agent.run(
        sample, args.request, explicit_task=args.task,
        questions=args.question or None, background=args.background
    )
    write_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved: {args.output}")

if __name__ == "__main__":
    main()
