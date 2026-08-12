#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
from tqdm import tqdm
from rs_change_agents.utils import load_yaml, seed_everything, ensure_dir, append_jsonl
from rs_change_agents.pipeline import build_system
from rs_change_agents.dataset import ManifestDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen7b.yaml")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--task", choices=["caption", "vqa", "both"], default="both")
    ap.add_argument("--request", default="Perform multimodal remote-sensing change understanding.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    _, main_agent = build_system(cfg)
    ds = ManifestDataset(args.manifest)
    out_dir = ensure_dir(cfg.get("output_dir", "outputs/run"))
    out_path = Path(args.output) if args.output else out_dir / "predictions.jsonl"
    if out_path.exists():
        out_path.unlink()

    n = len(ds) if args.limit <= 0 else min(len(ds), args.limit)
    for i in tqdm(range(n), desc="samples"):
        sample = ds[i]
        result = main_agent.run(sample, args.request, explicit_task=args.task)
        append_jsonl(out_path, result)
    print(f"Saved {n} results to {out_path}")

if __name__ == "__main__":
    main()
