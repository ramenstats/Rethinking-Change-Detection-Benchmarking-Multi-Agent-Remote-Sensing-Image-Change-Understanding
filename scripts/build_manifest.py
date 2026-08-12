#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
from PIL import Image
import numpy as np

EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def changed_from_mask(mask_path: Path) -> bool:
    a = np.asarray(Image.open(mask_path).convert("L"))
    return bool((a > 0).any())


def main():
    ap = argparse.ArgumentParser(description="Build a JSONL manifest from paired BEFORE/AFTER folders.")
    ap.add_argument("--pre-dir", required=True)
    ap.add_argument("--post-dir", required=True)
    ap.add_argument("--mask-dir", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    pre_dir, post_dir = Path(args.pre_dir), Path(args.post_dir)
    mask_dir = Path(args.mask_dir) if args.mask_dir else None
    post_by_stem = {p.stem: p for p in post_dir.iterdir() if p.suffix.lower() in EXTS}
    mask_by_stem = {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() in EXTS} if mask_dir else {}
    rows = []
    for pre in sorted(p for p in pre_dir.iterdir() if p.suffix.lower() in EXTS):
        post = post_by_stem.get(pre.stem)
        if not post:
            continue
        mask = mask_by_stem.get(pre.stem)
        row = {
            "id": pre.stem,
            "pre_path": str(pre.resolve()),
            "post_path": str(post.resolve()),
            "mask_path": str(mask.resolve()) if mask else None,
        }
        if mask:
            row["change_type"] = "changed" if changed_from_mask(mask) else "no_change"
        rows.append(row)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} pairs to {args.output}")

if __name__ == "__main__": main()
