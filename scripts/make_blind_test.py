#!/usr/bin/env python
from __future__ import annotations
import argparse, csv, random
from rs_change_agents.utils import read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--public-csv", default="blind_test.csv")
    ap.add_argument("--key-csv", default="blind_test_key.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    preds = {r.get("id"): r for r in read_jsonl(args.predictions)}
    rows = []
    for ref in read_jsonl(args.manifest):
        if not ref.get("reference_caption") or ref.get("id") not in preds: continue
        sel = preds[ref["id"]].get("caption", {}).get("selected", [])
        if not sel: continue
        rows.append((ref["id"], "A", ref["reference_caption"], "human"))
        rows.append((ref["id"], "B", sel[0], "llm"))
    random.shuffle(rows)
    with open(args.public_csv, "w", newline="", encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["item_id","option","text","participant_guess_human_or_llm"])
        for i,(sid,opt,text,label) in enumerate(rows): w.writerow([i,opt,text,""])
    with open(args.key_csv, "w", newline="", encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["item_id","source_id","label"])
        for i,(sid,opt,text,label) in enumerate(rows): w.writerow([i,sid,label])
    print(f"Saved {args.public_csv} and hidden key {args.key_csv}")

if __name__ == "__main__": main()
