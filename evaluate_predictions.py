#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from rs_change_agents.utils import read_jsonl, write_json
from rs_change_agents.evaluator import EmbeddingEvaluator


def main():
    ap = argparse.ArgumentParser(description="Compute Human-LLM / reference-generated cosine similarity when references are available.")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--task", choices=["caption", "vqa"], default="caption")
    ap.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--output", default="embedding_eval.json")
    args = ap.parse_args()

    preds = {r.get("id"): r for r in read_jsonl(args.predictions)}
    refs = read_jsonl(args.manifest)
    ref_texts, hyp_texts = [], []

    if args.task == "caption":
        for r in refs:
            p = preds.get(r.get("id"))
            if not p or not r.get("reference_caption"):
                continue
            selected = p.get("caption", {}).get("selected", [])
            if selected:
                ref_texts.append(r["reference_caption"])
                hyp_texts.append(selected[0])
    else:
        for r in refs:
            p = preds.get(r.get("id"))
            if not p:
                continue
            ref_q = {x["question"]: x.get("answer") for x in r.get("questions", []) if isinstance(x, dict)}
            for x in p.get("vqa", []):
                if ref_q.get(x["question"]):
                    ref_texts.append(ref_q[x["question"]])
                    hyp_texts.append(x["answer"])

    if not ref_texts:
        raise SystemExit("No matching references found. Add reference_caption or question/answer references to the manifest.")
    ev = EmbeddingEvaluator(args.embedding_model)
    result = ev.pair_similarity(ref_texts, hyp_texts)
    result["n"] = len(ref_texts)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
