# Multi-Agent Remote-Sensing Change Understanding — Reproduction Scaffold

This repository implements the workflow described in **“Rethinking Change Detection: Benchmarking Multi-Agent Remote Sensing Image Change Understanding” (ICASSP 2026)** as a runnable research scaffold.

## What is faithful to the paper

- Training-free agent orchestration.
- Main-Agent routing.
- RS-CC-Agent that generates multiple informative change-caption candidates.
- Embedding/cosine-based candidate filtering to obtain C*.
- RS-VQA-Agent for open-ended questions.
- Knowledge Bridge that passes C* from captioning to VQA.
- Optional use of an existing change mask.
- LLM-as-Judge utility and embedding-similarity evaluation utility.
- Human blind-test CSV generator.

## What the paper does NOT specify

The paper does not publish exact prompts, an exact SELECT(C) formula, the full VQA question bank, all human reference texts, dataset preprocessing details, or authors' source code. Therefore this repository uses clearly marked reproducible implementation choices for those missing details. It should be treated as a faithful **reference implementation**, not byte-for-byte reproduction of undisclosed code.

## 1. Environment

Recommended: Linux + NVIDIA GPU.

```bash
conda create -n rs_multiagent python=3.11 -y
conda activate rs_multiagent
pip install -r requirements.txt
```

If your Transformers build does not recognize Qwen2.5-VL, upgrade Transformers from the official repository:

```bash
pip install -U git+https://github.com/huggingface/transformers accelerate
pip install -U qwen-vl-utils
```

## 2. Quick pipeline test without a GPU model

This validates the code structure only:

```bash
python run_single.py \
  --config configs/mock.yaml \
  --pre /path/to/before.png \
  --post /path/to/after.png \
  --task both
```

## 3. Real local Qwen2.5-VL run

```bash
python run_single.py \
  --config configs/qwen7b.yaml \
  --pre /path/to/A.png \
  --post /path/to/B.png \
  --mask /path/to/label.png \
  --task both \
  --question "What buildings were added or removed?" \
  --question "Where are the main changes located?" \
  --output result.json
```

The mask is optional. The default config keeps the Qwen3 embedding selector on CPU to reduce GPU-memory pressure.

## 4. Dataset layout and manifest

A convenient LEVIR-style layout is:

```text
LEVIR-CD/
  train/
    A/
    B/
    label/
  test/
    A/
    B/
    label/
```

Create a manifest:

```bash
python scripts/build_manifest.py \
  --pre-dir /data/LEVIR-CD/test/A \
  --post-dir /data/LEVIR-CD/test/B \
  --mask-dir /data/LEVIR-CD/test/label \
  --output levir_test.jsonl
```

Each JSONL row contains `id`, `pre_path`, `post_path`, and optional `mask_path`, `weak_caption`, `reference_caption`, `background`, and `questions`.

## 5. Run a dataset

Start small:

```bash
python run_dataset.py \
  --config configs/qwen7b.yaml \
  --manifest levir_test.jsonl \
  --task both \
  --limit 10
```

Then full run:

```bash
python run_dataset.py \
  --config configs/qwen7b.yaml \
  --manifest levir_test.jsonl \
  --task both
```

Output is written to `outputs/qwen7b/predictions.jsonl`.

## 6. Caption-only and VQA-only modes

```bash
python run_dataset.py --config configs/qwen7b.yaml --manifest levir_test.jsonl --task caption
python run_dataset.py --config configs/qwen7b.yaml --manifest levir_test.jsonl --task vqa
```

In VQA-only mode, if Knowledge Bridge is enabled, the CC agent is first run internally to obtain C*.

## 7. Knowledge Bridge ablation

Edit:

```yaml
vqa_agent:
  use_knowledge_bridge: false
```

Run once with `true` and once with `false`, then compare judge or reference-based scores.

## 8. Background/CoT-style ablation

Pass background to a single run:

```bash
python run_single.py ... --background "The scene may contain urban expansion."
```

For dataset benchmarking, add a `background` field in each manifest row. Compare with and without this field.

## 9. Embedding similarity

The paper reports Qwen3-0.6B embedding cosine similarity. If you have human/reference captions, add:

```json
{"reference_caption":"Several new buildings and a road appeared..."}
```

Then:

```bash
python evaluate_predictions.py \
  --predictions outputs/qwen7b/predictions.jsonl \
  --manifest levir_test_with_refs.jsonl \
  --task caption
```

VQA reference format:

```json
"questions": [
  {"question":"What changed?", "answer":"Several buildings were constructed."}
]
```

Then use `--task vqa`.

## 10. LLM-as-Judge

```bash
python judge_predictions.py \
  --config configs/qwen7b.yaml \
  --predictions outputs/qwen7b/predictions.jsonl \
  --task caption \
  --output caption_judge.jsonl
```

This uses the configured backend as judge. The paper used GPT-4o; configure a suitable compatible endpoint if you need a different judge.

## 11. Human blind test

Requires human reference captions in the manifest:

```bash
python scripts/make_blind_test.py \
  --predictions outputs/qwen7b/predictions.jsonl \
  --manifest levir_test_with_refs.jsonl
```

This creates a public response CSV and a separate hidden answer key.

## 12. Faster serving with vLLM

Start a server:

```bash
pip install vllm
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000
```

Then run:

```bash
python run_dataset.py \
  --config configs/vllm_qwen7b.yaml \
  --manifest levir_test.jsonl \
  --task both \
  --limit 10
```

## 13. 32B paper-style VQA backbone

If your GPU resources are sufficient, replace model name with:

```yaml
name: Qwen/Qwen2.5-VL-32B-Instruct
```

For multi-GPU/high-throughput experiments, serving through vLLM or SGLang is preferable.

## Output schema

Typical output:

```json
{
  "id": "sample_001",
  "route": "both",
  "caption": {
    "candidates": ["...", "..."],
    "selected": ["..."],
    "selection": {"indices": [0], "scores": [0.71]}
  },
  "vqa": [
    {"question": "What changed?", "answer": "..."}
  ],
  "knowledge_bridge": {
    "captions": ["..."],
    "answers": [{"question": "...", "answer": "..."}]
  }
}
```
