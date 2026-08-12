from __future__ import annotations
from ..utils import extract_json
from ..prompts import planner_prompt, DEFAULT_QUESTION_BANK
from ..bridge import KnowledgeBridge

class MainAgent:
    def __init__(self, backend, cc_agent, vqa_agent, planner_cfg: dict):
        self.backend = backend
        self.cc_agent = cc_agent
        self.vqa_agent = vqa_agent
        self.planner_cfg = planner_cfg

    def route(self, task_request: str, explicit_task: str | None = None) -> str:
        if explicit_task in {"caption", "vqa", "both"}:
            return explicit_task
        if self.planner_cfg.get("mode", "rules") == "llm":
            out = self.backend.generate(planner_prompt(task_request), image_paths=None, temperature=0.0, max_new_tokens=64)
            route = extract_json(out).get("route")
            if route in {"caption", "vqa", "both"}:
                return route
        t = task_request.lower()
        has_q = "?" in task_request or any(w in t for w in ["what", "where", "which", "how many", "why", "question", "answer"])
        has_c = any(w in t for w in ["caption", "describe", "description", "summar", "explain the change", "changes"])
        if has_q and has_c:
            return "both"
        if has_q:
            return "vqa"
        if has_c:
            return "caption"
        return "both"

    def run(self, sample: dict, task_request: str, explicit_task: str | None = None,
            questions: list[str] | None = None, background: str | None = None) -> dict:
        route = self.route(task_request, explicit_task)
        bridge = KnowledgeBridge()
        out = {"id": sample.get("id"), "route": route}

        if route in {"caption", "both"}:
            cc = self.cc_agent.run(
                sample["pre_path"], sample["post_path"], sample.get("mask_path"),
                weak_caption=sample.get("weak_caption"), background=background or sample.get("background")
            )
            bridge.set_captions(cc["selected"])
            out["caption"] = cc

        if route in {"vqa", "both"}:
            # If VQA is requested alone and KB is enabled, run CC silently to obtain C*.
            if not bridge.captions and self.vqa_agent.cfg.get("use_knowledge_bridge", True):
                cc = self.cc_agent.run(
                    sample["pre_path"], sample["post_path"], sample.get("mask_path"),
                    weak_caption=sample.get("weak_caption"), background=background or sample.get("background")
                )
                bridge.set_captions(cc["selected"])
                out["bridge_caption_generation"] = cc

            q_list = questions or sample.get("questions") or DEFAULT_QUESTION_BANK
            answers = []
            for q in q_list:
                if isinstance(q, dict):
                    q_text = q["question"]
                else:
                    q_text = q
                ans = self.vqa_agent.answer(
                    sample["pre_path"], sample["post_path"], q_text,
                    bridge_captions=bridge.captions,
                    background=background or sample.get("background")
                )
                bridge.add_answer(q_text, ans)
                answers.append({"question": q_text, "answer": ans})
            out["vqa"] = answers

        if sample.get("mask_path"):
            out["optional_change_mask"] = sample["mask_path"]
        out["knowledge_bridge"] = bridge.snapshot()
        return out
