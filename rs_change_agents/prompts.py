from __future__ import annotations

SYSTEM_RS = (
    "You are a remote sensing image analysis expert. The first image is BEFORE and "
    "the second image is AFTER. Compare only visually supported evidence. Avoid "
    "inventing causes, dates, identities, or quantities that cannot be estimated from the images."
)

CAPTION_STRATEGIES = [
    "Describe the most important semantic changes between BEFORE and AFTER in 1-3 precise sentences. Mention changed object categories and what was added, removed, expanded, reduced, or unchanged.",
    "Write an informative remote-sensing change caption. Focus on land-cover/land-use transitions, buildings, roads, vegetation, water, bare land, and spatial concentration of changes when visible.",
    "Compare BEFORE and AFTER systematically. State the dominant change first, then supporting details about location and approximate extent/count. If there is no meaningful change, explicitly say so.",
    "Produce a concise but information-rich change description suitable for a benchmark annotation. Distinguish additions, removals, and modifications; do not merely say 'changed'.",
    "Act as an expert annotator. Explain what changed, where it changed relative to the image (top/bottom/left/right/center), and whether the scene suggests a land-use transition. Keep claims grounded in visible evidence.",
]

DEFAULT_QUESTION_BANK = [
    "What are the main semantic changes between the BEFORE and AFTER images?",
    "Which object categories appear to have been added, removed, expanded, reduced, or modified?",
    "Where are the most important changes located relative to the image boundaries or center?",
    "Approximately how many distinct changed objects or changed clusters are visible? State uncertainty if exact counting is difficult.",
    "What overall land-cover or land-use trend is visible across the two times?",
]


def caption_prompt(strategy: str, weak_caption: str | None = None, has_mask: bool = False, background: str | None = None) -> str:
    parts = [SYSTEM_RS, strategy]
    if weak_caption:
        parts.append(
            "A weak/preliminary caption is provided below. Enrich it only when supported by the images; correct it if it conflicts with the images.\n"
            f"WEAK CAPTION: {weak_caption}"
        )
    if has_mask:
        parts.append("A third image is an optional change mask. Use it only as a spatial cue; infer semantics from BEFORE/AFTER imagery.")
    if background:
        parts.append(f"OPTIONAL BACKGROUND CONTEXT: {background}")
    parts.append("Return only the final caption, without analysis notes or bullet labels.")
    return "\n\n".join(parts)


def vqa_prompt(question: str, caption_context: list[str] | None = None, background: str | None = None) -> str:
    parts = [SYSTEM_RS]
    if caption_context:
        joined = "\n".join(f"- {c}" for c in caption_context)
        parts.append(
            "KNOWLEDGE BRIDGE CONTEXT (candidate change captions). Treat these as fallible priors: verify against the images before using them.\n"
            + joined
        )
    if background:
        parts.append(f"OPTIONAL BACKGROUND CONTEXT: {background}")
    parts.append(f"QUESTION: {question}")
    parts.append(
        "Answer directly and specifically. Include object categories, spatial relations, approximate counts, or trend interpretation when relevant. "
        "If evidence is insufficient, say what is uncertain."
    )
    return "\n\n".join(parts)


def planner_prompt(task_request: str) -> str:
    return f"""You are the Main-Agent planner for a remote-sensing change-understanding system.
Route the user request to one of: caption, vqa, both.
- caption: requests to describe/summarize changes.
- vqa: asks a specific question that should be answered.
- both: requests both a change summary and question answering, or is ambiguous but benefits from both.
Return strict JSON only: {{"route":"caption|vqa|both"}}.
USER REQUEST: {task_request}"""


def judge_prompt(task_type: str, output_text: str, reference: str | None = None) -> str:
    ref = f"\nREFERENCE (if available): {reference}" if reference else ""
    return f"""You are evaluating a remote-sensing {task_type} output.
Score from 1 to 10 using these criteria: informativeness, factual/semantic quality, specificity, and relevance. For captions, also consider whether the wording captures diverse change aspects without redundancy. Do not reward unsupported detail.
OUTPUT: {output_text}{ref}
Return strict JSON only in this form:
{{"score": 1.0, "informativeness": 1.0, "quality": 1.0, "specificity": 1.0, "reason": "short reason"}}"""
