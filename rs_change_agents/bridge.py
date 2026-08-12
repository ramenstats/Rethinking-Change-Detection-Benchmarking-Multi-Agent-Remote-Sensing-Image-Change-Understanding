from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class KnowledgeBridge:
    captions: list[str] = field(default_factory=list)
    answers: list[dict] = field(default_factory=list)

    def set_captions(self, captions: list[str]) -> None:
        self.captions = list(captions)

    def add_answer(self, question: str, answer: str) -> None:
        self.answers.append({"question": question, "answer": answer})

    def snapshot(self) -> dict:
        return {"captions": self.captions, "answers": self.answers}
