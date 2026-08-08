from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class MemoryEntry:
    role: str
    content: str
    turn_index: int


class SessionMemory:
    def __init__(
        self,
        max_turns: int = 20,
        max_summary_chars: int = 800,
    ) -> None:
        self.max_turns = max_turns
        self.max_summary_chars = max_summary_chars
        self.entries: deque[MemoryEntry] = deque(maxlen=max_turns)
        self.summary: str = ""
        self._turn_counter = 0

    def add(self, role: str, content: str) -> None:
        self.entries.append(MemoryEntry(role, content, self._turn_counter))
        self._turn_counter += 1

    def get_history(self, max_turns: int | None = None) -> list[MemoryEntry]:
        k = max_turns if max_turns else self.max_turns
        return list(self.entries)[-k:]

    def entries_snapshot_oldest(self, count: int) -> list[MemoryEntry]:
        return list(self.entries)[:count]

    def fold_summary(self, summary: str, batch: list[MemoryEntry]) -> None:
        """Absorb a batch of old turns into the running summary and evict them."""
        if summary:
            self.summary = self._append_summary(self.summary, summary)
        batch_ids = {id(e) for e in batch}
        if batch_ids:
            self.entries = deque(
                (e for e in self.entries if id(e) not in batch_ids),
                maxlen=self.max_turns,
            )

    def _append_summary(self, old: str, new: str) -> str:
        merged = f"{old} {new}".strip()
        return merged[: self.max_summary_chars]

    def context_messages(self, system_prompt: str = "") -> list[dict[str, str]]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if self.summary:
            messages.append({
                "role": "system",
                "content": f"Conversation summary so far:\n{self.summary}",
            })
        for entry in self.entries:
            messages.append({"role": entry.role, "content": entry.content})
        return messages

    def token_estimate(self) -> int:
        base = sum(len(e.content.split()) + 4 for e in self.entries)
        if self.summary:
            base += len(self.summary.split())
        return base

    def truncate_to_budget(self, max_tokens: int = 4096) -> None:
        while self.token_estimate() > max_tokens and len(self.entries) > 1:
            self.entries.popleft()