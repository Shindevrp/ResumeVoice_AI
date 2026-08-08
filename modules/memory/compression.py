from __future__ import annotations

from modules.memory.session import MemoryEntry

_SYSTEM_PROMPT = (
    "You are a conversation summarizer for a voice assistant. You maintain "
    "a compact running summary of the dialogue. Given the current summary "
    "and a batch of new message transcripts, rewrite the summary to "
    "incorporate the new messages. Keep it under 100 words. Preserve the "
    "user's preferences, names, numbers, key facts, and any decisions. "
    "Write concise prose; do not include speaker labels or quotes."
)


class ContextCompressor:
    def __init__(self, max_chars: int = 800) -> None:
        self.max_chars = max_chars
        self.system_prompt = _SYSTEM_PROMPT

    def build_prompt(
        self,
        existing_summary: str,
        entries: list[MemoryEntry],
    ) -> list[dict[str, str]]:
        lines = []
        for e in entries:
            lines.append(f"{e.role}: {e.content}")
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Current summary:\n{existing_summary or '(none)'}"
                    f"\n\nNew messages:\n" + "\n".join(lines)
                ),
            },
        ]

    async def compress(
        self,
        llm,
        existing_summary: str,
        entries: list[MemoryEntry],
    ) -> str:
        """Stream an updated summary from the LLM, capped at max_chars."""
        prompt = self.build_prompt(existing_summary, entries)
        out = ""
        async for tok in llm.generate_stream(prompt):
            if len(out) >= self.max_chars:
                break
            out += tok
        return out[: self.max_chars].strip()
