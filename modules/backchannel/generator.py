from __future__ import annotations

import random

BACKCHANNEL_CANDIDATES = {
    "acknowledging": ["uh-huh", "hmm", "right", "yeah", "okay", "i see"],
    "agreeing": ["yeah", "right", "exactly", "totally", "for sure"],
    "surprised": ["oh", "wow", "really", "no way", "huh"],
    "thinking": ["hmm", "let me think", "well", "so"],
    "sympathetic": ["mm-hmm", "i hear you", "right", "yeah"],
}

BACKCHANNEL_CONTEXT_KEYWORDS: dict[str, str] = {
    "surprised": ["wow", "really", "oh", "unbelievable", "crazy", "amazing"],
    "agreeing": ["exactly", "right", "true", "agree", "definitely", "absolutely"],
    "sympathetic": ["sorry", "sad", "tough", "hard", "difficult", "frustrating"],
}


class BackchannelGenerator:
    def __init__(self, cooldown_seconds: float = 3.0) -> None:
        self.cooldown_seconds = cooldown_seconds

    def generate(
        self,
        transcript: str,
        recent_backchannel_time: float,
        current_time: float,
    ) -> str | None:
        if current_time - recent_backchannel_time < self.cooldown_seconds:
            return None

        category = self._infer_context(transcript)
        candidates = BACKCHANNEL_CANDIDATES.get(category, ["uh-huh", "hmm"])
        return random.choice(candidates)

    def generate_thinking(self) -> str:
        """Return a filler phrase for when the LLM is still computing."""
        return random.choice(BACKCHANNEL_CANDIDATES["thinking"])

    def _infer_context(self, transcript: str) -> str:
        if not transcript:
            return "acknowledging"
        text_lower = transcript.lower()
        for category, keywords in BACKCHANNEL_CONTEXT_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return category
        return "acknowledging"
