from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from utils.logger import get_logger

logger = get_logger("state")


class DialogueState(Enum):
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    INTERRUPTIBLE = auto()

    def can_transition_to(self, target: DialogueState) -> bool:
        allowed = {
            DialogueState.IDLE: {DialogueState.LISTENING},
            DialogueState.LISTENING: {DialogueState.PROCESSING, DialogueState.IDLE},
            DialogueState.PROCESSING: {DialogueState.INTERRUPTIBLE, DialogueState.IDLE},
            DialogueState.INTERRUPTIBLE: {
                DialogueState.IDLE,
                DialogueState.LISTENING,
            },
        }
        return target in allowed.get(self, set())


TRANSITION_NAMES: dict[tuple[DialogueState, DialogueState], str] = {
    (DialogueState.IDLE, DialogueState.LISTENING): "start_speech",
    (DialogueState.LISTENING, DialogueState.PROCESSING): "end_speech",
    (DialogueState.PROCESSING, DialogueState.INTERRUPTIBLE): "first_token",
    (DialogueState.INTERRUPTIBLE, DialogueState.IDLE): "response_done",
    (DialogueState.INTERRUPTIBLE, DialogueState.LISTENING): "barge_in",
    (DialogueState.LISTENING, DialogueState.IDLE): "cancel",
    (DialogueState.PROCESSING, DialogueState.IDLE): "cancel",
}


@dataclass
class Turn:
    role: str
    content: str
    confidence: float = 1.0


@dataclass
class SessionState:
    session_id: str
    state: DialogueState = DialogueState.IDLE
    transcript: list[Turn] = field(default_factory=list)
    memory: list[str] = field(default_factory=list)
    engagement_score: float = 0.5
    consecutive_silence: float = 0.0
    total_user_turns: int = 0
    total_ai_turns: int = 0
    last_activity: float = 0.0
    topic: str = ""
    intent: str = "statement"
    metadata: dict[str, Any] = field(default_factory=dict)

    def update(self, ctx: Any) -> None:
        """Sync the session with a ConversationContext (topic/intent)."""
        self.topic = getattr(ctx, "topic", "") or ""
        self.intent = getattr(ctx, "intent", "") or ""

    def add_user_turn(self, text: str, confidence: float = 1.0) -> None:
        self.transcript.append(Turn("user", text, confidence))
        self.total_user_turns += 1
        self._update_engagement()

    def add_ai_turn(self, text: str) -> None:
        self.transcript.append(Turn("assistant", text))
        self.total_ai_turns += 1

    def context_messages(self, max_turns: int = 10) -> list[dict[str, str]]:
        recent = self.transcript[-(max_turns * 2) :]
        messages = [{"role": "system", "content": self._system_prompt()}]
        for turn in recent:
            messages.append({"role": turn.role, "content": turn.content})
        return messages

    def _system_prompt(self) -> str:
        if self.engagement_score < 0.3:
            return "The user seems disengaged. Keep responses very brief and inviting."
        if self.engagement_score > 0.8:
            return "The user is highly engaged. Feel free to be more conversational."
        return "You are ResumeVoice AI, a real-time conversational assistant. Respond concisely and naturally."

    def _update_engagement(self) -> None:
        recency = 1.0 if self.total_user_turns > 0 else 0.3
        depth = min(self.total_user_turns / 10, 1.0)
        self.engagement_score = 0.5 * recency + 0.5 * depth

    def set_state(self, new_state: DialogueState) -> None:
        old = self.state
        if old == new_state:
            return
        if not old.can_transition_to(new_state):
            logger.warning(f"invalid state transition {old.name} -> {new_state.name}")
            return
        self.state = new_state
