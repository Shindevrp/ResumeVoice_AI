from __future__ import annotations

from dataclasses import dataclass

_STOPWORDS = {
    "about", "after", "again", "against", "also", "because", "been",
    "before", "being", "between", "could", "didnt", "doesnt", "doing",
    "during", "even", "every", "from", "going", "have", "having", "here",
    "into", "just", "know", "like", "more", "most", "much", "need",
    "next", "now", "only", "other", "over", "really", "right", "same",
    "should", "something", "still", "such", "than", "that", "their",
    "them", "then", "there", "these", "they", "this", "those", "through",
    "very", "want", "well", "were", "what", "when", "where", "which",
    "while", "with", "would", "your",
}


@dataclass
class Topic:
    name: str
    keywords: tuple[str, ...]
    start_turn: int
    end_turn: int | None = None


@dataclass
class TopicChange:
    shift: bool
    topic: str


class TopicTracker:
    """Tracks the current conversation topic via keyword overlap.

    Deterministic and offline: keywords are the current topic's signature;
    a new turn shifts the topic when its keywords barely overlap the
    established topic. An optional human-friendly label can be attached in
    the background (LLM) without changing the deterministic signal.
    """

    def __init__(self, shift_threshold: float = 0.2) -> None:
        self.shift_threshold = shift_threshold
        self.current: Topic | None = None
        self.history: list[Topic] = []
        self._label: str | None = None
        self._label_turn: int | None = None

    @property
    def topic(self) -> str:
        if self.current is None:
            return ""
        return self.current.name

    @property
    def since_turn(self) -> int:
        return self.current.start_turn if self.current else 0

    @property
    def label(self) -> str | None:
        if self._label is None:
            return None
        if self.current is None:
            return None
        if self._label_turn != self.current.start_turn:
            return None
        return self._label

    def needs_label(self) -> bool:
        return self.current is not None and self.label is None

    def set_label(self, label: str) -> None:
        """Attach a human label to the current topic (async/LLM upgrade)."""
        if self.current is not None and label:
            self._label = label.strip()
            self._label_turn = self.current.start_turn

    def keywords(self, text: str) -> list[str]:
        counts: dict[str, int] = {}
        for word in text.lower().split():
            clean = word.strip(".,!?;:'\"()[]-")
            if len(clean) <= 3:
                continue
            if clean in _STOPWORDS:
                continue
            if not clean.isalpha():
                continue
            counts[clean] = counts.get(clean, 0) + 1
        ranked = sorted(
            counts.items(), key=lambda kv: (-kv[1], -len(kv[0]))
        )
        return [w for w, _ in ranked[:3]]

    def update(self, text: str, turn_index: int) -> TopicChange:
        kws = self.keywords(text)
        if self.current is None or not kws:
            if kws:
                self._start(turn_index, kws)
            return TopicChange(shift=False, topic=self.topic)

        current_kw = set(self.current.keywords)
        cur_set = set(kws)
        overlap = len(cur_set & current_kw) / max(1, len(cur_set))

        if overlap < self.shift_threshold:
            self._close_current()
            self._start(turn_index, kws)
            return TopicChange(shift=True, topic=self.topic)

        self.current.name = " ".join(kws)
        self.current.keywords = tuple(kws)
        return TopicChange(shift=False, topic=self.topic)

    def _start(self, turn_index: int, kws: list[str]) -> None:
        self.current = Topic(
            name=" ".join(kws),
            keywords=tuple(kws),
            start_turn=turn_index,
        )
        self._label = None
        self._label_turn = None
        self.history.append(self.current)

    def _close_current(self) -> None:
        if self.current is not None:
            self.current.end_turn = max(0, self.current.start_turn)
            if len(self.history) > 12:
                self.history.pop(0)
