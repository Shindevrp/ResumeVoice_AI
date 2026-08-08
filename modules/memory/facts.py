from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

EXTRACTOR_SYSTEM_PROMPT = (
    "Extract durable personal facts from the user's spoken message. "
    "Return ONLY a JSON list of objects, each with "
    '{"key": "snake_case_slug", "value": "short text", "confidence": 0.0-1.0}. '
    "Include only stable facts: identity, preferences, family, pets, location, "
    "work, dislikes, possessions, dietary choices. Skip transient opinions and "
    "anything that is obvious or not durable. If there is nothing durable, "
    "return []. Do not include markdown or prose."
)


@dataclass(frozen=True)
class Fact:
    key: str
    value: str
    category: str
    source_turn: int
    confidence: float = 0.95
    source: str = "regex"


_CONNECTORS = {"and", "with", "for", "near", "at", "in", "on", "close", "but", "like", "close to"}

_ValueFn = Callable[[re.Match], tuple[str, str]]


def _clean(value: str) -> str:
    """Trim a multiword value at trailing connector words (e.g. 'Austin and')."""
    parts = value.split()
    for i in range(1, len(parts)):
        if parts[i].lower() in _CONNECTORS:
            return " ".join(parts[:i])
    return value


def _kv(key: str, group: int = 1) -> _ValueFn:
    def extract(m: re.Match) -> tuple[str, str]:
        return key, _clean(m.group(group).strip())

    return extract


def _pet(m: re.Match) -> tuple[str, str]:
    kind = m.group(1)
    return f"pet:{kind}", m.group(2) or kind


def _family_name(m: re.Match) -> tuple[str, str]:
    return f"family:{m.group(1)}", m.group(2)


def _family_age(m: re.Match) -> tuple[str, str]:
    return f"family:{m.group(1)}:age", m.group(2)


def _avoids(m: re.Match) -> tuple[str, str]:
    food = m.group(1).replace(" ", "_")
    return f"avoids:{food}", m.group(1)


def _owns(m: re.Match) -> tuple[str, str]:
    return f"owns:{m.group(1)}", m.group(1)


# (category, compiled regex, value extractor) — high-precision only.
_RULES: list[tuple[str, re.Pattern[str], _ValueFn]] = [
    ("identity", re.compile(r"\b(?:my name is|i'?m called|you can call me|call me)\s+([a-z]+)", re.IGNORECASE), _kv("name")),
    ("identity", re.compile(r"\bmy age is\s+(\d{1,3})\b", re.IGNORECASE), _kv("age")),
    ("identity", re.compile(r"\b(?:i'?m|i am)\s+(\d{1,3})\s+(?:years? old|yo)\b", re.IGNORECASE), _kv("age")),
    ("personal", re.compile(r"\bi (?:live in|stay in|am from|'m from|moved to)\s+([a-z]+(?: [a-z]+)?)", re.IGNORECASE), _kv("location")),
    ("personal", re.compile(r"\b(?:i work as|i work at|my job is)\s+([a-z]+(?: [a-z]+)?)", re.IGNORECASE), _kv("job")),
    ("personal", re.compile(r"\b(?:i|we) have (?:a|an)\s+(dog|cat|bird|fish|hamster|rabbit|parrot|turtle)\b(?:\s+named\s+([a-z]+))?", re.IGNORECASE), _pet),
    ("personal", re.compile(r"\bmy (wife|husband|daughter|son|mother|father|sister|brother|mom|dad|girlfriend|boyfriend)\s+is\s+(?:named\s+)?([a-z]+)\b", re.IGNORECASE), _family_name),
    ("personal", re.compile(r"\bmy (daughter|son|sister|brother|wife|husband|mom|dad)\s+is\s+(\d{1,3})\b", re.IGNORECASE), _family_age),
    ("preference", re.compile(r"\bi'?m\s+(?:a|an)?\s*(vegetarian|vegan|pescatarian|flexitarian)\b", re.IGNORECASE), _kv("diet")),
    ("preference", re.compile(r"\bi (?:don'?t|do not)\s+eat\s+([a-z]+(?: [a-z]+)?)", re.IGNORECASE), _avoids),
    ("preference", re.compile(r"\bmy favourite? food is\s+([a-z]+(?: [a-z]+)?)", re.IGNORECASE), _kv("favorite_food")),
    ("preference", re.compile(r"\b(?:my hobby is|i love to)\s+([a-z]+(?: [a-z]+)?)", re.IGNORECASE), _kv("hobby")),
    ("personal", re.compile(r"\b(?:i own|i bought)\s+(?:a|an|my)\s+(house|car|apartment|bike|motorcycle)\b", re.IGNORECASE), _owns),
]


class FactMemory:
    """Stable per-session user facts that survive history truncation.

    Extraction is regex-based (offline, instant); an optional LLM pass can
    add broader facts. Latest statement for a key always wins, so the user
    can correct earlier claims.
    """

    def __init__(self, max_facts: int = 24) -> None:
        self.max_facts = max_facts
        self._facts: dict[str, Fact] = {}
        self._turn = 0

    @property
    def turn(self) -> int:
        return self._turn

    def advance_turn(self) -> None:
        self._turn += 1

    def __len__(self) -> int:
        return len(self._facts)

    def add(self, fact: Fact) -> None:
        current = self._facts.get(fact.key)
        if current is None or fact.source_turn >= current.source_turn:
            self._facts[fact.key] = fact

    def add_all(self, facts: list[Fact]) -> None:
        for fact in facts:
            self.add(fact)

    def get(self, key: str) -> Fact | None:
        return self._facts.get(key)

    def extract(self, text: str) -> list[Fact]:
        facts: list[Fact] = []
        for category, regex, extract in _RULES:
            for match in regex.finditer(text):
                key, value = extract(match)
                if not value:
                    continue
                facts.append(
                    Fact(
                        key=key.lower(),
                        value=value,
                        category=category,
                        source_turn=self._turn,
                        confidence=0.95,
                        source="regex",
                    )
                )
        return facts

    def to_block(self) -> str:
        if not self._facts:
            return ""
        ranked = sorted(
            self._facts.values(), key=lambda f: (-f.confidence, -f.source_turn)
        )
        lines = [f"- {f.key}: {f.value}" for f in ranked[: self.max_facts]]
        return "Facts about the user:\n" + "\n".join(lines)
