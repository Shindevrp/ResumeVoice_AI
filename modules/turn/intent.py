from __future__ import annotations

import re

QUESTION_START_WORDS = {
    "what",
    "why",
    "how",
    "when",
    "where",
    "who",
    "which",
    "do",
    "does",
    "did",
    "is",
    "are",
    "was",
    "were",
    "can",
    "could",
    "would",
    "should",
    "will",
    "shall",
    "have",
    "has",
    "had",
}

GREETING_WORDS = {
    "hi",
    "hello",
    "hey",
    "yo",
    "sup",
    "howdy",
    "good morning",
    "good afternoon",
    "good evening",
    "hi there",
    "hello there",
}

FAREWELL_WORDS = {
    "bye",
    "goodbye",
    "see you",
    "good night",
    "goodnight",
    "take care",
    "peace out",
    "talk later",
    "that's all",
    "that is all",
}

BACKCHANNEL_WORDS = {
    "okay",
    "ok",
    "uh-huh",
    "uh huh",
    "mhmm",
    "mm-hmm",
    "got it",
    "right",
    "yeah",
    "yep",
    "sure",
    "alright",
    "aha",
    "cool",
    "thanks",
    "thank you",
    "understood",
}

CORRECTION_PATTERNS = [
    re.compile(r"\bnot that\b"),
    re.compile(r"\bi (?:meant|mean)\b"),
    re.compile(r"\bi didn'?t mean\b"),
    re.compile(r"\bthat'?s not what\b"),
    re.compile(r"\bnot exactly\b"),
    re.compile(r"\bthat'?s wrong\b"),
    re.compile(r"\byou'?re wrong\b"),
    re.compile(r"\bi changed my mind\b"),
    re.compile(r"\bscratch that\b"),
    re.compile(r"\bno[,.!]?\s+(?:no\b|wait|actually|i|that|it|the)"),
]

CONTINUATION_PREFIXES = ("and", "also", "but", "then", "plus", "or")

IMPERATIVE_VERBS = {
    "play",
    "stop",
    "start",
    "pause",
    "resume",
    "tell",
    "show",
    "remind",
    "set",
    "open",
    "close",
    "search",
    "find",
    "create",
    "add",
    "delete",
    "remove",
    "turn",
    "send",
    "call",
    "cancel",
    "skip",
    "repeat",
    "explain",
}

REFERENCE_PRONOUNS = {"it", "that", "this", "those", "these", "they", "them", "one"}


class IntentClassifier:
    """Deterministic, offline user-intent classification.

    Lexicon-first for zero latency and testability. Order matters: stronger
    signals (question, correction) beat weaker ones (greeting, backchannel).
    """

    def classify(self, text: str, prev_intent: str = "") -> str:
        text = text.strip()
        if not text:
            return "statement"

        tl = text.lower()
        words = tl.split()
        word_count = len(words)

        if tl.endswith("?"):
            return "question"
        if words[0] in QUESTION_START_WORDS:
            return "question"

        if word_count >= 3 and any(p.search(tl) for p in CORRECTION_PATTERNS):
            return "correction"

        if word_count <= 4 and any(w in tl for w in FAREWELL_WORDS):
            return "farewell"
        if word_count <= 4 and any(w in tl for w in GREETING_WORDS):
            return "greeting"
        if word_count <= 4 and any(w in tl for w in BACKCHANNEL_WORDS):
            return "backchannel"

        if words[0] in IMPERATIVE_VERBS or "{tool:" in text:
            return "command"

        if words[0] in CONTINUATION_PREFIXES:
            return "continuation"
        if (
            prev_intent == "statement"
            and word_count <= 8
            and words[-1] in REFERENCE_PRONOUNS
        ):
            return "continuation"

        return "statement"
