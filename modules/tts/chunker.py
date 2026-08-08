from __future__ import annotations

_SENTENCE_TERMINATORS = frozenset(".!?")
_CLAUSE_CHARS = frozenset(",;:\u2014-")
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "dept",
    "e.g", "i.e", "etc", "vs", "cf", "al", "fig", "no", "vol",
    "approx", "inc", "ltd", "co", "jan", "feb", "mar", "apr", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
}


class TTSChunker:
    """Accumulates LLM tokens into prosody-aware chunks for TTS.

    Splits on sentence boundaries first (most natural pause), then on
    clause boundaries for long spans, and finally on a hard character cap.
    Avoids splitting on abbreviations, initials, and awkwardly short fragments
    so that e.g. "That's a great question, let me explain." stays together.
    """

    def __init__(
        self,
        min_chars: int = 12,
        max_chars: int = 150,
        clause_threshold: int = 40,
    ) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.clause_threshold = clause_threshold
        self._buffer = ""

    def feed(self, token: str) -> list[str]:
        """Append a token and return any complete chunks (in order)."""
        self._buffer += token
        chunks: list[str] = []
        while True:
            cut = self._find_cut(self._buffer)
            if cut is None:
                break
            chunk = self._buffer[:cut].strip()
            self._buffer = self._buffer[cut:].lstrip()
            if chunk:
                chunks.append(chunk)
        return chunks

    def flush(self) -> str | None:
        """Return any remaining partial chunk and reset the buffer."""
        buf = self._buffer.strip()
        self._buffer = ""
        return buf or None

    @property
    def buffered(self) -> str:
        return self._buffer

    def reset(self) -> None:
        self._buffer = ""

    def _find_cut(self, text: str) -> int | None:
        sent_end = self._find_sentence_end(text)
        if sent_end is not None and len(text[:sent_end].strip()) >= self.min_chars:
            return sent_end

        if len(text) >= self.max_chars:
            clause = self._find_clause(text)
            if clause is not None and len(text[:clause].strip()) >= self.min_chars:
                return clause
            space = text.rfind(" ", 0, self.max_chars)
            if space > 0:
                return space
            return self.max_chars

        return None

    def _find_sentence_end(self, text: str) -> int | None:
        """Return the cut index (after a terminator) for the last real sentence end."""
        best: int | None = None
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            if ch in _SENTENCE_TERMINATORS:
                if self._is_abbreviation(text, i):
                    i += 1
                    continue
                j = i + 1
                while j < n and text[j] in "\"')\u201d\u2019]":
                    j += 1
                if j >= n:
                    # End of buffer: a complete sentence only if enough content.
                    if len(text) >= self.min_chars:
                        best = j
                elif text[j].isspace():
                    best = j + 1
            i += 1
        return best

    def _find_clause(self, text: str) -> int | None:
        positions = [i for i, ch in enumerate(text) if ch in _CLAUSE_CHARS]
        if not positions:
            return None
        for pos in reversed(positions):
            # Prefer a clause that yields a reasonably sized chunk.
            if pos >= self.clause_threshold:
                return pos + 1
        return None

    def _is_abbreviation(self, text: str, idx: int) -> bool:
        if text[idx] != ".":
            return False
        # Decimal / number like "3.5" or "1."
        start = idx - 1
        while start >= 0 and (text[start].isdigit() or text[start] == ","):
            start -= 1
        if text[start + 1:idx] and text[start + 1:idx].isdigit():
            return True
        # Word ending at the period.
        start = idx - 1
        while start >= 0 and (text[start].isalnum() or text[start] in "'._"):
            start -= 1
        word = text[start + 1:idx].strip().rstrip(".").lower()
        if not word:
            return False
        if word in _ABBREVIATIONS:
            return True
        if len(word) <= 2 and word.isalpha():
            return True
        return False
