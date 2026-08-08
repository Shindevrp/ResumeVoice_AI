from __future__ import annotations

import re

_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]+)`")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_BOLD_UNDER = re.compile(r"__(.+?)__")
_ITALIC_STAR = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")
_ITALIC_UNDER = re.compile(r"(?<!_)_([^_]+)_(?!_)")
_HEADER = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_BULLET = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_NUMBERED = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)
_QUOTE = re.compile(r"^\s*>\s?", re.MULTILINE)
_TABLE_PIPE = re.compile(r"\s*\|\s*")
_TABLE_SEPARATOR = re.compile(r"^[\s|:\-]+$", re.MULTILINE)

_LEFT_OVER_MARKERS = "`*_~"


def sanitize_for_tts(text: str) -> str:
    """Strip markdown / formatting so TTS reads plain speech, not literal symbols."""
    text = _CODE_FENCE.sub(" ", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _BOLD.sub(r"\1", text)
    text = _BOLD_UNDER.sub(r"\1", text)
    text = _ITALIC_STAR.sub(r"\1", text)
    text = _ITALIC_UNDER.sub(r"\1", text)
    text = _HEADER.sub("", text)
    text = _BULLET.sub("", text)
    text = _NUMBERED.sub("", text)
    text = _QUOTE.sub("", text)
    text = _TABLE_SEPARATOR.sub("", text)
    text = _TABLE_PIPE.sub(" ", text)
    for ch in _LEFT_OVER_MARKERS:
        text = text.replace(ch, "")
    return " ".join(text.split())
