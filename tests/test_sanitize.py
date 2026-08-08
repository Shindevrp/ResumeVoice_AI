from __future__ import annotations

from modules.tts.sanitize import sanitize_for_tts


class TestSanitizeForTTS:
    def test_bold_markers_removed(self) -> None:
        assert sanitize_for_tts("focus on **practical steps** now") == (
            "focus on practical steps now"
        )

    def test_italic_markers_removed(self) -> None:
        assert sanitize_for_tts("a *quick* and _easy_ fix") == "a quick and easy fix"

    def test_inline_code_kept_content(self) -> None:
        assert sanitize_for_tts("run `npm test` now") == "run npm test now"

    def test_code_fence_removed(self) -> None:
        assert sanitize_for_tts("Try this:\n```\nprint(1)\n```\nnext step") == (
            "Try this: next step"
        )

    def test_links_keep_label(self) -> None:
        assert sanitize_for_tts("see [the docs](https://x.dev) for info") == (
            "see the docs for info"
        )

    def test_headers_stripped(self) -> None:
        assert sanitize_for_tts("## Practical Steps\nthen this") == (
            "Practical Steps then this"
        )

    def test_numbered_and_bullet_lists(self) -> None:
        text = "Steps:\n1. **Monitor** levels\n2. Recycle water\n- Reuse greywater"
        out = sanitize_for_tts(text)
        assert "**" not in out
        assert "Monitor" in out
        assert "Recycle" in out
        assert "Reuse" in out
        assert out.startswith("Steps:")

    def test_table_pipes_spoken_as_spaces(self) -> None:
        assert sanitize_for_tts("A | B\n---|---\nx | y") == "A B x y"

    def test_empty_and_whitespace_collapsed(self) -> None:
        assert sanitize_for_tts("  hello    world  ") == "hello world"

    def test_plain_text_unchanged(self) -> None:
        assert sanitize_for_tts("That's a great question.") == (
            "That's a great question."
        )
