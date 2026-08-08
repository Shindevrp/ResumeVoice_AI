from __future__ import annotations

from modules.turn.intent import IntentClassifier


class TestIntentClassifier:
    def setup_method(self) -> None:
        self.c = IntentClassifier()

    def test_question_ending_with_mark(self) -> None:
        assert self.c.classify("What time is it?") == "question"

    def test_question_leading_word(self) -> None:
        assert self.c.classify("how does this work") == "question"

    def test_correction(self) -> None:
        assert self.c.classify("no, I meant the other one") == "correction"
        assert self.c.classify("that's not what I said") == "correction"

    def test_bare_no_is_not_correction(self) -> None:
        assert self.c.classify("no") != "correction"

    def test_farewell(self) -> None:
        assert self.c.classify("goodbye") == "farewell"
        assert self.c.classify("see you later") == "farewell"

    def test_greeting(self) -> None:
        assert self.c.classify("hello there") == "greeting"

    def test_backchannel(self) -> None:
        assert self.c.classify("okay") == "backchannel"
        assert self.c.classify("got it") == "backchannel"

    def test_command(self) -> None:
        assert self.c.classify("play some music") == "command"
        assert self.c.classify("remind me to call mom") == "command"

    def test_continuation_prefix(self) -> None:
        assert self.c.classify("and also, the flight was late") == "continuation"

    def test_continuation_reference_after_statement(self) -> None:
        assert (
            self.c.classify("I really like that", prev_intent="statement")
            == "continuation"
        )

    def test_plain_statement(self) -> None:
        assert self.c.classify("I really enjoy hiking in the mountains") == "statement"

    def test_tool_marker_is_command(self) -> None:
        assert self.c.classify("please do {tool:create_timer}") == "command"
