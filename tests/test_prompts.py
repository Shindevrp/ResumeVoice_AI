from modules.dialogue.prompts import build_system_prompt


class TestSystemPrompt:
    def test_no_repetitive_question_guidance(self) -> None:
        p = build_system_prompt()
        assert "Check if the user is following along" not in p

    def test_anti_repetition_block_included(self) -> None:
        p = build_system_prompt()
        assert "NEVER end every response with a question" in p

    def test_no_premature_goodbye(self) -> None:
        p = build_system_prompt()
        assert "Do not greet again or say goodbye" in p

    def test_disengaged_prompt_not_pushing_questions(self) -> None:
        p = build_system_prompt(engagement=0.2)
        assert "Ask simple follow-ups" not in p

    def test_ongoing_context_after_many_turns(self) -> None:
        p = build_system_prompt(turn_count=5)
        assert "ongoing conversation" in p
