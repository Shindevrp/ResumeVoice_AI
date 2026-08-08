from __future__ import annotations

from modules.turn.topic import TopicTracker


class TestTopicKeywords:
    def test_extracts_content_words(self) -> None:
        t = TopicTracker()
        kws = t.keywords("I want to plan a birthday party for my daughter")
        assert "birthday" in kws
        assert "daughter" in kws
        assert "party" in kws

    def test_drops_stopwords_and_short_words(self) -> None:
        t = TopicTracker()
        kws = t.keywords("that is really going to the store")
        assert not any(w in kws for w in ("that", "the", "going"))


class TestTopicShift:
    def test_no_shift_on_same_topic(self) -> None:
        t = TopicTracker()
        t.update("Let us talk about space travel", 0)
        change = t.update("I love learning about space travel", 1)
        assert not change.shift
        assert "space" in change.topic and "travel" in change.topic

    def test_shift_on_new_topic(self) -> None:
        t = TopicTracker()
        t.update("Let us plan a birthday party", 0)
        change = t.update("I need help fixing my laptop", 1)
        assert change.shift
        assert "laptop" in change.topic

    def test_first_turn_no_shift(self) -> None:
        t = TopicTracker()
        change = t.update("hello there", 0)
        assert not change.shift
        assert "hello" in change.topic

    def test_since_turn_tracks_topic_age(self) -> None:
        t = TopicTracker()
        t.update("discussing space travel", 0)
        t.update("more about space travel", 1)
        assert t.since_turn == 0
        t.update("now about cooking recipes", 2)
        assert t.since_turn == 2


class TestTopicLabel:
    def test_label_attached_to_current_topic(self) -> None:
        t = TopicTracker()
        t.update("plan a birthday party", 0)
        assert t.label is None
        t.set_label("planning the birthday")
        assert t.label == "planning the birthday"

    def test_label_dropped_on_shift(self) -> None:
        t = TopicTracker()
        t.update("plan a birthday party", 0)
        t.set_label("planning the birthday")
        t.update("fixing my laptop instead", 1)
        assert t.label is None

    def test_needs_label_before_and_after(self) -> None:
        t = TopicTracker()
        assert not t.needs_label()
        t.update("plan a birthday party", 0)
        assert t.needs_label()
        t.set_label("planning the birthday")
        assert not t.needs_label()

    def test_needs_label_after_shift(self) -> None:
        t = TopicTracker()
        t.update("plan a birthday party", 0)
        t.set_label("planning the birthday")
        t.update("fixing my laptop instead", 1)
        assert t.needs_label()
