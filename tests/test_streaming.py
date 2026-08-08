from __future__ import annotations

import asyncio
import itertools
import struct
import time

import pytest

from core.pipeline import (
    FRAME_BYTES,
    ConversationContext,
    PipelineEvent,
    StreamingPipeline,
)
from core.state import DialogueState
from modules.backchannel.generator import BACKCHANNEL_CANDIDATES, BackchannelGenerator
from modules.turn.timing import TurnTiming


class FakeSTT:
    async def transcribe(self, audio_blob: bytes) -> str:
        return ""


class FakeLLM:
    async def generate_stream(self, messages):
        if False:
            yield ""


class FakeVAD:
    sample_rate = 16000

    def is_speech(self, chunk: bytes) -> bool:
        return False

    def reset(self) -> None:
        pass


class FakeTTS:
    sample_rate = 16000

    def __init__(self) -> None:
        self.synthesized: list[str] = []
        self.prosody_labels: list[str | None] = []

    async def synthesize_stream(self, text_chunks, prosody=None):
        buf = ""
        async for chunk in text_chunks:
            buf += chunk
        self.synthesized.append(buf)
        self.prosody_labels.append(prosody.label if prosody else None)
        yield b"\x00\x01\x02\x03"

    async def synthesize(self, text: str, prosody=None) -> bytes:
        return b""


def _make_pipeline() -> StreamingPipeline:
    return StreamingPipeline(FakeSTT(), FakeLLM(), FakeTTS(), FakeVAD())


class TestBackchannelGenerator:
    def test_generate_thinking(self) -> None:
        g = BackchannelGenerator()
        assert g.generate_thinking() in BACKCHANNEL_CANDIDATES["thinking"]


class TestTTSWorkerPriority:
    def test_backchannel_priority_before_response(self) -> None:
        async def run() -> list[str]:
            p = _make_pipeline()
            q: asyncio.PriorityQueue = asyncio.PriorityQueue()
            seq = itertools.count()
            q.put_nowait((1, next(seq), "Second sentence.", None))
            q.put_nowait((0, next(seq), "hmm", None))
            q.put_nowait((2, next(seq), None, None))
            await p._tts_worker(q, "sess", asyncio.Event())
            return p.tts.synthesized

        synthesized = asyncio.run(run())
        assert synthesized == ["hmm", "Second sentence."]

    def test_interrupt_drains_without_synthesizing(self) -> None:
        async def run() -> list[str]:
            p = _make_pipeline()
            q: asyncio.PriorityQueue = asyncio.PriorityQueue()
            seq = itertools.count()
            q.put_nowait((1, next(seq), "Should be dropped.", None))
            q.put_nowait((2, next(seq), None, None))
            int_ev = p._int_event("sess")
            int_ev.set()
            await p._tts_worker(q, "sess", int_ev)
            return p.tts.synthesized

        synthesized = asyncio.run(run())
        assert synthesized == []

    def test_stop_tts_stops_worker(self) -> None:
        async def run() -> list[str]:
            p = _make_pipeline()
            q: asyncio.PriorityQueue = asyncio.PriorityQueue()
            seq = itertools.count()
            q.put_nowait((1, next(seq), "Drop me.", None))
            stop = asyncio.Event()
            stop.set()
            await p._tts_worker(q, "sess", stop)
            return p.tts.synthesized

        synthesized = asyncio.run(run())
        assert synthesized == []


class FakeSTTText:
    async def transcribe(self, audio_blob: bytes) -> str:
        return "hello world"


class FakeLLMText:
    async def generate_stream(self, messages):
        for token in ["Hello ", "there, ", "this ", "is ", "a test."]:
            yield token


class FakeTTSStream:
    sample_rate = 16000

    def __init__(self) -> None:
        self.synthesized: list[str] = []
        self.prosody_labels: list[str | None] = []

    async def synthesize_stream(self, text_chunks, prosody=None):
        buf = ""
        async for chunk in text_chunks:
            buf += chunk
        if buf:
            self.synthesized.append(buf)
            self.prosody_labels.append(prosody.label if prosody else None)
            yield b"\x00\x00"

    async def synthesize(self, text: str, prosody=None) -> bytes:
        return b""


def _is_label_call(messages) -> bool:
    if not messages:
        return False
    first = messages[0]
    return first.get("role") == "system" and "label conversation topics" in first.get(
        "content", ""
    )


class HangSTT:
    async def transcribe(self, audio_blob: bytes) -> str:
        await asyncio.Event().wait()
        return ""


class TrackLLM:
    def __init__(self) -> None:
        self.started = 0
        self.cancelled = 0

    async def generate_stream(self, messages):
        if _is_label_call(messages):
            if False:
                yield ""
            return
        self.started += 1
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        if False:
            yield ""


class FakeLLMLabel:
    def __init__(self) -> None:
        self.calls: list[list[dict]] = []

    async def generate_stream(self, messages):
        self.calls.append(messages)
        for token in ["Latency", " tuning"]:
            yield token


class TestSegmentCleanup:
    def test_cancel_cleans_up_speculative_llm(self) -> None:
        async def run() -> None:
            p = _make_pipeline()
            p.stt = HangSTT()
            p.llm = TrackLLM()
            ctx = ConversationContext()
            ctx.last_partial_transcript = "hello world"

            task = asyncio.create_task(
                p._process_speech_segment(b"\x00" * 1600, "sess", ctx)
            )
            await asyncio.sleep(0.05)
            assert p.llm.started == 1
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.sleep(0.05)
            assert p.llm.cancelled == 1
            assert "sess" not in p._current_tasks

        asyncio.run(run())


class TestBackgroundTopicLabeling:
    def test_labels_current_topic_in_background(self) -> None:
        async def run() -> str | None:
            llm = FakeLLMLabel()
            p = StreamingPipeline(FakeSTT(), llm, FakeTTS(), FakeVAD())
            tracker = p._topic_tracker("sess")
            tracker.update("optimizing latency for streaming", 0)
            p._maybe_label_topic("sess")
            await asyncio.sleep(0.05)
            return tracker.label

        label = asyncio.run(run())
        assert label == "Latency tuning"

    def test_no_repeat_task_for_same_topic(self) -> None:
        async def run() -> tuple[int, str | None]:
            llm = FakeLLMLabel()
            p = StreamingPipeline(FakeSTT(), llm, FakeTTS(), FakeVAD())
            tracker = p._topic_tracker("sess")
            tracker.update("optimizing latency for streaming", 0)
            p._maybe_label_topic("sess")
            p._maybe_label_topic("sess")
            await asyncio.sleep(0.05)
            return len(llm.calls), tracker.label

        calls, label = asyncio.run(run())
        assert calls == 1
        assert label == "Latency tuning"

    def test_spec_reuse_emits_llm_token(self) -> None:
        async def run() -> None:
            p = _make_pipeline()
            p.stt = FakeSTTText()
            p.llm = FakeLLMText()
            p.tts = FakeTTSStream()
            ctx = ConversationContext()
            ctx.last_partial_transcript = "hello world"

            await p._process_speech_segment(b"\x00" * 1600, "sess", ctx)

            msgs: list[tuple[PipelineEvent, object]] = []
            while not p._output_queue.empty():
                msg = p._output_queue.get_nowait()
                msgs.append((msg.event, msg.data))
            tokens = [d for ev, d in msgs if ev == PipelineEvent.LLM_TOKEN]
            assert tokens, "expected an LLM_TOKEN from the speculated response"
            assert tokens[0] == "Hello there, this is a test."

        asyncio.run(run())


class TestMultiSessionIsolation:
    def test_signal_interrupt_cancels_only_target_session(self) -> None:
        async def run() -> None:
            p = _make_pipeline()
            cancelled: dict[str, bool] = {"A": False, "B": False}
            block_a, block_b = asyncio.Event(), asyncio.Event()

            async def blocker(name: str, block: asyncio.Event) -> None:
                try:
                    await block.wait()
                except asyncio.CancelledError:
                    cancelled[name] = True
                    raise

            task_a = asyncio.create_task(blocker("A", block_a))
            task_b = asyncio.create_task(blocker("B", block_b))
            p._current_tasks["A"] = task_a
            p._current_tasks["B"] = task_b

            await asyncio.sleep(0.01)
            await p.signal_interrupt("A")
            await asyncio.sleep(0.05)

            assert cancelled["A"] is True
            assert cancelled["B"] is False
            assert not task_b.done()

            task_b.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task_b

        asyncio.run(run())

    def test_signal_interrupt_unknown_session_does_not_crash(self) -> None:
        async def run() -> None:
            p = _make_pipeline()
            await p.signal_interrupt("unknown_session")
            assert "unknown_session" not in p._current_tasks

        asyncio.run(run())

    def test_process_segment_streams_and_clears_task_map(self) -> None:
        async def run() -> None:
            p = _make_pipeline()
            p.stt = FakeSTTText()
            p.llm = FakeLLMText()
            p.tts = FakeTTSStream()
            ctx = ConversationContext()

            await p._process_speech_segment(b"\x00" * 1600, "sess", ctx)

            assert "sess" not in p._current_tasks
            assert p.tts.synthesized, "expected synthesized audio chunks"

        asyncio.run(run())

    def test_new_segment_replaces_previous_task_entry(self) -> None:
        async def run() -> None:
            p = _make_pipeline()
            p.stt = FakeSTTText()
            p.llm = FakeLLMText()
            p.tts = FakeTTSStream()

            await p._process_speech_segment(
                b"\x00" * 1600, "sess", ConversationContext()
            )
            await p._process_speech_segment(
                b"\x00" * 1600, "sess", ConversationContext()
            )

            assert "sess" not in p._current_tasks
            assert len(p.tts.synthesized) == 2

        asyncio.run(run())


class FakeVADTrue:
    sample_rate = 16000

    def is_speech(self, chunk: bytes) -> bool:
        return True

    def reset(self) -> None:
        pass


async def _push_speech(p: StreamingPipeline, n: int) -> None:
    for _ in range(n):
        await p.push_audio(b"\x7f" * FRAME_BYTES, "sess")


def _const_energy_chunk(value: int) -> bytes:
    return struct.pack("<h", value) * (FRAME_BYTES // 2)


class TestPlaybackActive:
    def test_tts_worker_marks_playback_active_then_clears(self) -> None:
        async def run() -> None:
            p = _make_pipeline()
            q: asyncio.PriorityQueue = asyncio.PriorityQueue()
            seq = itertools.count()
            q.put_nowait((1, next(seq), "Say something.", None))
            q.put_nowait((2, next(seq), None, None))
            await p._tts_worker(q, "sess", asyncio.Event())
            assert p._playback_active.get("sess") is True
            await asyncio.sleep(0.7)
            assert p._playback_active.get("sess") is False

        asyncio.run(run())

    def test_onset_barges_in_when_client_still_playing(self) -> None:
        async def run() -> None:
            p = _make_pipeline()
            p.vad = FakeVADTrue()
            p._playback_active["sess"] = True
            p._ctx("sess").dialogue_state = DialogueState.IDLE

            p._running = True
            loop_task = asyncio.create_task(p._pipeline_loop())
            seen: list[tuple[PipelineEvent, str]] = []

            async def collect() -> None:
                async for msg in p.output_stream():
                    seen.append((msg.event, msg.session_id))

            col_task = asyncio.create_task(collect())
            await _push_speech(p, 4)
            await asyncio.sleep(0.2)
            p._running = False
            loop_task.cancel()
            await asyncio.gather(loop_task, col_task, return_exceptions=True)

            assert (PipelineEvent.INTERRUPT, "sess") in seen
            assert p._playback_active.get("sess") is False

        asyncio.run(run())

    def test_no_interrupt_when_idle_and_not_playing(self) -> None:
        async def run() -> None:
            p = _make_pipeline()
            p.vad = FakeVADTrue()
            p._ctx("sess").dialogue_state = DialogueState.IDLE

            p._running = True
            loop_task = asyncio.create_task(p._pipeline_loop())
            seen: list[tuple[PipelineEvent, str]] = []

            async def collect() -> None:
                async for msg in p.output_stream():
                    seen.append((msg.event, msg.session_id))

            col_task = asyncio.create_task(collect())
            await _push_speech(p, 4)
            await asyncio.sleep(0.2)
            p._running = False
            loop_task.cancel()
            await asyncio.gather(loop_task, col_task, return_exceptions=True)

            assert (PipelineEvent.INTERRUPT, "sess") not in seen

        asyncio.run(run())

    def test_worker_cancel_still_schedules_playback_clear(self) -> None:
        async def run() -> None:
            p = _make_pipeline()
            q: asyncio.PriorityQueue = asyncio.PriorityQueue()
            seq = itertools.count()
            q.put_nowait((1, next(seq), "Played already.", None))
            worker = asyncio.create_task(p._tts_worker(q, "sess", asyncio.Event()))
            await asyncio.sleep(0.3)
            worker.cancel()
            with pytest.raises(asyncio.CancelledError):
                await worker
            assert p._playback_active.get("sess") is True
            assert "sess" in p._playback_clear_tasks
            await asyncio.sleep(0.7)
            assert p._playback_active.get("sess") is False

        asyncio.run(run())

    def test_interrupt_clears_playback_flag_and_schedule(self) -> None:
        async def run() -> None:
            p = _make_pipeline()
            p._playback_active["sess"] = True
            clear = asyncio.create_task(asyncio.sleep(10))
            p._playback_clear_tasks["sess"] = clear
            await p.signal_interrupt("sess")
            await asyncio.sleep(0)
            assert p._playback_active.get("sess") is False
            assert "sess" not in p._playback_clear_tasks
            assert clear.cancelled()

        asyncio.run(run())

    def test_echo_within_grace_does_not_barge_in(self) -> None:
        async def run() -> None:
            p = _make_pipeline()
            p.vad = FakeVADTrue()
            p._playback_active["sess"] = True
            p._playback_onset["sess"] = time.monotonic()
            p._echo_floor["sess"] = 0.15
            p._ctx("sess").dialogue_state = DialogueState.IDLE

            p._running = True
            loop_task = asyncio.create_task(p._pipeline_loop())
            seen: list[tuple[PipelineEvent, str]] = []

            async def collect() -> None:
                async for msg in p.output_stream():
                    seen.append((msg.event, msg.session_id))

            col_task = asyncio.create_task(collect())
            await _push_speech(p, 4)
            await asyncio.sleep(0.2)
            p._running = False
            loop_task.cancel()
            await asyncio.gather(loop_task, col_task, return_exceptions=True)

            assert (PipelineEvent.INTERRUPT, "sess") not in seen
            assert p._playback_active.get("sess") is True

        asyncio.run(run())

    def test_echo_floor_suppresses_own_voice_after_grace(self) -> None:
        async def run() -> None:
            p = _make_pipeline()
            p.vad = FakeVADTrue()
            p._playback_active["sess"] = True
            p._playback_onset["sess"] = time.monotonic() - 5.0
            p._echo_floor["sess"] = 0.15
            p._ctx("sess").dialogue_state = DialogueState.IDLE

            p._running = True
            loop_task = asyncio.create_task(p._pipeline_loop())
            seen: list[tuple[PipelineEvent, str]] = []

            async def collect() -> None:
                async for msg in p.output_stream():
                    seen.append((msg.event, msg.session_id))

            col_task = asyncio.create_task(collect())
            for _ in range(4):
                await p.push_audio(_const_energy_chunk(4000), "sess")
            await asyncio.sleep(0.2)
            p._running = False
            loop_task.cancel()
            await asyncio.gather(loop_task, col_task, return_exceptions=True)

            assert (PipelineEvent.INTERRUPT, "sess") not in seen

        asyncio.run(run())

    def test_louder_than_echo_barges_in_after_grace(self) -> None:
        async def run() -> None:
            p = _make_pipeline()
            p.vad = FakeVADTrue()
            p._playback_active["sess"] = True
            p._playback_onset["sess"] = time.monotonic() - 5.0
            p._echo_floor["sess"] = 0.15
            p._ctx("sess").dialogue_state = DialogueState.IDLE

            p._running = True
            loop_task = asyncio.create_task(p._pipeline_loop())
            seen: list[tuple[PipelineEvent, str]] = []

            async def collect() -> None:
                async for msg in p.output_stream():
                    seen.append((msg.event, msg.session_id))

            col_task = asyncio.create_task(collect())
            for _ in range(3):
                await p.push_audio(_const_energy_chunk(4000), "sess")
            for _ in range(4):
                await p.push_audio(_const_energy_chunk(32639), "sess")
            await asyncio.sleep(0.2)
            p._running = False
            loop_task.cancel()
            await asyncio.gather(loop_task, col_task, return_exceptions=True)

            assert (PipelineEvent.INTERRUPT, "sess") in seen

        asyncio.run(run())

    def test_barge_in_state_is_per_session(self) -> None:
        async def run() -> None:
            p = _make_pipeline()
            p.vad = FakeVADTrue()
            p._playback_active["A"] = True
            p._playback_onset["A"] = time.monotonic()
            p._echo_floor["A"] = 0.15
            p._playback_active["B"] = True
            p._playback_onset["B"] = time.monotonic() - 5.0
            p._echo_floor["B"] = 0.15

            p._running = True
            loop_task = asyncio.create_task(p._pipeline_loop())
            seen: list[tuple[PipelineEvent, str]] = []

            async def collect() -> None:
                async for msg in p.output_stream():
                    seen.append((msg.event, msg.session_id))

            col_task = asyncio.create_task(collect())
            for _ in range(4):
                await p.push_audio(_const_energy_chunk(32639), "A")
                await p.push_audio(_const_energy_chunk(32639), "B")
            await asyncio.sleep(0.2)
            p._running = False
            loop_task.cancel()
            await asyncio.gather(loop_task, col_task, return_exceptions=True)

            assert (PipelineEvent.INTERRUPT, "A") not in seen
            assert (PipelineEvent.INTERRUPT, "B") in seen

        asyncio.run(run())


class FakeToolLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def generate_stream(self, messages):
        self.calls += 1
        if self.calls == 1:
            for token in ["Let me ", "check: ", "{tool:calculate(2+2)}"]:
                yield token
        else:
            for token in ["The result is ", "four."]:
                yield token


class TestToolCallStreaming:
    def test_tool_followup_is_streamed_to_tts(self) -> None:
        async def run() -> None:
            p = _make_pipeline()
            p.stt = FakeSTTText()
            llm = FakeToolLLM()
            p.llm = llm
            p.tts = FakeTTSStream()
            ctx = ConversationContext()

            await p._process_speech_segment(b"\x00" * 1600, "sess", ctx)

            assert llm.calls == 2
            assert p.tts.synthesized == ["The result is four."]
            assert "sess" not in p._current_tasks

        asyncio.run(run())


class TestBackchannelTiming:
    def test_timer_emits_backchannel_after_delay(self) -> None:
        async def run() -> None:
            p = _make_pipeline()
            q: asyncio.PriorityQueue = asyncio.PriorityQueue()
            seq = itertools.count()
            int_ev = asyncio.Event()

            await p._backchannel_timer(
                q, p._ctx("sess"), "sess", int_ev, seq, delay=0.05
            )

            assert not q.empty()
            prio, _, text, _prosody = q.get_nowait()
            assert prio == 0
            assert text in BACKCHANNEL_CANDIDATES["thinking"]

        asyncio.run(run())

    def test_timer_suppressed_when_interrupted(self) -> None:
        async def run() -> None:
            p = _make_pipeline()
            q: asyncio.PriorityQueue = asyncio.PriorityQueue()
            seq = itertools.count()
            int_ev = asyncio.Event()
            int_ev.set()

            await p._backchannel_timer(
                q, p._ctx("sess"), "sess", int_ev, seq, delay=0.05
            )

            assert q.empty()

        asyncio.run(run())


class FirstBlockThenStreamLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.cancelled = 0

    async def generate_stream(self, messages):
        if _is_label_call(messages):
            for token in ["Latency ", "tuning"]:
                yield token
            return
        self.calls += 1
        if self.calls == 1:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
        else:
            for token in ["Done. "]:
                yield token


class TestInterruptDuringThinking:
    def test_signal_interrupt_cancels_inflight_llm(self) -> None:
        async def run() -> None:
            p = _make_pipeline()
            p.turn_timing = TurnTiming(base_delay=0.0, min_delay=0.0, max_delay=0.0)
            p.stt = FakeSTTText()
            llm = TrackLLM()
            p.llm = llm
            p.tts = FakeTTSStream()

            task = asyncio.create_task(
                p._process_speech_segment(b"\x00" * 1600, "sess", ConversationContext())
            )
            await asyncio.sleep(0.05)
            assert llm.started == 1

            await p.signal_interrupt("sess")
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.sleep(0.05)

            assert llm.cancelled == 1
            assert "sess" not in p._current_tasks
            assert p.tts.synthesized == []

        asyncio.run(run())

    def test_new_segment_cancels_previous_turn(self) -> None:
        async def run() -> None:
            p = _make_pipeline()
            p.turn_timing = TurnTiming(base_delay=0.0, min_delay=0.0, max_delay=0.0)
            p.stt = FakeSTTText()
            llm = FirstBlockThenStreamLLM()
            p.llm = llm
            p.tts = FakeTTSStream()

            first = asyncio.create_task(
                p._process_speech_segment(b"\x00" * 1600, "sess", ConversationContext())
            )
            await asyncio.sleep(0.05)
            assert llm.calls == 1

            second = asyncio.create_task(
                p._process_speech_segment(b"\x00" * 1600, "sess", ConversationContext())
            )
            await asyncio.sleep(0.05)

            assert first.cancelled()
            assert llm.cancelled == 1
            assert llm.calls == 2
            await second

            msgs: list[PipelineEvent] = []
            while not p._output_queue.empty():
                msgs.append(p._output_queue.get_nowait().event)
            assert PipelineEvent.INTERRUPT in msgs
            assert "sess" not in p._current_tasks

        asyncio.run(run())


class TestFrameSegmentation:
    def test_large_chunk_segmented_into_fixed_frames(self) -> None:
        async def run() -> None:
            p = _make_pipeline()
            await p.push_audio(b"\x00" * (FRAME_BYTES * 4 + 100), "sess")
            frames = 0
            while not p._audio_queue.empty():
                msg = p._audio_queue.get_nowait()
                assert len(msg.data) == FRAME_BYTES
                frames += 1
            assert frames == 4
            assert len(p._frame_buffers["sess"]) == 100
            await p.push_audio(b"\x00" * (FRAME_BYTES - 100), "sess")
            assert p._audio_queue.qsize() == 1
            msg = p._audio_queue.get_nowait()
            assert len(msg.data) == FRAME_BYTES
            assert len(p._frame_buffers["sess"]) == 0

        asyncio.run(run())

    def test_unregister_drops_partial_frame(self) -> None:
        async def run() -> None:
            p = _make_pipeline()
            await p.push_audio(b"\x00" * 2000, "sess")
            assert len(p._frame_buffers["sess"]) == 2000
            p.unregister_session("sess")
            assert "sess" not in p._frame_buffers

        asyncio.run(run())


class TestLowEnergySilenceBackstop:
    def test_swallowed_trailing_silence_ends_turn(self) -> None:
        async def run() -> None:
            p = _make_pipeline()
            p.vad = FakeVADTrue()
            p._running = True
            loop_task = asyncio.create_task(p._pipeline_loop())
            seen: list[tuple[PipelineEvent, str]] = []

            async def collect() -> None:
                async for msg in p.output_stream():
                    seen.append((msg.event, msg.session_id))

            col_task = asyncio.create_task(collect())
            for _ in range(4):
                await p.push_audio(_const_energy_chunk(32639), "sess")
            for _ in range(4):
                await p.push_audio(b"\x00" * FRAME_BYTES, "sess")
            await asyncio.sleep(0.3)
            p._running = False
            loop_task.cancel()
            await asyncio.gather(loop_task, col_task, return_exceptions=True)

            assert (PipelineEvent.SPEECH_START, "sess") in seen
            assert (PipelineEvent.SPEECH_END, "sess") in seen
            assert p._speaking.get("sess") is False

        asyncio.run(run())

    def test_real_energy_speech_not_flagged_as_silence(self) -> None:
        async def run() -> None:
            p = _make_pipeline()
            p.vad = FakeVADTrue()
            p._running = True
            loop_task = asyncio.create_task(p._pipeline_loop())
            seen: list[tuple[PipelineEvent, str]] = []

            async def collect() -> None:
                async for msg in p.output_stream():
                    seen.append((msg.event, msg.session_id))

            col_task = asyncio.create_task(collect())
            for _ in range(8):
                await p.push_audio(_const_energy_chunk(32639), "sess")
            await asyncio.sleep(0.3)
            p._running = False
            loop_task.cancel()
            await asyncio.gather(loop_task, col_task, return_exceptions=True)

            assert (PipelineEvent.SPEECH_START, "sess") in seen
            assert (PipelineEvent.SPEECH_END, "sess") not in seen
            assert p._speaking.get("sess") is True

        asyncio.run(run())
