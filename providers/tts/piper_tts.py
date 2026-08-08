from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import AsyncGenerator

import piper
import numpy as np

from providers.tts.base import TTSProvider
from modules.tts.prosody import (
    ProsodyProfile,
    comma_fractions,
    comma_pause,
    pause_for,
    splice_audio,
    split_emphasis,
)


class PiperTTS(TTSProvider):
    def __init__(
        self,
        model_path: str,
        model_config_path: str | None = None,
        sentence_silence: float = 0.02,
        length_scale: float = 0.65,
        noise_scale: float = 0.4,
        noise_w: float = 0.5,
    ) -> None:
        self.model_path = Path(model_path)
        self.model_config_path = (
            Path(model_config_path) if model_config_path else None
        )
        self.sentence_silence = sentence_silence
        self._syn_config = piper.SynthesisConfig(
            length_scale=length_scale,
            noise_scale=noise_scale,
            noise_w_scale=noise_w,
        )
        self._voice = None
        self._sample_rate: int = 22050
        self._warm_up()

    def _warm_up(self) -> None:
        _ = self.voice

    @property
    def voice(self) -> piper.PiperVoice:
        if self._voice is None:
            self._voice = piper.PiperVoice.load(
                self.model_path,
                config_path=self.model_config_path,
                use_cuda=True,
            )
            self._sample_rate = self._voice.config.sample_rate
        return self._voice

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def _config_for(self, prosody: ProsodyProfile | None) -> piper.SynthesisConfig:
        if prosody is None:
            return self._syn_config
        return piper.SynthesisConfig(
            length_scale=prosody.length_scale,
            noise_scale=prosody.noise_scale,
            noise_w_scale=prosody.noise_w,
        )

    def _sentence_silence(self, prosody: ProsodyProfile | None) -> float:
        if prosody is None:
            return self.sentence_silence
        return prosody.sentence_silence

    def _silence_pad(self, seconds: float) -> bytes:
        if seconds <= 0:
            return b""
        count = int(seconds * self.sample_rate) * 2
        return b"\x00\x00" * count

    def _split_sentences(self, text: str) -> list[tuple[str, str]]:
        """Return [(sentence, terminator)] with ellipsis-aware boundaries."""
        out: list[tuple[str, str]] = []
        cur = ""
        i = 0
        n = len(text)
        while i < n:
            if text.startswith("...", i) or text[i] == "…":
                if cur.strip():
                    out.append((cur.strip(), "..."))
                cur = ""
                i += 3 if text.startswith("...", i) else 1
                while i < n and text[i].isspace():
                    i += 1
                continue
            c = text[i]
            cur += c
            if c in ".!?":
                j = i + 1
                while j < n and text[j] in "'\")\u201d\u2019]":
                    cur += text[j]
                    j += 1
                if j >= n or text[j].isspace():
                    if cur.strip():
                        out.append((cur.strip(), c))
                    cur = ""
                    i = j
                    while i < n and text[i].isspace():
                        i += 1
                    continue
            i += 1
        if cur.strip():
            out.append((cur.strip(), ""))
        return out

    def _synthesize_sentence_chunks(
        self, sentence: str, syn_config: piper.SynthesisConfig
    ) -> list[bytes]:
        segments = split_emphasis(sentence)
        if len(segments) <= 1:
            chunks = self.voice.synthesize(sentence, syn_config=syn_config)
            return [bytes(c.audio_int16_bytes) for c in chunks]
        out: list[bytes] = []
        base_length = syn_config.length_scale or 1.0
        base_noise = syn_config.noise_scale or 0.5
        for seg, emphasized in segments:
            if not seg.strip():
                continue
            cfg = syn_config
            if emphasized:
                cfg = replace(
                    syn_config,
                    length_scale=base_length * 0.85,
                    noise_scale=min(0.7, base_noise + 0.05),
                    volume=1.1,
                )
            for c in self.voice.synthesize(seg, syn_config=cfg):
                out.append(bytes(c.audio_int16_bytes))
        return out

    def _assemble_sentence(
        self,
        sentence: str,
        syn_config: piper.SynthesisConfig,
        comma_pad_seconds: float,
    ) -> list[bytes]:
        raw = self._synthesize_sentence_chunks(sentence, syn_config)
        full = b"".join(raw)
        if not full:
            return []
        pad = self._silence_pad(comma_pad_seconds)
        return splice_audio(
            full, comma_fractions(sentence), self.sample_rate, pad
        )

    def _pause_after(self, terminator: str, prosody: ProsodyProfile | None) -> bytes:
        return self._silence_pad(
            pause_for(terminator, self._sentence_silence(prosody))
        )

    async def synthesize_stream(
        self,
        text_chunks: AsyncGenerator[str, None],
        prosody: ProsodyProfile | None = None,
    ) -> AsyncGenerator[bytes, None]:
        syn_config = self._config_for(prosody)
        comma_pad_seconds = comma_pause(self._sentence_silence(prosody))
        buffer = ""
        async for chunk in text_chunks:
            buffer += chunk
            sentences = self._split_sentences(buffer)
            if not sentences:
                continue
            for sentence, terminator in sentences[:-1]:
                pieces = await asyncio.to_thread(
                    self._assemble_sentence,
                    sentence,
                    syn_config,
                    comma_pad_seconds,
                )
                for audio_chunk in pieces:
                    yield audio_chunk
                    await asyncio.sleep(0)
                pad = self._pause_after(terminator, prosody)
                if pad:
                    yield pad
            buffer = sentences[-1][0]

        if buffer.strip():
            pieces = await asyncio.to_thread(
                self._assemble_sentence,
                buffer.strip(),
                syn_config,
                comma_pad_seconds,
            )
            for audio_chunk in pieces:
                yield audio_chunk

    async def synthesize(
        self, text: str, prosody: ProsodyProfile | None = None
    ) -> bytes:
        syn_config = self._config_for(prosody)
        comma_pad_seconds = comma_pause(self._sentence_silence(prosody))
        audio = bytearray()
        sentences = self._split_sentences(text)
        for i, (sentence, terminator) in enumerate(sentences):
            pieces = await asyncio.to_thread(
                self._assemble_sentence,
                sentence,
                syn_config,
                comma_pad_seconds,
            )
            for piece in pieces:
                audio.extend(piece)
            if i < len(sentences) - 1:
                audio.extend(self._pause_after(terminator, prosody))
        return bytes(audio)