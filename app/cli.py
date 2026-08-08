from __future__ import annotations

import argparse
import asyncio
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="ResumeVoice AI real-time voice agent")
    sub = parser.add_subparsers(dest="command", required=True)

    server = sub.add_parser("server", help="Run the ResumeVoice AI WebSocket server")
    server.add_argument("--host", default="127.0.0.1", help="Bind address")
    server.add_argument("--port", type=int, default=8000, help="Bind port")

    demo = sub.add_parser("demo", help="Run a local microphone demo")
    demo.add_argument("--stt-model", default="base", help="faster-whisper model size")
    demo.add_argument(
        "--llm-url", default="http://localhost:8000/v1", help="vLLM base URL"
    )
    demo.add_argument(
        "--llm-model", default="Qwen/Qwen2.5-7B-Instruct-AWQ", help="LLM model name"
    )
    demo.add_argument(
        "--tts-model",
        default="/usr/share/piper/voices/en_US-lessac-medium.onnx",
        help="Piper TTS model path",
    )
    demo.add_argument("--vad-threshold", type=float, default=0.5, help="VAD threshold")

    args = parser.parse_args()

    if args.command == "server":
        _run_server(host=args.host, port=args.port)
    elif args.command == "demo":
        asyncio.run(_run_demo(args))


def _run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run("app.server:app", host=host, port=port, reload=False)


async def _run_demo(args: argparse.Namespace) -> None:
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError:
        print("Install sounddevice: pip install sounddevice")
        sys.exit(1)

    from core.pipeline import PipelineEvent, StreamingPipeline
    from modules.vad.silero_vad import SileroVAD
    from providers.llm.vllm_llm import VLLMProvider
    from providers.stt.faster_whisper_stt import FasterWhisperSTT
    from providers.tts.piper_tts import PiperTTS

    print("Initializing ResumeVoice AI pipeline...")
    stt = FasterWhisperSTT(model_size=args.stt_model, device="cuda")
    llm = VLLMProvider(base_url=args.llm_url, model=args.llm_model)
    tts = PiperTTS(model_path=args.tts_model)
    vad = SileroVAD(threshold=args.vad_threshold, device="cuda")
    pipeline = StreamingPipeline(stt=stt, llm=llm, tts=tts, vad=vad)
    await pipeline.start()

    SAMPLE_RATE = 16000
    BLOCK_SIZE = 800  # 50ms at 16kHz

    async def audio_callback(indata: np.ndarray, frames: int, time_info, status):
        if status:
            return
        chunk = (indata * 32767).astype(np.int16).tobytes()
        await pipeline.push_audio(chunk)

    async def print_output():
        async for msg in pipeline.output_stream():
            if msg.event == PipelineEvent.FINAL_TRANSCRIPT:
                print(f"\n[You] {msg.data}", flush=True)
            elif msg.event == PipelineEvent.LLM_DONE:
                print(f"\n[ResumeVoice AI] {msg.data}", flush=True)
            elif msg.event == PipelineEvent.SPEECH_START:
                print("\n[listening...]", end="", flush=True)
            elif msg.event == PipelineEvent.LLM_TOKEN:
                print(msg.data, end="", flush=True)

    out_task = asyncio.create_task(print_output())

    print("ResumeVoice AI ready. Speak into your microphone. Press Ctrl+C to stop.\n")

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=1,
        dtype="float32",
        callback=lambda *a: asyncio.run_coroutine_threadsafe(
            audio_callback(*a), asyncio.get_event_loop()
        ),
    )

    try:
        stream.start()
        while True:
            await asyncio.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stream.stop()
        stream.close()
        out_task.cancel()
        await pipeline.stop()


if __name__ == "__main__":
    main()
