import argparse
import asyncio
import json
import sys
import wave
from difflib import unified_diff
from pathlib import Path
from time import perf_counter, time
from uuid import uuid4

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from livekit import rtc
from livekit.agents import APIError, utils

from src.agent.graph import build_graph
from src.api.telemetry import RunTelemetry

from .worker import answer_voice, speech_models


def read_wav(path):
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2 or not wav.getnframes():
            raise ValueError("Use a nonempty 16-bit PCM WAV file.")
        channels = wav.getnchannels()
        pcm = wav.readframes(wav.getnframes())
        frames, partial = divmod(len(pcm), channels * 2)
        if not frames or partial:
            raise ValueError("The WAV file has missing or incomplete audio frames.")
        return rtc.AudioFrame(pcm, wav.getframerate(), channels, frames)


async def replay(path):
    audio = read_wav(path)
    graph = await build_graph(MemorySaver())
    run = RunTelemetry(operation="replay", thread_id=f"replay-{uuid4().hex}")
    async with utils.http_context.open():
        stt, tts = speech_models()
        async with stt, tts:
            started = perf_counter()
            recognized = await stt.recognize(audio)
            run.timing("asr", (perf_counter() - started) * 1000)
            run.timestamps_ms["stt_final_ms"] = round(time() * 1000, 1)
            transcript = recognized.alternatives[0].text.strip() if recognized.alternatives else ""
            if not transcript:
                raise ValueError("The WAV file produced no transcript.")
            answer = await answer_voice(graph, transcript, run)

            tts_started = perf_counter()
            async with tts.synthesize(answer) as stream:
                async for _ in stream:
                    first_audio = perf_counter()
                    if "tts_ttfb" not in run.timings_ms:
                        run.timing("tts_ttfb", (first_audio - tts_started) * 1000)
                        run.timing("total_e2e", (first_audio - started) * 1000)
            if "tts_ttfb" not in run.timings_ms:
                raise ValueError("The reply produced no audio.")
    timings = {name: round(sum(samples), 1) for name, samples in run.timings_ms.items()}
    if "llm_ttft" in timings:
        timings["llm_ttft"] = run.timings_ms["llm_ttft"][0]
    return {"transcript": transcript, "reply": answer, "status": run.status, "timings_ms": timings}


def print_result(result, baseline=None):
    print(f"ASR transcript: {result['transcript']}")
    print(f"Agent reply:    {result['reply']}")
    print(f"Status:         {result['status']}")
    print("Component       Latency (ms)")
    for name in ("asr", "llm_ttft", "tool_exec", "tts_ttfb", "total_e2e"):
        value = result["timings_ms"].get(name)
        timing = f"{value:.1f}" if value is not None else "n/a"
        if baseline is not None:
            previous = baseline["timings_ms"].get(name)
            if value is not None and previous is not None:
                timing += f" ({value - previous:+.1f})"
        print(f"{name:<15} {timing}")
    if baseline is not None:
        for line in unified_diff([baseline["transcript"]], [result["transcript"]], "baseline", "replay", lineterm=""):
            print(line)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Replay a WAV through Nova-3, LangGraph, and Aura.")
    parser.add_argument("--file", required=True, type=Path)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--json", action="store_true", help="Print JSON for a later baseline comparison.")
    output.add_argument("--baseline", type=Path, help="Compare timings and transcript against a JSON replay.")
    args = parser.parse_args(argv)
    load_dotenv()
    try:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline else None
        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
            result = runner.run(replay(args.file))
        if args.json:
            print(json.dumps(result, ensure_ascii=False))
        else:
            print_result(result, baseline)
    except (APIError, OSError, ValueError, wave.Error, EOFError, KeyError, TypeError) as error:
        detail = str(error) if isinstance(error, ValueError) else type(error).__name__
        print(f"Replay failed: {detail}", file=sys.stderr)
        return 1
    return int(result["status"] == "error")


if __name__ == "__main__":
    sys.exit(main())
