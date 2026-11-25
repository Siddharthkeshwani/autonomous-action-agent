import asyncio
import json
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from livekit.agents import APIConnectOptions, llm

from services.mcp_server import server
from src.agent.graph import build_graph
from src.agent.prompts import VOICE
from src.api import telemetry
from src.api.main import create_app
from src.voice import replay, worker


SAMPLE = Path("evaluation/audio_samples/check_upi.wav")
TRANSCRIPT = "How long does it take for a failed UPI payment to be reversed?"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.delenv("UPSTASH_REDIS_URL", raising=False)
    monkeypatch.setattr(replay, "load_dotenv", Mock())


def test_format_speech():
    answer = (
        "**Three cases** are open. See [the policy](https://example.com/policy). A third sentence.\n"
        "Count | Status\n--- | ---\n3 | Open\n"
        "```sql\nSELECT * FROM cases;\n```\n"
        "~~~python\nprint('hello')\n~~~\n"
    )
    assert worker.spoken_text(answer) == "Three cases are open. See the policy."
    assert worker.spoken_text("See the UPI dispute SLA (upi_dispute_sla.md).") == "See the UPI dispute SLA."
    assert worker.spoken_text("```sql\nSELECT * FROM cases;") == "See the details on screen."
    assert len(worker.spoken_text("Long answer " * 80)) <= 260


def test_worker_starts_audio_session_with_silero_and_barge_in(monkeypatch):
    vad, stt, tts = Mock(), Mock(), Mock()
    proc = SimpleNamespace(userdata={})
    load_vad = Mock(return_value=vad)
    monkeypatch.setattr(worker.silero.VAD, "load", load_vad)
    monkeypatch.setattr(worker.sys, "platform", "linux")
    monkeypatch.setattr(worker.deepgram, "STT", Mock(return_value=stt))
    monkeypatch.setattr(worker.deepgram, "TTS", Mock(return_value=tts))
    session = Mock(start=AsyncMock(), aclose=AsyncMock())
    factory = Mock(return_value=session)
    monkeypatch.setattr(worker, "AgentSession", factory)

    @asynccontextmanager
    async def graph_context():
        yield Mock()

    monkeypatch.setattr(worker, "open_graph", graph_context)
    ctx = Mock(proc=proc, connect=AsyncMock(), wait_for_participant=AsyncMock(return_value=Mock(identity="caller")))

    async def run():
        worker.setup(proc)
        await worker.entrypoint(ctx)
        await ctx.add_shutdown_callback.call_args.args[0]()

    asyncio.run(run())
    load_vad.assert_called_once_with(min_silence_duration=0.95, min_speech_duration=0.15)
    worker.deepgram.STT.assert_called_once_with(model="nova-3", keyterm=["UPI"], endpointing_ms=900)
    worker.deepgram.TTS.assert_called_once_with(model="aura-2-asteria-en")
    ctx.connect.assert_awaited_once_with(auto_subscribe=worker.AutoSubscribe.AUDIO_ONLY)
    assert factory.call_args.kwargs["vad"] is vad
    options = factory.call_args.kwargs["turn_handling"]
    assert options["turn_detection"] == "vad"
    assert options["endpointing"] == {"min_delay": 1.1}
    assert options["interruption"] == {"enabled": True, "mode": "vad", "min_duration": 0.6}
    assert options["preemptive_generation"] == {"enabled": False}
    assert session.start.call_args.kwargs["room_options"].participant_identity == "caller"
    assert session.start.call_args.kwargs["room_options"].text_output is False
    assert {call.args[0] for call in session.on.call_args_list} == {"user_input_transcribed", "speech_created"}
    session.aclose.assert_awaited_once()


def packets(participant):
    for call in participant.publish_data.await_args_list:
        assert call.kwargs == {"topic": worker.TOPIC, "reliable": True, "destination_identities": ["caller"]}
    return [json.loads(call.args[0]) for call in participant.publish_data.await_args_list]


async def voice_turn(agent, text, *, interrupted=False, speech=None, final_segment=None):
    user = llm.ChatMessage(role="user", content=[text], metrics={"stopped_speaking_at": 10.0})
    agent.transcribed(SimpleNamespace(is_final=True, transcript=final_segment or text, created_at=10.075))
    await agent.on_user_turn_completed(llm.ChatContext(), user)
    turn_id = agent.turns[user.id].turn_id
    handle = Mock(interrupted=interrupted, chat_items=[speech] if speech else [])
    handle.exception.return_value = None
    agent.speech_created(SimpleNamespace(source="generate_reply", speech_handle=handle))
    context = llm.ChatContext(items=[user])
    async with agent.llm.chat(chat_ctx=context, conn_options=APIConnectOptions()) as stream:
        replies = [chunk.delta.content async for chunk in stream]
    handle.add_done_callback.call_args.args[0](handle)
    await agent.drain()
    return turn_id, replies


def test_voice_turn_sends_transcripts_steps_and_latency(monkeypatch):
    participant = Mock(publish_data=AsyncMock())
    store = AsyncMock()
    monkeypatch.setattr(worker, "store_voice_run", store)
    db = Mock()
    db.execute.return_value.fetchall.return_value = [{"count": 3}]

    @contextmanager
    def readonly_db():
        yield db

    monkeypatch.setattr(server, "readonly_db", readonly_db)

    async def run():
        model = FakeListChatModel(responses=["**Policy:** Escalate after three business days."])

        async def reason(messages):
            assert VOICE in messages[0].content
            if messages[-1].type == "human":
                return AIMessage(content="", tool_calls=[{
                    "name": "query_readonly_sql", "id": "count", "args": {"sql": "SELECT 3 AS count"},
                }])
            return await model.ainvoke(messages)

        graph = await build_graph(MemorySaver(), route=AsyncMock(return_value="faq"), reason=reason)
        agent = worker.VoiceAgent(graph, participant, "caller")
        speech = llm.ChatMessage(role="assistant", content=["Policy: Escalate"], interrupted=True,
                                 metrics={"tts_node_ttfb": .115, "e2e_latency": .92})
        turn, replies = await voice_turn(agent, TRANSCRIPT, interrupted=True, speech=speech)
        assert replies == ["Policy: Escalate after three business days."]
        for packet in packets(participant):
            assert (packet["thread_id"], packet["turn_id"]) == (agent.thread_id, turn)

    asyncio.run(run())
    events = [(packet["event"], packet["data"]) for packet in packets(participant)]
    assert events[1] == ("transcript", {"role": "user", "text": TRANSCRIPT, "is_final": True})
    assert ("step_status", {"step": "model", "status": "running", "message": "Thinking..."}) in events
    assert ("transcript", {"role": "assistant", "text": "Policy: Escalate", "is_final": True, "interrupted": True}) in events
    assert events[-1][0] == "latency"
    run = store.await_args.args[0]
    assert events[-1][1] == run.model_dump()
    assert run.status == "interrupted"
    assert run.timestamps_ms == {"vad_end_ms": 10000.0, "stt_final_ms": 10075.0}
    assert run.timings_ms["asr"] == [75.0]
    assert run.timings_ms["llm_ttft"][0] >= 0
    assert run.timings_ms["tool_exec"] == run.timings_ms["sql"]
    assert run.tool_counts == {"query_readonly_sql": 1}
    assert run.timings_ms["tts_ttfb"] == [115.0]
    assert run.timings_ms["total_e2e"] == [920.0]


def test_voice_keeps_stt_segments_in_one_turn(monkeypatch):
    participant = Mock(publish_data=AsyncMock())
    answer = AsyncMock(return_value="Check with Payments Ops.")
    monkeypatch.setattr(worker, "answer_voice", answer)
    monkeypatch.setattr(worker, "store_voice_run", AsyncMock())

    async def run():
        agent = worker.VoiceAgent(Mock(), participant, "caller")
        agent.transcribed(SimpleNamespace(is_final=False, transcript="How long", created_at=8.0))
        agent.transcribed(SimpleNamespace(is_final=True, transcript="How long does it take", created_at=9.0))
        await agent.drain()
        answer.assert_not_awaited()
        turn_id, replies = await voice_turn(agent, TRANSCRIPT, final_segment="for a failed UPI payment to be reversed?")
        assert replies == [answer.return_value]
        answer.assert_awaited_once()
        assert answer.await_args.args[1] == TRANSCRIPT
        captions = [packet for packet in packets(participant) if packet["event"] == "transcript"]
        assert {packet["turn_id"] for packet in captions} == {turn_id}
        assert [(packet["data"]["text"], packet["data"]["is_final"]) for packet in captions] == [
            ("How long", False), ("How long does it take", False), (TRANSCRIPT, False), (TRANSCRIPT, True),
        ]

    asyncio.run(run())


def test_voice_requests_approval_without_executing_the_write(monkeypatch):
    business = AsyncMock()
    monkeypatch.setattr(server, "business_request", business)
    monkeypatch.setattr(worker, "store_voice_run", AsyncMock())
    participant = Mock(publish_data=AsyncMock())
    reason = AsyncMock(return_value=AIMessage(content="", tool_calls=[{
        "name": "initiate_refund", "id": "refund-1", "args": {"refund": {
            "order_id": 1, "amount_inr": "25", "reason": "Return", "idempotency_key": "refund-1",
        }},
    }]))

    async def run():
        graph = await build_graph(MemorySaver(), route=AsyncMock(return_value="action"), reason=reason)
        agent = worker.VoiceAgent(graph, participant, "caller")
        _, first = await voice_turn(agent, "Refund order one for 25 rupees.")
        _, second = await voice_turn(agent, "Yes, go ahead.")
        assert first == second == [worker.APPROVAL_REPLY]

    asyncio.run(run())
    approvals = [packet["data"] for packet in packets(participant) if packet["event"] == "hil_required"]
    assert len(approvals) == 2 and approvals[0] == approvals[1]
    assert approvals[0]["interrupt_id"] and approvals[0]["thread_id"]
    assert approvals[0]["payload"]["amount_inr"] == "25"
    reason.assert_awaited_once()
    business.assert_not_awaited()


def test_large_chart_uses_a_text_stream():
    participant = Mock(publish_data=AsyncMock(), send_text=AsyncMock())
    run = telemetry.RunTelemetry(operation="voice", thread_id="thread", turn_id="turn")
    data = {"image_base64": "A" * worker.PACKET_LIMIT, "mime_type": "image/png"}
    asyncio.run(worker.publish_event(participant, "caller", run, "chart_artifact", data))
    participant.publish_data.assert_not_awaited()
    args, kwargs = participant.send_text.await_args
    assert kwargs == {"topic": worker.TOPIC, "destination_identities": ["caller"]}
    assert json.loads(args[0]) == {"event": "chart_artifact", "thread_id": "thread", "turn_id": "turn", "data": data}


def test_worker_metrics_are_visible_in_gateway(monkeypatch):
    monkeypatch.setenv("UPSTASH_REDIS_URL", "redis://unused")
    client, pipe = AsyncMock(), AsyncMock()
    client.__aenter__.return_value = client
    client.pipeline = Mock(return_value=pipe)
    pipe.__aenter__.return_value = pipe
    pipe.lpush = Mock(return_value=pipe)
    pipe.ltrim = Mock(return_value=pipe)
    monkeypatch.setattr(telemetry.Redis, "from_url", Mock(return_value=client))

    run = telemetry.RunTelemetry(operation="voice", thread_id="thread", turn_id="turn", status="completed")
    run.timing("asr", 24.0)
    run.timing("tts_ttfb", 98.0)
    asyncio.run(telemetry.store_voice_run(run))
    pipe.lpush.assert_called_once_with(telemetry.VOICE_RUNS_KEY, run.model_dump_json())
    pipe.ltrim.assert_called_once_with(telemetry.VOICE_RUNS_KEY, 0, 99)
    pipe.execute.assert_awaited_once()

    client.lrange.return_value = [pipe.lpush.call_args.args[1]]
    with TestClient(create_app()) as api:
        metrics = api.get("/api/v1/metrics").json()
        assert metrics["runs"] == [run.model_dump()]
        assert metrics["latency"]["asr"]["p50_ms"] == 24.0
        assert metrics["latency"]["tts_ttfb"]["p50_ms"] == 98.0


def test_replay_cli_runs_the_wav_and_prints_latency(monkeypatch, capsys, tmp_path):
    stt, tts, synthesis = AsyncMock(), AsyncMock(), AsyncMock()
    stt.__aenter__.return_value = stt
    tts.__aenter__.return_value = tts
    synthesis.__aenter__.return_value = synthesis
    stt.recognize.return_value = SimpleNamespace(alternatives=[SimpleNamespace(text=TRANSCRIPT)])
    synthesis.__aiter__.return_value = [object()]
    tts.synthesize = Mock(return_value=synthesis)
    monkeypatch.setattr(replay, "speech_models", lambda: (stt, tts))

    async def fake_graph(checkpointer):
        model = FakeListChatModel(responses=["**Check with Payments Ops.**"])
        return await build_graph(checkpointer, route=AsyncMock(return_value="faq"), reason=model.ainvoke)

    monkeypatch.setattr(replay, "build_graph", fake_graph)
    assert replay.main(["--file", str(SAMPLE), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["transcript"] == TRANSCRIPT
    assert result["reply"] == "Check with Payments Ops."
    assert result["status"] == "completed"
    assert set(result["timings_ms"]) == {"asr", "llm_ttft", "tts_ttfb", "total_e2e"}
    assert stt.recognize.await_args.args[0].duration > 2
    tts.synthesize.assert_called_once_with(result["reply"])

    baseline = tmp_path / "replay.json"
    baseline.write_text(json.dumps({**result, "transcript": "What is the UPI policy?"}), encoding="utf-8")
    assert replay.main(["--file", str(SAMPLE), "--baseline", str(baseline)]) == 0
    output = capsys.readouterr().out
    assert f"ASR transcript: {TRANSCRIPT}" in output
    assert "Agent reply:    Check with Payments Ops." in output
    assert "Latency (ms)" in output and "total_e2e" in output
    assert f"+{TRANSCRIPT}" in output


def test_replay_cli_prints_diagnostics_for_an_agent_failure(monkeypatch, capsys):
    monkeypatch.setattr(replay, "replay", AsyncMock(return_value={
        "transcript": TRANSCRIPT, "reply": worker.ERROR_REPLY, "status": "error", "timings_ms": {"asr": 50},
    }))
    assert replay.main(["--file", str(SAMPLE)]) == 1
    output = capsys.readouterr().out
    assert TRANSCRIPT in output and "Status:         error" in output and "50.0" in output
