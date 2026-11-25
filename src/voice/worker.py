import asyncio
import json
import logging
import re
import sys
from contextlib import AsyncExitStack, aclosing
from dataclasses import replace
from textwrap import shorten
from uuid import uuid4

import psycopg
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from livekit.agents import Agent, AgentServer, AgentSession, AutoSubscribe, cli, llm, room_io
from livekit.plugins import deepgram, silero
from livekit.rtc.participant import PublishDataError

from src.agent.graph import open_graph
from src.api.schemas import VOICE_AGENT_NAME
from src.api.stream import agent_events, approval_details
from src.api.telemetry import RunTelemetry, store_voice_run


log = logging.getLogger(__name__)
TOPIC = "omniops.voice"
PACKET_LIMIT = 15 * 1024
APPROVAL_REPLY = "Please review the action and confirm it on screen."
ERROR_REPLY = "I could not finish this request."


def speech_models():
    stt = deepgram.STT(model="nova-3", keyterm=["UPI"], endpointing_ms=900)
    return stt, deepgram.TTS(model="aura-2-asteria-en")


def spoken_text(answer):
    answer = re.sub(r"(?ms)^ {0,3}(`{3,}|~{3,})[^\n]*\n.*?(?:^ {0,3}\1[ \t]*$|\Z)", "", answer)
    lines = []
    for line in answer.splitlines():
        if "|" not in line and not line.startswith(("    ", "\t")):
            lines.append(re.sub(r"^\s*(?:[#>]+|[-*+]|\d+[.)])\s+", "", line))
    answer = " ".join(lines)
    answer = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", answer)
    answer = re.sub(r"<https?://[^>]+>|https?://\S+", "", answer)
    answer = re.sub(r"\s*\(`?[\w./-]+\.md`?\)", "", answer)
    answer = " ".join(answer.translate(str.maketrans("", "", "*_`~")).split())
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", answer, maxsplit=2)
    return shorten(" ".join(sentences[:2]), width=260, placeholder="…") or "See the details on screen."


async def answer_voice(graph, transcript, run, send=None):
    config = {"configurable": {"thread_id": run.thread_id}}
    try:
        snapshot = await graph.aget_state(config)
    except (psycopg.Error, TimeoutError, ValueError):
        run.status = "error"
        if send is not None:
            await send("error", {"message": "Agent checkpoints are unavailable."})
        return ERROR_REPLY
    if snapshot.interrupts:
        run.status = "awaiting_approval"
        if send is not None:
            await send("hil_required", approval_details(config, snapshot.interrupts[0]))
        return APPROVAL_REPLY

    graph_input = {"messages": [HumanMessage(content=transcript)], "channel": "voice"}
    answer = ERROR_REPLY
    async with aclosing(agent_events(graph, graph_input, config, run=run)) as events:
        async for kind, data in events:
            if send is not None:
                await send(kind, data)
            if kind == "final_answer":
                answer = spoken_text(data["answer"])
            elif kind == "hil_required":
                answer = APPROVAL_REPLY
    return answer


async def publish_event(participant, identity, run, event, data):
    payload = json.dumps({"event": event, "thread_id": run.thread_id, "turn_id": run.turn_id, "data": data},
                         ensure_ascii=False)
    packet = payload.encode()
    options = {"topic": TOPIC, "destination_identities": [identity]}
    try:
        async with asyncio.timeout(2):
            if len(packet) <= PACKET_LIMIT:
                await participant.publish_data(packet, reliable=True, **options)
            else:
                await participant.send_text(payload, **options)
    except (PublishDataError, ConnectionError, TimeoutError):
        log.warning("Could not send voice event %s.", event)


class GraphLLM(llm.LLM):
    def __init__(self, reply):
        super().__init__()
        self.reply = reply

    def chat(self, *, chat_ctx, conn_options, tools=None, **kwargs):
        # Retrying a LiveKit stream would submit the same turn to LangGraph again.
        return GraphStream(self, chat_ctx=chat_ctx, conn_options=replace(conn_options, max_retry=0), tools=tools or [])


class GraphStream(llm.LLMStream):
    async def _run(self):
        user = next(message for message in reversed(self._chat_ctx.messages()) if message.role == "user")
        reply = await self._llm.reply(user)
        self._event_ch.send_nowait(llm.ChatChunk(id=user.id, delta=llm.ChoiceDelta(role="assistant", content=reply)))


class VoiceAgent(Agent):
    def __init__(self, graph, participant, identity):
        super().__init__(instructions="Use the operations graph for each spoken turn.", llm=GraphLLM(self.reply))
        self.graph = graph
        self.participant = participant
        self.identity = identity
        self.thread_id = f"voice-{uuid4().hex}"
        self.turns = {}
        self.pending_id = None
        self.final_at = None
        self.caption_run = None
        self.caption_text = ""
        self.caption_lock = asyncio.Lock()
        self.tasks = set()
        self.graph_lock = asyncio.Lock()

    def attach(self, session):
        session.on("user_input_transcribed", self.transcribed)
        session.on("speech_created", self.speech_created)

    def transcribed(self, event):
        text = event.transcript.strip()
        if not text:
            return
        if self.caption_run is None:
            self.caption_run = RunTelemetry(operation="voice", thread_id=self.thread_id, turn_id=uuid4().hex)
        caption = f"{self.caption_text} {text}".strip()
        if event.is_final:
            self.caption_text = caption
            self.final_at = event.created_at
        task = asyncio.create_task(self.user_caption(self.caption_run, caption, is_final=False))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def user_caption(self, run, text, *, is_final):
        async with self.caption_lock:
            await publish_event(self.participant, self.identity, run, "transcript", {
                "role": "user", "text": text, "is_final": is_final,
            })

    async def on_user_turn_completed(self, turn_ctx, new_message):
        turn_id = new_message.id
        run = self.caption_run or RunTelemetry(operation="voice", thread_id=self.thread_id, turn_id=turn_id)
        self.caption_run = None
        self.caption_text = ""
        vad_end = new_message.metrics.get("stopped_speaking_at")
        if vad_end is not None:
            run.timestamps_ms["vad_end_ms"] = round(vad_end * 1000, 1)
        if self.final_at is not None:
            run.timestamps_ms["stt_final_ms"] = round(self.final_at * 1000, 1)
            if vad_end is not None:
                run.timing("asr", max(0, self.final_at - vad_end) * 1000)
        self.final_at = None
        self.pending_id = turn_id
        self.turns[turn_id] = run

    def speech_created(self, event):
        if event.source != "generate_reply" or self.pending_id is None:
            return
        turn_id = self.pending_id
        self.pending_id = None

        def done(handle):
            run = self.turns.pop(turn_id)
            task = asyncio.create_task(self.finish(run, handle))
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)

        event.speech_handle.add_done_callback(done)

    async def reply(self, user):
        run = self.turns[user.id]

        async def send(event, data):
            await publish_event(self.participant, self.identity, run, event, data)

        transcript = user.text_content
        await self.user_caption(run, transcript, is_final=True)
        async with self.graph_lock:
            return await answer_voice(self.graph, transcript, run, send)

    async def finish(self, run, handle):
        if handle.exception() is not None:
            run.status = "error"
        elif handle.interrupted:
            run.status = "interrupted"
        for item in handle.chat_items:
            if isinstance(item, llm.ChatMessage) and item.role == "assistant":
                if (ttfb := item.metrics.get("tts_node_ttfb")) is not None:
                    run.timing("tts_ttfb", ttfb * 1000)
                if (e2e := item.metrics.get("e2e_latency")) is not None:
                    run.timing("total_e2e", e2e * 1000)
                    run.duration_ms = round(e2e * 1000, 1)
                await publish_event(self.participant, self.identity, run, "transcript", {
                    "role": "assistant", "text": item.text_content, "is_final": True,
                    "interrupted": item.interrupted,
                })
        await asyncio.gather(store_voice_run(run), publish_event(
            self.participant, self.identity, run, "latency", run.model_dump(),
        ))

    async def drain(self):
        await asyncio.gather(*self.tasks)


def make_session(vad):
    stt, tts = speech_models()
    return AgentSession(stt=stt, tts=tts, vad=vad, turn_handling={
        "turn_detection": "vad", "endpointing": {"min_delay": 1.1},
        "interruption": {"enabled": True, "mode": "vad", "min_duration": 0.6},
        "preemptive_generation": {"enabled": False},
    })


def setup(proc):
    if sys.platform == "win32":
        # Psycopg needs a selector loop in the LiveKit job process.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    proc.userdata["vad"] = silero.VAD.load(min_silence_duration=0.95, min_speech_duration=0.15)


load_dotenv()
server = AgentServer()
server.setup_fnc = setup


@server.rtc_session(agent_name=VOICE_AGENT_NAME)
async def entrypoint(ctx):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    caller = await ctx.wait_for_participant()
    async with AsyncExitStack() as resources:
        graph = await resources.enter_async_context(open_graph())
        agent = VoiceAgent(graph, ctx.room.local_participant, caller.identity)
        session = make_session(ctx.proc.userdata["vad"])
        agent.attach(session)
        resources.push_async_callback(agent.drain)
        resources.push_async_callback(session.aclose)
        await session.start(agent, room=ctx.room, room_options=room_io.RoomOptions(
            participant_identity=caller.identity, text_input=False, text_output=False,
        ))
        ctx.add_shutdown_callback(resources.pop_all().aclose)


if __name__ == "__main__":
    cli.run_app(server)
