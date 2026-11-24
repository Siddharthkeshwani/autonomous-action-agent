import asyncio
import json
import logging
import os
from collections import Counter, deque
from math import ceil

from pydantic import BaseModel, Field
from redis.asyncio import Redis
from redis.exceptions import RedisError


log = logging.getLogger(__name__)
COMPONENTS = ("asr", "llm_ttft", "tool_exec", "sql", "sandbox", "tts_ttfb", "total_e2e")
STEP_COMPONENTS = {"query_readonly_sql": "sql", "execute_python_sandbox": "sandbox"}
VOICE_RUNS_KEY = "omniops:voice:runs"
MAX_RUNS = 100


class RunTelemetry(BaseModel):
    operation: str
    thread_id: str | None = None
    turn_id: str | None = None
    status: str = "interrupted"
    duration_ms: float = 0
    timestamps_ms: dict[str, float] = Field(default_factory=dict)
    timings_ms: dict[str, list[float]] = Field(default_factory=dict)
    tool_counts: dict[str, int] = Field(default_factory=dict)

    def timing(self, component, ms):
        self.timings_ms.setdefault(component, []).append(round(ms, 1))

    def tool_finished(self, step, ms):
        self.tool_counts[step] = self.tool_counts.get(step, 0) + 1
        if step in STEP_COMPONENTS:
            self.timing(STEP_COMPONENTS[step], ms)
        if self.operation in ("voice", "replay"):
            self.timing("tool_exec", ms)


class Latency(BaseModel):
    samples: int
    p50_ms: float | None = None
    p95_ms: float | None = None


class MetricsResponse(BaseModel):
    latency: dict[str, Latency]
    tool_counts: dict[str, int]
    runs: list[RunTelemetry]


def percentiles(values):
    ordered = sorted(values)
    count = len(ordered)
    if not count:
        return Latency(samples=0)
    return Latency(samples=count, p50_ms=ordered[ceil(count * .50) - 1],
                   p95_ms=ordered[ceil(count * .95) - 1])


class Telemetry:
    def __init__(self):
        self.runs = deque(maxlen=MAX_RUNS)

    def record(self, run):
        self.runs.append(run)
        log.info(json.dumps({"event": "agent_run", **run.model_dump()}))

    def metrics(self, voice_runs=()):
        runs = [*self.runs, *voice_runs]
        tools = Counter()
        for run in runs:
            tools.update(run.tool_counts)
        latency = {
            name: percentiles(ms for run in runs for ms in run.timings_ms.get(name, ()))
            for name in COMPONENTS
        }
        return MetricsResponse(latency=latency, tool_counts=dict(tools), runs=runs)


async def store_voice_run(run):
    log.info(json.dumps({"event": "agent_run", **run.model_dump()}))
    if not (url := os.getenv("UPSTASH_REDIS_URL")):
        return
    try:
        async with asyncio.timeout(2), Redis.from_url(url) as client:
            async with client.pipeline(transaction=True) as pipe:
                await (pipe.lpush(VOICE_RUNS_KEY, run.model_dump_json())
                       .ltrim(VOICE_RUNS_KEY, 0, MAX_RUNS - 1).execute())
    except (RedisError, TimeoutError, ValueError):
        log.warning("Voice metrics could not be saved to Redis.")


async def recent_voice_runs():
    if not (url := os.getenv("UPSTASH_REDIS_URL")):
        return []
    try:
        async with asyncio.timeout(2), Redis.from_url(url) as client:
            entries = await client.lrange(VOICE_RUNS_KEY, 0, MAX_RUNS - 1)
        return [RunTelemetry.model_validate_json(entry) for entry in reversed(entries)]
    except (RedisError, TimeoutError, ValueError):
        log.warning("Voice metrics could not be read from Redis.")
        return []
