import json
import logging
from collections import Counter, deque
from math import ceil

from pydantic import BaseModel, Field


log = logging.getLogger(__name__)
COMPONENTS = ("asr", "llm_ttft", "sql", "sandbox", "tts_ttfb")
STEP_COMPONENTS = {"query_readonly_sql": "sql", "execute_python_sandbox": "sandbox"}


class RunTelemetry(BaseModel):
    operation: str
    status: str = "interrupted"
    duration_ms: float = 0
    timings_ms: dict[str, list[float]] = Field(default_factory=dict)
    tool_counts: dict[str, int] = Field(default_factory=dict)

    def timing(self, component, ms):
        self.timings_ms.setdefault(component, []).append(round(ms, 1))

    def tool_finished(self, step, ms):
        self.tool_counts[step] = self.tool_counts.get(step, 0) + 1
        if step in STEP_COMPONENTS:
            self.timing(STEP_COMPONENTS[step], ms)


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
        self.runs = deque(maxlen=100)

    def record(self, run):
        self.runs.append(run)
        log.info(json.dumps({"event": "agent_run", **run.model_dump()}))

    def metrics(self):
        runs = list(self.runs)
        tools = Counter()
        for run in runs:
            tools.update(run.tool_counts)
        latency = {
            name: percentiles(ms for run in runs for ms in run.timings_ms.get(name, ()))
            for name in COMPONENTS
        }
        return MetricsResponse(latency=latency, tool_counts=dict(tools), runs=runs)
