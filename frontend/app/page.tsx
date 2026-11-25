"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import ChatPanel from "../components/ChatPanel";
import type { Receipt, Step, Turn } from "../components/ChatPanel";
import HilCard from "../components/HilCard";
import type { Approval } from "../components/HilCard";
import VoicePanel from "../components/VoicePanel";

const apiUrl = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/+$/, "");
const latencyRows = [
  ["asr", "ASR"],
  ["llm_ttft", "LLM TTFT"],
  ["sandbox", "Sandbox"],
  ["tts_ttfb", "TTS"],
] as const;

type Metrics = { latency: Record<string, { p50_ms: number | null }> };
type Waiting = { approval: Approval; turnId: string; error?: string };
type SpeechRecognition = {
  continuous: boolean;
  interimResults: boolean;
  onresult: ((event: { results: ArrayLike<ArrayLike<{ transcript: string }>> }) => void) | null;
  onerror: ((event: { error: string }) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
  abort: () => void;
};

function speechRecognition() {
  const browser = window as Window & {
    SpeechRecognition?: new () => SpeechRecognition;
    webkitSpeechRecognition?: new () => SpeechRecognition;
  };
  return browser.SpeechRecognition || browser.webkitSpeechRecognition;
}

type GatewayEvent =
  | { type: "step_status"; data: Step }
  | { type: "chart_artifact"; data: { image_base64: string; mime_type: string } }
  | { type: "final_answer"; data: { answer: string } }
  | { type: "hil_required"; data: Approval }
  | { type: "execution_receipt"; data: Receipt }
  | { type: "error"; data: { message: string } };

async function stream(path: string, body: object, receive: (event: GatewayEvent) => void) {
  const response = await fetch(`${apiUrl}/api/v1/agent/${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(`Request failed (HTTP ${response.status}).`);
  if (!response.body || !response.headers.get("content-type")?.includes("text/event-stream")) {
    throw new Error("The gateway did not return a stream.");
  }

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += value;
      const frames = buffer.split(/\r?\n\r?\n/);
      buffer = frames.pop() ?? "";
      for (const frame of frames) {
        const lines = frame.split(/\r?\n/);
        const type = lines.find((line) => line.startsWith("event:"))?.slice(6).trim();
        const data = lines.filter((line) => line.startsWith("data:")).map((line) => line.slice(5)).join("\n");
        if (type && data) receive({ type, data: JSON.parse(data) } as GatewayEvent);
      }
    }
  } finally {
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}

function Latency({ metrics, error }: { metrics: Metrics | null; error: boolean }) {
  const max = Math.max(1, ...latencyRows.map(([key]) => metrics?.latency[key]?.p50_ms ?? 0));

  return (
    <section aria-labelledby="latency-heading" className="panel p-5">
      <h2 id="latency-heading" className="text-base font-semibold">Latency (ms)</h2>
      <p className="mt-1 text-sm text-muted">Median of recent samples</p>
      {error && <p role="status" className="mt-2 text-sm text-error">Metrics unavailable.</p>}
      {!metrics && !error && <p role="status" className="mt-2 text-sm text-muted">Loading metrics</p>}
      {metrics && (
        <dl className="mt-4 space-y-3 text-sm">
          {latencyRows.map(([key, label]) => {
            const value = metrics.latency[key]?.p50_ms;
            return (
              <div key={key}>
                <div className="flex justify-between gap-2">
                  <dt className="font-medium">{label}</dt>
                  <dd className="tabular-nums text-muted">{value == null ? "No samples" : value}</dd>
                </div>
                <div aria-hidden="true" className="mt-1.5 h-1.5 rounded-full bg-track">
                  <div className="h-full rounded-full bg-accent" style={{ width: `${((value ?? 0) / max) * 100}%` }} />
                </div>
              </div>
            );
          })}
        </dl>
      )}
    </section>
  );
}

export default function Home() {
  const threadId = useRef<string | null>(null);
  const requestActive = useRef(false);
  const recognition = useRef<SpeechRecognition | null>(null);
  const [message, setMessage] = useState("");
  const [canRecord, setCanRecord] = useState(false);
  const [dictation, setDictation] = useState<"idle" | "recording" | "stopping">("idle");
  const [dictationError, setDictationError] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [pending, setPending] = useState<Waiting[]>([]);
  const [busy, setBusy] = useState(false);
  const [health, setHealth] = useState<boolean | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [metricsError, setMetricsError] = useState(false);

  useEffect(() => {
    setCanRecord(Boolean(speechRecognition()));
    return () => {
      const session = recognition.current;
      recognition.current = null;
      session?.abort();
    };
  }, []);

  function toggleDictation() {
    if (recognition.current) {
      setDictation("stopping");
      recognition.current.stop();
      return;
    }

    const Recognition = speechRecognition();
    if (!Recognition) return;
    setDictationError("");
    const prefix = message && !/\s$/.test(message) ? message + " " : message;
    try {
      const session = new Recognition();
      session.continuous = true;
      session.interimResults = true;
      session.onresult = ({ results }) => {
        if (recognition.current !== session) return;
        const transcript = Array.from(results, (result) => result[0].transcript.trim()).filter(Boolean).join(" ");
        setMessage(transcript ? prefix + transcript : message);
      };
      session.onerror = ({ error }) => {
        if (recognition.current === session && error !== "no-speech") {
          setDictationError("Dictation failed. Check microphone access and try again.");
        }
      };
      session.onend = () => {
        if (recognition.current !== session) return;
        recognition.current = null;
        setDictation("idle");
      };
      recognition.current = session;
      session.start();
      setDictation("recording");
    } catch {
      recognition.current = null;
      setDictationError("Dictation could not start.");
    }
  }

  useEffect(() => {
    let active = true;
    async function refresh() {
      const get = (path: string) => fetch(apiUrl + path, { signal: AbortSignal.timeout(5000), cache: "no-store" });
      const [status, timings] = await Promise.allSettled([
        get("/health"),
        get("/api/v1/metrics").then((response) => {
          if (!response.ok) throw new Error("Metrics unavailable.");
          return response.json() as Promise<Metrics>;
        }),
      ]);
      if (!active) return;
      setHealth(status.status === "fulfilled" && status.value.ok);
      setMetricsError(timings.status === "rejected");
      setMetrics(timings.status === "fulfilled" ? timings.value : null);
    }
    void refresh();
    const timer = window.setInterval(refresh, 10000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  const queueApproval = useCallback((approval: Approval, turnId: string) => {
    setPending((items) => {
      const previous = items.find((item) => item.approval.thread_id === approval.thread_id);
      if (previous?.approval.interrupt_id === approval.interrupt_id) return items;
      return [...items.filter((item) => item !== previous), { approval, turnId }];
    });
  }, []);

  const voiceApproval = useCallback((approval: Approval) => {
    const id = `voice-${approval.interrupt_id}`;
    queueApproval(approval, id);
    setTurns((items) => items.some((item) => item.id === id) ? items
      : [...items, { id, question: "Voice request", steps: [], charts: [] }]);
  }, [queueApproval]);

  function updateTurn(id: string, update: (turn: Turn) => Turn) {
    setTurns((items) => items.map((turn) => turn.id === id ? update(turn) : turn));
  }

  async function run(id: string, message: string, decision?: { waiting: Waiting; approved: boolean }) {
    if (requestActive.current) return;
    requestActive.current = true;
    setBusy(true);
    if (decision) {
      setPending((items) => items.map((item) => item === decision.waiting ? { ...item, error: undefined } : item));
      updateTurn(id, (turn) => ({ ...turn, error: undefined }));
    } else {
      setTurns((items) => [...items, { id, question: message, steps: [], charts: [] }]);
    }

    const approval = decision?.waiting.approval;
    const chatThread = threadId.current ??= crypto.randomUUID();
    let finished = false;
    try {
      await stream(approval ? "confirm" : "chat", approval ? {
        thread_id: approval.thread_id,
        interrupt_id: approval.interrupt_id,
        approved: decision.approved,
      } : { thread_id: chatThread, message }, ({ type, data }) => {
        switch (type) {
          case "step_status":
            updateTurn(id, (turn) => ({ ...turn, steps: [...turn.steps, data] }));
            break;
          case "chart_artifact":
            if (data.mime_type === "image/png") {
              updateTurn(id, (turn) => ({ ...turn, charts: [...turn.charts, `data:image/png;base64,${data.image_base64}`] }));
            }
            break;
          case "hil_required":
            queueApproval(data, id);
            finished = true;
            break;
          case "execution_receipt":
            updateTurn(id, (turn) => ({ ...turn, receipt: data }));
            setPending((items) => items.filter((item) => item.approval.interrupt_id !== approval?.interrupt_id
              || item.approval.thread_id !== approval?.thread_id));
            finished = true;
            break;
          case "final_answer":
            updateTurn(id, (turn) => ({ ...turn, answer: data.answer }));
            if (!approval) finished = true;
            break;
          case "error":
            throw new Error(data.message);
        }
      });
      if (!finished) throw new Error("The stream ended before the request finished.");
    } catch (error) {
      const detail = error instanceof Error ? error.message : "Request failed.";
      if (approval && !finished) {
        setPending((items) => items.map((item) => item.approval === approval ? { ...item, error: detail } : item));
      } else {
        updateTurn(id, (turn) => ({ ...turn, error: detail }));
      }
    } finally {
      requestActive.current = false;
      setBusy(false);
    }
  }

  return (
    <>
      <header className="border-b border-line bg-paper">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 px-5 py-4 sm:px-8">
          <h1 className="break-words text-xl font-semibold">OmniOps Workspace</h1>
          <p role="status" className="flex items-center gap-2 text-sm text-muted">
            <span aria-hidden="true" className={`h-2 w-2 rounded-full ${health === null ? "bg-muted" : health ? "bg-ready" : "bg-error"}`} />
            {health === null ? "Checking gateway" : health ? "Gateway online" : "Gateway unavailable"}
          </p>
        </div>
      </header>
      <main className="mx-auto grid min-h-[calc(100dvh-61px)] max-w-7xl items-start gap-5 px-5 py-6 sm:px-8 lg:grid-cols-[minmax(0,13fr)_minmax(0,7fr)]">
        <ChatPanel turns={turns} busy={busy} waiting={pending.some((item) => item.approval.thread_id === threadId.current)}
          message={message} onMessageChange={setMessage} dictating={dictation !== "idle"} dictationError={dictationError}
          recordControl={
            <span className="inline-flex" title={canRecord ? undefined : "Dictation is unavailable in this browser."}>
              <button type="button" className={canRecord ? "button" : "button pointer-events-none"}
                disabled={!canRecord || dictation === "stopping"} onClick={toggleDictation}>
                {dictation === "idle" ? "Record" : "Stop"}
              </button>
            </span>
          }
          onSend={(message) => void run(crypto.randomUUID(), message)} />
        <div className="min-w-0 space-y-5">
          <VoicePanel apiUrl={apiUrl} onApproval={voiceApproval}
            demoNote="Live WebRTC voice runs on-demand during demos. Use the Record button in Chat & Analytics for 24/7 voice input and charts." />
          {pending.map((waiting) => (
            <HilCard key={waiting.approval.interrupt_id} approval={waiting.approval} error={waiting.error} busy={busy}
              onDecision={(approved) => void run(waiting.turnId, "", { waiting, approved })} />
          ))}
          <Latency metrics={metrics} error={metricsError} />
        </div>
      </main>
    </>
  );
}
