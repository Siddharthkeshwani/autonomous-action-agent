import { useCallback, useEffect, useState } from "react";
import {
  BarVisualizer, LiveKitRoom, RoomAudioRenderer, StartAudio, useDataChannel,
  useLocalParticipant, useRoomContext, useTranscriptions, useVoiceAssistant,
} from "@livekit/components-react";
import type { Approval } from "./HilCard";

const topic = "omniops.voice";
type Token = { url: string; token: string };
type Caption = { id: string; role: string; text: string };

function VoiceRoom({ onApproval }: { onApproval: (approval: Approval) => void }) {
  const room = useRoomContext();
  const { localParticipant } = useLocalParticipant();
  const { audioTrack } = useVoiceAssistant();
  const transcriptions = useTranscriptions();
  const [fallback, setFallback] = useState<Caption[]>([]);
  const [error, setError] = useState("");

  const receive = useCallback((text: string) => {
    let packet;
    try {
      packet = JSON.parse(text);
    } catch {
      setError("A voice update could not be read.");
      return;
    }
    if (!packet?.data) return;
    const { event, data, turn_id } = packet;
    if (event === "hil_required") onApproval(data);
    if (event === "error") setError(data.message);
    if (event === "transcript") {
      const caption = { id: `${turn_id}:${data.role}`, role: data.role, text: data.text };
      setFallback((previous) => [...previous.filter((item) => item.id !== caption.id), caption].slice(-6));
      if (data.role === "user") setError("");
    }
  }, [onApproval]);

  useDataChannel(topic, (message) => {
    if (message.from?.isAgent) receive(new TextDecoder().decode(message.payload));
  });

  useEffect(() => {
    room.registerTextStreamHandler(topic, async (reader, sender) => {
      if (!room.remoteParticipants.get(sender.identity)?.isAgent) return;
      try {
        receive(await reader.readAll());
      } catch {
        setError("A voice update could not be received.");
      }
    });
    return () => room.unregisterTextStreamHandler(topic);
  }, [room, receive]);

  const captions = transcriptions.length ? transcriptions.slice(-6).map((item) => ({
    id: item.streamInfo.id,
    role: item.participantInfo.identity === localParticipant.identity ? "user" : "assistant",
    text: item.text,
  })) : fallback;

  return (
    <>
      <RoomAudioRenderer />
      <StartAudio label="Enable playback" className="button mt-3" />
      <BarVisualizer trackRef={audioTrack} barCount={7} aria-hidden="true"
        className={`mt-3 flex h-7 items-center gap-1 ${audioTrack ? "voice-audio-active" : ""}`} />
      <h3 className="mt-3 text-sm font-medium">Live captions</h3>
      <div className="mt-2 max-h-44 space-y-2 overflow-auto text-sm text-muted" aria-live="polite">
        {captions.length === 0 && <p>Speak to start. You can speak again to interrupt the reply.</p>}
        {captions.map((caption) => (
          <p key={caption.id} className="break-words">
            <span className="font-medium text-ink">{caption.role === "user" ? "You: " : "Reply: "}</span>{caption.text}
          </p>
        ))}
      </div>
      {error && <p role="alert" className="mt-3 text-sm text-error">{error}</p>}
    </>
  );
}

export default function VoicePanel({ apiUrl, onApproval, demoNote }: {
  apiUrl: string;
  onApproval: (approval: Approval) => void;
  demoNote: string;
}) {
  const [room, setRoom] = useState<Token | null>(null);
  const [status, setStatus] = useState<"idle" | "starting" | "connecting" | "connected">("idle");
  const [error, setError] = useState("");

  const stop = useCallback(() => {
    setRoom(null);
    setStatus("idle");
  }, []);

  async function start() {
    setError("");
    setStatus("starting");
    try {
      const response = await fetch(`${apiUrl}/api/v1/voice/token`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ identity: `browser-${crypto.randomUUID()}` }),
        signal: AbortSignal.timeout(10000),
      });
      if (!response.ok) throw new Error(`Voice could not start (HTTP ${response.status}).`);
      setRoom(await response.json());
      setStatus("connecting");
    } catch (error) {
      setStatus("idle");
      setError(error instanceof Error ? error.message : "Voice could not start.");
    }
  }

  function connectionFailed(error: Error) {
    stop();
    setError(error.message || "The voice connection failed.");
  }

  return (
    <section aria-labelledby="voice-heading" className="panel p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 id="voice-heading" className="text-base font-semibold">Browser Voice</h2>
        {room
          ? <button type="button" className="button" onClick={stop}>Stop Voice</button>
          : <button type="button" className="button button-primary" disabled={status === "starting"}
              onClick={start}>Start Voice</button>}
      </div>
      <p className="mt-2 text-sm text-muted">{demoNote}</p>
      <p role="status" className="mt-3 text-sm text-muted">
        {status === "idle" && "Microphone off"}
        {status === "starting" && "Requesting a voice session"}
        {status === "connecting" && "Connecting"}
        {status === "connected" && "Connected"}
      </p>
      {error && <p role="alert" className="mt-2 text-sm text-error">{error}</p>}
      {room && (
        <LiveKitRoom serverUrl={room.url} token={room.token} connect audio
          onConnected={() => setStatus("connected")} onDisconnected={stop}
          onError={connectionFailed} onMediaDeviceFailure={() => connectionFailed(new Error("Microphone access failed."))}>
          <VoiceRoom onApproval={onApproval} />
        </LiveKitRoom>
      )}
    </section>
  );
}
