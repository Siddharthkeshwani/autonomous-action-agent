import type { FormEvent, ReactNode } from "react";

const examples = [
  ["Audio/UPI", "What is the dispute resolution timeline for a failed UPI payment?"],
  ["BFSI", "Plot the monthly UPI transaction volume by status as a bar chart."],
  ["Healthcare", "What is the policy for rescheduling a healthcare appointment?"],
  ["E-commerce", "What is the refund policy for a cancelled order?"],
];

const stepNames: Record<string, string> = {
  router: "Route request",
  model: "Prepare response",
  query_readonly_sql: "SQL query",
  execute_python_sandbox: "Sandbox run",
  initiate_refund: "Refund",
  create_ticket_or_email: "Support action",
  search_policy_kb: "Policy search",
  get_db_schema: "Read database schema",
  get_transaction_or_order: "Record lookup",
};

export type Step = { step: string; status: string; message?: string; latency_ms?: number };
export type Receipt = { status: "completed" | "rejected" | "error"; receipt?: object; message?: string };
export type Turn = {
  id: string;
  question: string;
  steps: Step[];
  charts: string[];
  answer?: string;
  receipt?: Receipt;
  error?: string;
};

function stepText(step: Step) {
  const name = stepNames[step.step] ?? step.step;
  if (step.status === "retry") return step.message ?? "Retrying";
  if (step.status === "running") return name + " (running)";
  if (step.status === "ok") {
    return name + (step.step === "execute_python_sandbox" ? " (verified)" : " (done)");
  }
  return name + " (" + step.status + ")";
}

export default function ChatPanel({
  turns, busy, waiting, message, onMessageChange, dictating, dictationError, recordControl, onSend,
}: {
  turns: Turn[];
  busy: boolean;
  waiting: boolean;
  message: string;
  onMessageChange: (message: string) => void;
  dictating: boolean;
  dictationError: string;
  recordControl: ReactNode;
  onSend: (message: string) => void;
}) {
  function send(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (busy || waiting || dictating || !message.trim()) return;
    onSend(message.trim());
    onMessageChange("");
  }

  return (
    <section aria-labelledby="chat-heading" className="panel flex min-h-[35rem] flex-col">
      <div className="border-b border-line px-5 py-4 sm:px-6">
        <h2 id="chat-heading" className="text-lg font-semibold">Chat &amp; Analytics</h2>
      </div>

      <div className="min-h-52 flex-1 space-y-6 px-5 py-5 sm:px-6">
        {turns.length === 0 && (
          <p className="text-sm text-muted">Ask about a record, policy, or transaction trend.</p>
        )}
        {turns.map((turn) => (
          <article key={turn.id} className="space-y-3 border-b border-line pb-5 last:border-0 last:pb-0">
            <p className="ml-auto w-fit max-w-full whitespace-pre-wrap break-words rounded-md bg-track px-3 py-2 text-sm font-medium">
              {turn.question}
            </p>
            {turn.steps.length > 0 && (
              <details className="rounded-md border border-line bg-surface px-3 py-2 text-sm">
                <summary className="cursor-pointer break-words font-medium">
                  Steps ({turn.steps.length}) <span className="font-normal text-muted">
                    {stepText(turn.steps[turn.steps.length - 1])}
                  </span>
                </summary>
                <ol className="mt-2 space-y-1 pl-4 text-muted">
                  {turn.steps.map((step, index) => (
                    <li key={index} className="flex justify-between gap-2">
                      <span>{stepText(step)}</span>
                      {step.latency_ms != null && <span className="shrink-0 tabular-nums">{step.latency_ms} ms</span>}
                    </li>
                  ))}
                </ol>
              </details>
            )}
            {turn.charts.map((chart, index) => (
              <img key={index} src={chart} alt={`Chart ${index + 1} for ${turn.question}`}
                className="h-auto max-w-full rounded-md border border-line" />
            ))}
            {turn.receipt && (
              <div className="rounded-md border border-line px-3 py-2 text-sm">
                <p className={turn.receipt.status === "completed" ? "font-medium text-ready" : "font-medium"}>
                  {turn.receipt.status === "completed" ? "Execution receipt"
                    : turn.receipt.status === "rejected" ? "Action rejected" : "Action failed"}
                </p>
                {turn.receipt.receipt && (
                  <pre className="mt-2 whitespace-pre-wrap break-all text-xs">{JSON.stringify(turn.receipt.receipt, null, 2)}</pre>
                )}
                {turn.receipt.message && <p className="mt-1 break-words">{turn.receipt.message}</p>}
              </div>
            )}
            {turn.answer && <p className="whitespace-pre-wrap break-words text-sm leading-6">{turn.answer}</p>}
            {turn.error && <p role="alert" className="break-words text-sm text-error">{turn.error}</p>}
          </article>
        ))}
        {busy && <p role="status" className="text-sm text-muted">Request in progress</p>}
      </div>

      <div className="space-y-3 border-t border-line px-5 py-4 sm:px-6">
        {waiting && <p role="status" className="text-sm text-muted">Review the pending action before sending another message.</p>}
        <div role="group" aria-label="Example questions" className="flex flex-wrap gap-2">
          {examples.map(([label, question]) => (
            <button key={label} type="button" className="button" title={question} disabled={busy || waiting}
              onClick={() => onSend(question)}>{label}</button>
          ))}
        </div>
        <form onSubmit={send} className="flex gap-2">
          <label htmlFor="question" className="sr-only">Question</label>
          <input id="question" value={message} readOnly={dictating} onChange={(event) => onMessageChange(event.target.value)}
            placeholder="Ask a question..." className="min-w-0 flex-1 rounded-md border border-line px-3 py-2 text-sm" />
          {recordControl}
          <button type="submit" className="button button-primary"
            disabled={busy || waiting || dictating || !message.trim()}>Send</button>
        </form>
        {dictationError && <p role="alert" className="text-sm text-error">{dictationError}</p>}
      </div>
    </section>
  );
}
