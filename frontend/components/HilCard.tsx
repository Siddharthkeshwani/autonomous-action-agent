export type Approval = {
  thread_id: string;
  interrupt_id: string;
  tool_name: string;
  payload: Record<string, unknown>;
  impact_summary: string;
};

export default function HilCard({ approval, busy, error, onDecision }: {
  approval: Approval;
  busy: boolean;
  error?: string;
  onDecision: (approved: boolean) => void;
}) {
  const { order_id, amount_inr, policy_check } = approval.payload;

  return (
    <section className="panel border-l-4 border-l-accent p-5">
      <h2 className="text-base font-semibold">Pending Approval</h2>
      <p className="mt-2 break-words font-mono text-sm">{approval.tool_name}</p>
      <p className="mt-1 text-sm text-muted">{approval.impact_summary}</p>
      <dl className="mt-3 space-y-1 break-words text-sm">
        {order_id != null && <div><dt className="inline font-medium">Order ID </dt><dd className="inline">{String(order_id)}</dd></div>}
        {amount_inr != null && <div><dt className="inline font-medium">Amount </dt><dd className="inline">INR {String(amount_inr)}</dd></div>}
        <div><dt className="inline font-medium">Policy check </dt>
          <dd className="inline">{policy_check == null ? "No policy check reported." : String(policy_check)}</dd></div>
      </dl>
      <details className="mt-3 text-sm" open>
        <summary className="cursor-pointer font-medium">Payload</summary>
        <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap break-all rounded-md bg-surface p-3 text-xs">
          {JSON.stringify(approval.payload, null, 2)}
        </pre>
      </details>
      {error && <p role="alert" className="mt-3 text-sm text-error">{error}</p>}
      <div className="mt-4 flex flex-wrap gap-2">
        <button type="button" className="button button-primary" disabled={busy} onClick={() => onDecision(true)}>Approve</button>
        <button type="button" className="button text-error" disabled={busy} onClick={() => onDecision(false)}>Reject</button>
      </div>
    </section>
  );
}
