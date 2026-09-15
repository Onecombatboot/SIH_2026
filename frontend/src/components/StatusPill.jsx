const LABELS = {
  open: 'Live',
  connecting: 'Connecting',
  closed: 'Reconnecting',
}

export default function StatusPill({ status }) {
  return (
    <span className="status-pill" data-status={status} role="status">
      <span className="status-dot" />
      {LABELS[status] ?? status}
    </span>
  )
}
