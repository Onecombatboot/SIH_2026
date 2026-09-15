import { useWebSocket } from '../hooks/useWebSocket'
import AppShell from '../components/AppShell'
import AudioEngine from '../components/AudioEngine'
import ErrorBanner from '../components/ErrorBanner'
import VideoFeed from '../components/VideoFeed'

// Mirrors RESPONSE_WINDOW in apps/neuromap/config.py.
const RESPONSE_WINDOW = 3.0
const MAX_LEVEL = 3

function Prompt({ game }) {
  if (!game) {
    return (
      <div className="card prompt">
        <p className="prompt-kicker">Session</p>
        <p className="prompt-main">Get ready</p>
      </div>
    )
  }

  if (game.state === 'cue') {
    return (
      <div className="card prompt" aria-live="polite">
        <p className="prompt-kicker">Keep still</p>
        <p className="prompt-main">Listen for the tone…</p>
      </div>
    )
  }

  if (game.state === 'response') {
    const remaining = Math.max(0, Math.min(1, game.response_remaining / RESPONSE_WINDOW))
    return (
      <div className="card prompt" data-tone="go" aria-live="polite">
        <p className="prompt-kicker">Turn your head toward</p>
        <p className="prompt-main">{game.target_label}</p>
        <div className="bar">
          <div
            className="bar-fill"
            style={{ width: `${remaining * 100}%`, background: remaining < 0.4 ? 'var(--warn)' : undefined }}
          />
        </div>
      </div>
    )
  }

  return (
    <div className="card prompt" data-tone={game.correct ? 'success' : 'danger'} aria-live="polite">
      <p className="prompt-kicker">{game.correct ? 'Well done' : 'Out of time'}</p>
      <p className="prompt-main">{game.correct ? 'Correct!' : 'Missed'}</p>
    </div>
  )
}

function ZoneMap({ game }) {
  const n = game?.n_zones ?? 3
  const grid = n === 9
  const showTarget = game?.state === 'response'

  // Level 3 zones are indexed col * 3 + row; render row-major so the grid reads naturally.
  const cells = grid
    ? [0, 1, 2].flatMap((row) => [0, 1, 2].map((col) => col * 3 + row))
    : Array.from({ length: n }, (_, i) => i)

  return (
    <div className="card">
      <div className="card-title">Spatial zones</div>
      <div className="zones" style={{ gridTemplateColumns: `repeat(${grid ? 3 : n}, 1fr)` }}>
        {cells.map((zone) => (
          <div
            key={zone}
            className="zone"
            data-target={showTarget && zone === game.target_zone}
            data-here={zone === game?.patient_zone}
          />
        ))}
      </div>
      <div className="legend">
        <span className="l-target">Target</span>
        <span className="l-here">Patient gaze</span>
      </div>
    </div>
  )
}

function formatAngle(value) {
  if (value == null) return '—'
  return `${value > 0 ? '+' : ''}${value.toFixed(1)}°`
}

export default function NeuroMap() {
  const { frame, cue, game, error, status, connected } = useWebSocket('/neuromap/ws')

  const accuracy = game && game.total > 0 ? `${Math.round(game.accuracy * 100)}%` : '—'

  return (
    <AppShell
      title="NeuroMap"
      subtitle="Spatial-attention rehabilitation training"
      status={status}
    >
      <ErrorBanner error={error} />

      <div className="toolbar">
        <AudioEngine cue={cue} />
        <span className="hint">
          Face the camera. When you hear a tone, turn your head toward where it came from.
        </span>
      </div>

      <div className="session">
        <VideoFeed frame={frame} label="Patient view" connected={connected} />

        <div className="side">
          <Prompt game={game} />
          <ZoneMap game={game} />
          <div className="card">
            <div className="card-title">Head pose</div>
            <div className="pose">
              <span>Yaw {formatAngle(game?.current_yaw)}</span>
              <span>Pitch {formatAngle(game?.current_pitch)}</span>
            </div>
          </div>
        </div>
      </div>

      <div className="metrics">
        <div className="metric">
          <div className="metric-label">Level</div>
          <div className="metric-value">
            {game?.level ?? 1}
            <span className="metric-sub"> / {MAX_LEVEL}</span>
          </div>
          <div className="metric-sub">{game?.n_zones ?? 3} zones</div>
        </div>
        <div className="metric">
          <div className="metric-label">Score</div>
          <div className="metric-value">
            {game?.score ?? 0}
            <span className="metric-sub"> / {game?.total ?? 0}</span>
          </div>
          <div className="metric-sub">correct responses</div>
        </div>
        <div className="metric">
          <div className="metric-label">Streak</div>
          <div className="metric-value">{game?.streak ?? 0}</div>
          <div className="metric-sub">in a row</div>
        </div>
        <div className="metric">
          <div className="metric-label">Accuracy</div>
          <div className="metric-value">{accuracy}</div>
          <div className="metric-sub">this session</div>
        </div>
      </div>
    </AppShell>
  )
}
