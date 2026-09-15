import { useWebSocket } from '../hooks/useWebSocket'
import AppShell from '../components/AppShell'
import AudioEngine from '../components/AudioEngine'
import DepthOverlay from '../components/DepthOverlay'
import ErrorBanner from '../components/ErrorBanner'
import VideoFeed from '../components/VideoFeed'

// Loudest cue volume the server emits (apps/visioncane/config.py MAX_VOL); used to normalise proximity.
const MAX_VOLUME = 0.95

function describeDirection(pan) {
  if (pan < -0.33) return 'Left'
  if (pan > 0.33) return 'Right'
  return 'Ahead'
}

function describeProximity(p) {
  if (p > 0.7) return 'Very close'
  if (p > 0.4) return 'Near'
  return 'Approaching'
}

export default function VisionCane() {
  const { frame, depth, cue, error, status, connected } = useWebSocket('/visioncane/ws')

  const present = Boolean(cue?.present)
  const proximity = present ? Math.min(1, cue.volume / MAX_VOLUME) : 0

  return (
    <AppShell
      title="VisionCane"
      subtitle="Obstacle guidance through spatial audio"
      status={status}
    >
      <ErrorBanner error={error} />

      <div className="toolbar">
        <AudioEngine cue={cue} />
        <span className="hint">Wear headphones: obstacle direction is encoded in stereo.</span>
      </div>

      <div className="feeds">
        <VideoFeed frame={frame} label="Camera" connected={connected} />
        <DepthOverlay depth={depth} connected={connected} />
      </div>

      <div className="metrics">
        <div className="metric">
          <div className="metric-label">Obstacle</div>
          <div className="metric-value">{present ? describeDirection(cue.pan) : 'Path clear'}</div>
          <div className="track" aria-hidden="true">
            <span
              className="track-marker"
              style={{ left: `${((present ? cue.pan : 0) + 1) * 50}%`, opacity: present ? 1 : 0.15 }}
            />
          </div>
          <div className="track-labels">
            <span>Left</span>
            <span>Ahead</span>
            <span>Right</span>
          </div>
        </div>

        <div className="metric">
          <div className="metric-label">Proximity</div>
          <div className="metric-value">{present ? describeProximity(proximity) : '—'}</div>
          <div className="bar">
            <div
              className="bar-fill"
              style={{
                width: `${Math.round(proximity * 100)}%`,
                background: proximity > 0.7 ? 'var(--danger)' : proximity > 0.4 ? 'var(--warn)' : undefined,
              }}
            />
          </div>
        </div>

        <div className="metric">
          <div className="metric-label">Alert rate</div>
          <div className="metric-value">{present ? (1 / cue.interval).toFixed(1) : '0.0'}</div>
          <div className="metric-sub">beeps per second</div>
        </div>
      </div>
    </AppShell>
  )
}
