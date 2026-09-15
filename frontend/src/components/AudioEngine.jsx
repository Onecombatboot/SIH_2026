import { useAudioEngine } from '../hooks/useAudioEngine'
import Icon from './Icon'

// Plays the spatial cue stream and renders the enable / mute control.
export default function AudioEngine({ cue }) {
  const { enabled, muted, enable, toggleMute } = useAudioEngine(cue)

  if (!enabled) {
    return (
      <button type="button" className="btn btn-primary btn-lg" onClick={enable}>
        <Icon name="sound" />
        Tap to enable audio
      </button>
    )
  }

  return (
    <button type="button" className="btn" onClick={toggleMute} aria-pressed={muted}>
      <Icon name={muted ? 'mute' : 'sound'} />
      {muted ? 'Audio muted' : 'Audio on'}
    </button>
  )
}
