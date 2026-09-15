import { useCallback, useEffect, useRef, useState } from 'react'

const BEEP_SECONDS = 0.05

export function useAudioEngine(cue) {
  const ctxRef = useRef(null)
  const nextBeepRef = useRef(0)
  const [enabled, setEnabled] = useState(false)
  const [muted, setMuted] = useState(false)

  // Browsers (mobile especially) only let audio start from a user gesture,
  // so the AudioContext is created here rather than on mount.
  const enable = useCallback(async () => {
    if (!ctxRef.current) {
      const Ctx = window.AudioContext || window.webkitAudioContext
      ctxRef.current = new Ctx()
    }
    await ctxRef.current.resume()
    setMuted(false)
    setEnabled(true)
  }, [])

  const toggleMute = useCallback(() => setMuted((m) => !m), [])

  useEffect(
    () => () => {
      ctxRef.current?.close()
      ctxRef.current = null
    },
    []
  )

  useEffect(() => {
    const ctx = ctxRef.current
    if (!enabled || muted || !ctx || !cue?.present) return

    const now = ctx.currentTime
    if (now < nextBeepRef.current) return

    const { pan, pitch, volume, interval } = cue

    const osc = ctx.createOscillator()
    const gain = ctx.createGain()
    const panner = ctx.createStereoPanner()

    osc.frequency.setValueAtTime(pitch, now)
    gain.gain.setValueAtTime(0, now)
    gain.gain.linearRampToValueAtTime(volume, now + BEEP_SECONDS * 0.1)
    gain.gain.linearRampToValueAtTime(0, now + BEEP_SECONDS)
    panner.pan.setValueAtTime(pan, now)

    osc.connect(gain)
    gain.connect(panner)
    panner.connect(ctx.destination)

    osc.start(now)
    osc.stop(now + BEEP_SECONDS)

    nextBeepRef.current = now + interval
  }, [cue, enabled, muted])

  return { enabled, muted, enable, toggleMute }
}
