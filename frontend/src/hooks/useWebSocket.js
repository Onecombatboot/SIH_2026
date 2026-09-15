import { useEffect, useState } from 'react'

const EMPTY = { frame: null, depth: null, cue: null, game: null }
const MAX_RETRY_MS = 5000

export function useWebSocket(path) {
  const [payload, setPayload] = useState(EMPTY)
  const [error, setError] = useState(null)
  const [status, setStatus] = useState('connecting')

  useEffect(() => {
    let ws
    let retryTimer
    let attempt = 0
    let disposed = false

    const connect = () => {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      ws = new WebSocket(`${protocol}//${window.location.host}${path}`)
      setStatus('connecting')

      ws.onopen = () => {
        attempt = 0
        setStatus('open')
      }

      ws.onmessage = (event) => {
        let data
        try {
          data = JSON.parse(event.data)
        } catch {
          return
        }
        if (data.error) {
          setError(data.error)
          return
        }
        setError(null)
        setPayload({
          frame: data.frame ?? null,
          depth: data.depth ?? null,
          cue: data.cue ?? null,
          game: data.game ?? null,
        })
      }

      // Reconnect with exponential backoff so a server restart mid-demo recovers on its own.
      ws.onclose = () => {
        if (disposed) return
        setStatus('closed')
        retryTimer = setTimeout(connect, Math.min(MAX_RETRY_MS, 500 * 2 ** attempt++))
      }
    }

    connect()

    return () => {
      disposed = true
      clearTimeout(retryTimer)
      ws?.close()
    }
  }, [path])

  return { ...payload, error, status, connected: status === 'open' }
}
