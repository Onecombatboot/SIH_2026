import { useEffect, useRef, useState } from 'react'

// Draws a base64 JPEG stream onto a canvas, with a placeholder until the first frame lands.
export default function VideoFeed({ frame, label, connected }) {
  const canvasRef = useRef(null)
  const [hasFrame, setHasFrame] = useState(false)

  useEffect(() => {
    if (!frame) return
    const img = new Image()
    img.onload = () => {
      const canvas = canvasRef.current
      if (!canvas) return
      // Resizing a canvas clears it, so only do it when the stream size changes.
      if (canvas.width !== img.width) canvas.width = img.width
      if (canvas.height !== img.height) canvas.height = img.height
      canvas.getContext('2d').drawImage(img, 0, 0)
      setHasFrame(true)
    }
    img.src = `data:image/jpeg;base64,${frame}`
  }, [frame])

  return (
    <figure className="feed" data-empty={!hasFrame}>
      <canvas ref={canvasRef} aria-label={label} />
      {label && <figcaption className="feed-label">{label}</figcaption>}
      {!hasFrame && (
        <div className="feed-placeholder">
          <div className="spinner" />
          {connected ? 'Warming up the perception engine…' : 'Waiting for the HEIMDALL server…'}
        </div>
      )}
    </figure>
  )
}
