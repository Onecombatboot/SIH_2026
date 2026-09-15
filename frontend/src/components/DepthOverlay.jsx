import VideoFeed from './VideoFeed'

// Colour-mapped depth visualisation (warm = near, dark = far).
export default function DepthOverlay({ depth, connected }) {
  return <VideoFeed frame={depth} label="Depth" connected={connected} />
}
