import { HashRouter, Navigate, Route, Routes } from 'react-router-dom'
import Home from './pages/Home'
import NeuroMap from './pages/NeuroMap'
import VisionCane from './pages/VisionCane'

// HashRouter keeps page refreshes working when the build is served by FastAPI's static mount,
// and stops page routes from colliding with the backend's /visioncane and /neuromap paths.
export default function App() {
  return (
    <HashRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/visioncane" element={<VisionCane />} />
        <Route path="/neuromap" element={<NeuroMap />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </HashRouter>
  )
}
