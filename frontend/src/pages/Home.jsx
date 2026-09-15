import { Link } from 'react-router-dom'
import AppShell from '../components/AppShell'
import Icon from '../components/Icon'

const APPS = [
  {
    path: '/visioncane',
    name: 'VisionCane',
    tag: 'Assistive navigation',
    description:
      'A digital white cane for people with visual impairment. Finds the most urgent obstacle ahead and tells you where it is through sound.',
    points: [
      'Real-time depth and obstacle fusion',
      'Stereo panning encodes direction',
      'Beep rate and loudness encode distance',
    ],
  },
  {
    path: '/neuromap',
    name: 'NeuroMap',
    tag: 'Neuro-rehabilitation',
    description:
      'Gamified spatial-attention therapy for patients recovering from traumatic brain injury or stroke-related spatial neglect.',
    points: [
      'Camera-based head-pose tracking, no wearables',
      'Adaptive difficulty: 3 → 5 → 9 spatial zones',
      'Live accuracy, streak and progress metrics',
    ],
  },
]

const STEPS = [
  {
    title: 'Capture',
    body: 'A single ordinary RGB camera. No LiDAR, no depth sensor, no special hardware.',
  },
  {
    title: 'Perceive',
    body: 'The HEIMDALL perception engine estimates dense depth, detects obstacles and tracks head pose on-device in real time.',
  },
  {
    title: 'Sonify',
    body: 'Spatial audio cues are streamed to any phone browser over the local network. Nothing leaves the device.',
  },
]

export default function Home() {
  return (
    <AppShell>
      <section className="hero">
        <span className="eyebrow">Spatial perception for health</span>
        <h1 className="hero-title">
          Turning one camera into a <em>sense of space.</em>
        </h1>
        <p className="hero-copy">
          HEIMDALL reconstructs 3D structure from a single camera and translates it into intuitive
          spatial audio — helping blind users navigate safely and brain-injury patients rebuild
          spatial awareness.
        </p>
      </section>

      <div className="app-grid">
        {APPS.map((app) => (
          <Link key={app.path} to={app.path} className="app-card">
            <span className="tag">{app.tag}</span>
            <h2>{app.name}</h2>
            <p>{app.description}</p>
            <ul>
              {app.points.map((point) => (
                <li key={point}>{point}</li>
              ))}
            </ul>
            <span className="app-card-cta">
              Launch {app.name}
              <Icon name="arrow" />
            </span>
          </Link>
        ))}
      </div>

      <section className="section">
        <h2 className="section-title">How it works</h2>
        <div className="steps">
          {STEPS.map((step) => (
            <div key={step.title} className="card step">
              <h3>{step.title}</h3>
              <p>{step.body}</p>
            </div>
          ))}
        </div>
      </section>

      <footer className="footer">HEIMDALL · Private by design — all processing runs locally.</footer>
    </AppShell>
  )
}
