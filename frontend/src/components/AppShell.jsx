import { Link } from 'react-router-dom'
import Icon from './Icon'
import StatusPill from './StatusPill'

export default function AppShell({ title, subtitle, status, children }) {
  return (
    <>
      <header className="topbar">
        <div className="container topbar-inner">
          {title && (
            <Link to="/" className="btn btn-ghost" aria-label="Back to home">
              <Icon name="back" />
            </Link>
          )}
          <Link to="/" className="brand">
            <Icon name="logo" className="brand-mark" />
            <span className="brand-text">HEIMDALL</span>
          </Link>
          <div className="topbar-spacer" />
          {status && <StatusPill status={status} />}
        </div>
      </header>

      <main className="container page">
        {title && (
          <div className="page-head">
            <div>
              <h1 className="page-title">{title}</h1>
              {subtitle && <p className="page-subtitle">{subtitle}</p>}
            </div>
          </div>
        )}
        {children}
      </main>
    </>
  )
}
