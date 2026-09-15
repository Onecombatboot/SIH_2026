import Icon from './Icon'

export default function ErrorBanner({ error }) {
  if (!error) return null
  return (
    <div className="alert" role="alert">
      <Icon name="alert" className="alert-icon" />
      <span>{error}</span>
    </div>
  )
}
