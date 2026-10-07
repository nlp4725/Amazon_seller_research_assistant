export default function Logo({ onClick }) {
  return (
    <a className="logo" href="./" onClick={(e) => { if (onClick) { e.preventDefault(); onClick() } }}>
      <span className="logo-mark">KP</span>
      <span className="logo-name">KeepaPulse</span>
    </a>
  )
}
