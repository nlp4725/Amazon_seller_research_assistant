export default function ChartCard({ title, subtitle, badge, badgeTone = 'purple', children, className = '' }) {
  return (
    <section className={`panel ${className}`}>
      <div className="panel-head">
        <div>
          <h3 className="panel-title">{title}</h3>
          {subtitle && <p className="panel-subtitle">{subtitle}</p>}
        </div>
        {badge && <span className={`badge badge-${badgeTone}`}>{badge}</span>}
      </div>
      {children}
    </section>
  )
}
