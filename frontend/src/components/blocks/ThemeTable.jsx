import { fmtPrice } from './format.js'

const TAG_LABEL = { RISING: 'Rising', DECLINING: 'Declining', EMERGING: 'New', STABLE: 'Stable', INSUFFICIENT_DATA: 'Too little data' }

export default function ThemeTable({ themes }) {
  if (!themes.length) return null
  const years = Object.keys(themes[0].by_year) // zero-filled server-side, identical across themes

  return (
    <div className="table-scroll">
      <table className="data-table">
        <thead>
          <tr>
            <th>Theme (closest example)</th>
            {years.map((y) => <th key={y}>{y}</th>)}
            <th>Growth ’24→’25</th>
            <th>Avg price</th>
          </tr>
        </thead>
        <tbody>
          {themes.map((t, i) => (
            <tr key={i}>
              <td className="cell-title" title={t.label}>{t.label}</td>
              {years.map((y) => <td key={y}>{t.by_year[y]}</td>)}
              <td>
                <span className={`trend-tag trend-${t.trend_tag.toLowerCase()}`}>
                  {t.growth_pct != null ? `${t.growth_pct > 0 ? '+' : ''}${t.growth_pct}% · ` : ''}
                  {TAG_LABEL[t.trend_tag] ?? t.trend_tag}
                </span>
              </td>
              <td>{fmtPrice(t.avg_price)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
