import { fmtPct, fmtPrice } from './format.js'

export default function SellerTable({ sellers }) {
  if (!sellers.length) return null
  const maxPct = Math.max(...sellers.map((s) => s.pct ?? 0)) || 1 // bar scale only

  return (
    <div className="table-scroll">
      <table className="data-table">
        <thead>
          <tr><th>Seller ID</th><th>Launches</th><th>Avg price</th><th>Share</th></tr>
        </thead>
        <tbody>
          {sellers.map((s) => (
            <tr key={s.seller_id}>
              <td className="cell-mono">
                {s.seller_id}
                {s.is_dominant && <span className="trend-tag trend-declining dominant-tag">Dominant</span>}
              </td>
              <td>{s.launches}</td>
              <td>{fmtPrice(s.avg_price)}</td>
              <td>
                <div className="share">
                  <div className="share-track">
                    <div className="share-fill" style={{ width: `${((s.pct ?? 0) / maxPct) * 100}%` }} />
                  </div>
                  <span className="share-value">{fmtPct(s.pct)}</span>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
