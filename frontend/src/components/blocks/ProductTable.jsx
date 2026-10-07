import { fmtMonth, fmtPrice } from './format.js'

export default function ProductTable({ products }) {
  if (!products.length) return null

  return (
    <div className="table-scroll">
      <table className="data-table">
        <thead>
          <tr><th>Product</th><th>Seller ID</th><th>Price</th><th>Launched</th></tr>
        </thead>
        <tbody>
          {products.map((p) => (
            <tr key={p.asin}>
              <td className="cell-product">
                <p className="product-title">{p.title}</p>
                <a className="product-asin" href={`https://www.amazon.com/dp/${p.asin}`} target="_blank" rel="noreferrer">{p.asin}</a>
              </td>
              <td className="cell-mono">{p.seller ?? 'Unknown'}</td>
              <td className="cell-strong">{fmtPrice(p.price)}</td>
              <td className="cell-nowrap">{fmtMonth(p.launched)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
