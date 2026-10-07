// Display formatting only -- every number arrives precomputed from the backend's report_view().
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

export const fmtMonth = (ym) => { // "2025-03" -> "Mar '25"
  const [y, m] = ym.split('-')
  return `${MONTHS[Number(m) - 1]} '${y.slice(2)}`
}

export const fmtPrice = (n) => (n == null ? '—' : `$${n.toFixed(2)}`)

export const fmtPct = (n) => (n == null ? '—' : `${n}%`)
