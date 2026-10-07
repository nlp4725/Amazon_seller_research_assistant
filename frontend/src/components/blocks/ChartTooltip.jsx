export default function ChartTooltip({ active, payload, label, formatLabel = (l) => l, unit }) {
  if (!active || !payload?.length) return null
  return (
    <div className="chart-tooltip">
      <p className="chart-tooltip-label">{formatLabel(label)}</p>
      <p className="chart-tooltip-value">{payload[0].value.toLocaleString()} {unit}</p>
    </div>
  )
}
