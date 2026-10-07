import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import ChartTooltip from './ChartTooltip.jsx'
import { AXIS_TICK, BLUE, CHART_MARGIN, GRID } from './chartTheme.js'
import { fmtMonth } from './format.js'

export default function CumulativeChart({ data }) {
  if (data.length < 2) return <p className="panel-empty">Not enough months of launches to draw a trend.</p>
  let total = 0 // running sum of the backend's monthly counts -- a display transform, no new figures
  const running = data.map((d) => ({ month: d.month, total: (total += d.count) }))

  return (
    <ResponsiveContainer width="100%" height={240}>
      <LineChart data={running} margin={CHART_MARGIN}>
        <CartesianGrid strokeDasharray="3 3" stroke={GRID} />
        <XAxis dataKey="month" tickFormatter={fmtMonth} tick={AXIS_TICK} tickLine={false} axisLine={false} minTickGap={24} />
        <YAxis allowDecimals={false} tick={AXIS_TICK} tickLine={false} axisLine={false} />
        <Tooltip content={<ChartTooltip formatLabel={fmtMonth} unit="launches to date" />} />
        <Line type="monotone" dataKey="total" stroke={BLUE} strokeWidth={2.5} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  )
}
