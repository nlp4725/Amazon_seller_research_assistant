import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import ChartTooltip from './ChartTooltip.jsx'
import { AXIS_TICK, CHART_MARGIN, GRID, PRIMARY } from './chartTheme.js'
import { fmtMonth } from './format.js'

export default function LaunchChart({ data }) {
  if (data.length < 2) return <p className="panel-empty">Not enough months of launches to draw a trend.</p>

  return (
    <ResponsiveContainer width="100%" height={260}>
      <AreaChart data={data} margin={CHART_MARGIN}>
        <defs>
          <linearGradient id="launch-fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={PRIMARY} stopOpacity={0.12} />
            <stop offset="100%" stopColor={PRIMARY} stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke={GRID} vertical={false} />
        <XAxis dataKey="month" tickFormatter={fmtMonth} tick={AXIS_TICK} tickLine={false} axisLine={false} minTickGap={24} />
        <YAxis allowDecimals={false} tick={AXIS_TICK} tickLine={false} axisLine={false} />
        <Tooltip content={<ChartTooltip formatLabel={fmtMonth} unit="launches" />} />
        <Area
          type="monotone" dataKey="count" stroke={PRIMARY} strokeWidth={2} fill="url(#launch-fill)" dot={false}
          activeDot={{ r: 3, fill: PRIMARY, stroke: 'white', strokeWidth: 2 }}
        />
      </AreaChart>
    </ResponsiveContainer>
  )
}
