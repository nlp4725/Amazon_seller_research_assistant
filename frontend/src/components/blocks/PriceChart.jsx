import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import ChartTooltip from './ChartTooltip.jsx'
import { AXIS_TICK, CHART_MARGIN, GRID, MUTED_BAR, PRIMARY } from './chartTheme.js'

export default function PriceChart({ data, peak }) {
  return (
    <ResponsiveContainer width="100%" height={240}>
      <BarChart data={data} margin={CHART_MARGIN}>
        <CartesianGrid strokeDasharray="3 3" stroke={GRID} vertical={false} />
        <XAxis dataKey="range" tick={AXIS_TICK} tickLine={false} axisLine={false} />
        <YAxis allowDecimals={false} tick={AXIS_TICK} tickLine={false} axisLine={false} />
        <Tooltip cursor={{ fill: 'rgba(121, 82, 204, 0.06)' }} content={<ChartTooltip unit="launches" />} />
        <Bar dataKey="count" radius={[3, 3, 0, 0]}>
          {data.map((d) => <Cell key={d.range} fill={d.range === peak ? PRIMARY : MUTED_BAR} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
