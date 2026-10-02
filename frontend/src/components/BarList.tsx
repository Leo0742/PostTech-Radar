import type { BreakdownRow } from '../types'
import { formatNumber, percent } from '../lib/api'

export function BarList({ rows, mode = 'count', limit = 8 }: { rows: BreakdownRow[]; mode?: 'count' | 'risk'; limit?: number }) {
  const visible = rows.slice(0, limit)
  const maximum = Math.max(1, ...visible.map((row) => mode === 'count' ? row.count : row.overdue_share))
  return <div className="bar-list">
    {visible.map((row) => {
      const value = mode === 'count' ? row.count : row.overdue_share
      return <div className="bar-row" key={row.name}>
        <div className="bar-label"><span title={row.name}>{row.name}</span><strong>{mode === 'count' ? formatNumber(row.count) : percent(row.overdue_share)}</strong></div>
        <div className="bar-track"><span style={{ width: `${Math.max(2, (value / maximum) * 100)}%` }} /></div>
      </div>
    })}
  </div>
}

