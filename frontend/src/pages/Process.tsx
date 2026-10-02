import { AlertTriangle, GitBranch } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { api, percent } from '../lib/api'
import type { ProcessData } from '../types'

const duration = (seconds: number) => {
  if (!seconds) return '—'
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.round((seconds % 3600) / 60)
  return hours ? `${hours} ч ${minutes} мин` : `${minutes} мин`
}

export function Process() {
  const [data, setData] = useState<ProcessData | null>(null)
  const [error, setError] = useState('')
  useEffect(() => { api<ProcessData>('/api/process/summary').then(setData).catch((reason: Error) => setError(reason.message)) }, [])

  return <div className="page process-page-v6">
    <header className="monitor-heading">
      <div><h1>Как обращения проходят поддержку</h1><p>Показано участие линий по доступным данным.</p></div>
    </header>
    {error && <div className="alert error"><AlertTriangle size={18} />{error}</div>}
    {!data ? <div className="loading-block">Загружаем процесс…</div> : <ProcessContent data={data} />}
  </div>
}

function ProcessContent({ data }: { data: ProcessData }) {
  const total = data.participation.ticket_count
  const multi = data.participation.multi_line_count
  const groups = data.participation.duration_groups
  const multiCombos = data.participation.combinations.filter((row) => row.combination.includes('+'))
  const frequent = multiCombos.slice(0, 6)
  const risky = [...multiCombos].filter((row) => row.count >= 5).sort((a, b) => b.overdue_rate - a.overdue_rate).slice(0, 6)
  const slow = [...multiCombos].sort((a, b) => b.median_duration_seconds - a.median_duration_seconds).slice(0, 6)
  return <>
    <section className="summary-band process-summary" aria-label="Сводка процесса">
      <SummaryStat value={total.toLocaleString('ru-RU')} label="обращений" />
      <SummaryStat value={percent(total ? multi / total : 0)} label="затронули несколько линий" />
      <SummaryStat value={duration(groups?.single_line?.median_seconds ?? 0)} label="медиана · одна линия" sub={`P90 ${duration(groups?.single_line?.p90_seconds ?? 0)}`} />
      <SummaryStat value={duration(groups?.multi_line?.median_seconds ?? 0)} label="медиана · несколько линий" sub={`P90 ${duration(groups?.multi_line?.p90_seconds ?? 0)}`} />
    </section>

    <section className="process-flow-section">
      <div className="section-title-row">
        <div><h2>Участие линий → финальная линия</h2><p>Толщина связи показывает число обращений.</p></div>
        <span className="method-tag"><GitBranch size={15} />не хронология</span>
      </div>
      <SupportFlow edges={data.participation.resolver_edges} />
      <p className="flow-caption">Это участие линий и финальный исполнитель, не временная последовательность.</p>
    </section>

    <section className="process-insights">
      <InsightList title="Частые комбинации линий" rows={frequent} mode="count" />
      <InsightList title="Где чаще нарушается SLA" rows={risky} mode="risk" />
      <InsightList title="Самые долгие multi-line случаи" rows={slow} mode="duration" />
    </section>

    <details className="process-methodology">
      <summary>Как читать страницу</summary>
      <p>{data.participation.edge_semantics}. Участие фиксируется по ненулевому времени реакции или работы линии.</p>
    </details>

    <section className="status-limit-callout">
      <div><strong>Переходы статусов</strong><span>{data.status_history.available ? `${data.status_history.event_count ?? 0} событий доступно` : 'Недоступно для текущей выгрузки'}</span></div>
      {!data.status_history.available
        ? <p>Для построения нужны: {data.status_history.required_columns.join(', ')}.</p>
        : <p>{data.status_history.message}</p>}
    </section>
  </>
}

function SummaryStat({ value, label, sub }: { value: string; label: string; sub?: string }) {
  return <div className="summary-stat"><strong>{value}</strong><span>{label}</span>{sub && <small>{sub}</small>}</div>
}

function SupportFlow({ edges }: { edges: ProcessData['participation']['resolver_edges'] }) {
  const normalized = useMemo(() => edges
    .map((edge) => ({ from: cleanLine(edge.participant), to: cleanLine(edge.resolver), count: edge.count }))
    .filter((edge) => edge.from && edge.to), [edges])
  const left = ['1', '2', '3', '4']
  const right = ['1', '2', '3', '4']
  const max = Math.max(...normalized.map((edge) => edge.count), 1)
  const leftTotal = Object.fromEntries(left.map((line) => [line, normalized.filter((edge) => edge.from === line).reduce((sum, edge) => sum + edge.count, 0)]))
  const rightTotal = Object.fromEntries(right.map((line) => [line, normalized.filter((edge) => edge.to === line).reduce((sum, edge) => sum + edge.count, 0)]))
  const y = (index: number) => 62 + index * 96
  return <div className="support-flow-wrap">
    <svg className="support-flow" viewBox="0 0 960 420" role="img" aria-label="Связи между участвовавшими линиями и финальными линиями поддержки">
      <defs><marker id="flow-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" /></marker></defs>
      <text x="34" y="24" className="flow-axis-label">Участвовала</text>
      <text x="774" y="24" className="flow-axis-label">Финальная линия</text>
      {normalized.map((edge) => {
        const li = left.indexOf(edge.from)
        const ri = right.indexOf(edge.to)
        if (li < 0 || ri < 0) return null
        const width = 1.4 + 13 * Math.sqrt(edge.count / max)
        return <path key={`${edge.from}-${edge.to}`} d={`M 178 ${y(li)} C 400 ${y(li)}, 550 ${y(ri)}, 780 ${y(ri)}`} className={`flow-link flow-from-${edge.from}`} strokeWidth={width}><title>{edge.from} линия участвовала → финальная {edge.to}: ${edge.count}</title></path>
      })}
      {left.map((line, index) => <g key={`left-${line}`} className="flow-node"><rect x="34" y={y(index) - 29} width="144" height="58" rx="10" /><text x="52" y={y(index) - 3}>{line} линия</text><text x="52" y={y(index) + 17} className="flow-node-count">{leftTotal[line].toLocaleString('ru-RU')} участий</text></g>)}
      {right.map((line, index) => <g key={`right-${line}`} className="flow-node final"><rect x="780" y={y(index) - 29} width="146" height="58" rx="10" /><text x="798" y={y(index) - 3}>{line} линия</text><text x="798" y={y(index) + 17} className="flow-node-count">{rightTotal[line].toLocaleString('ru-RU')} решений</text></g>)}
    </svg>
  </div>
}

function cleanLine(value: string) {
  const match = String(value).match(/[1-4]/)
  return match?.[0] ?? ''
}

function comboLabel(value: string) {
  if (!value || value === 'Нет данных') return 'Линии не определены'
  const parts = value.split('+')
  if (parts.length === 1) return `Участвовала ${parts[0]} линия`
  return `Участвовали: ${parts.join(', ')} линии`
}

function InsightList({ title, rows, mode }: { title: string; rows: ProcessData['participation']['combinations']; mode: 'count' | 'risk' | 'duration' }) {
  return <section className="process-insight"><h3>{title}</h3>{rows.length ? <div>{rows.map((row) => <div className="insight-row" key={`${title}-${row.combination}`}><span>{comboLabel(row.combination)}<small>N = {row.count}</small></span><strong>{mode === 'count' ? row.count.toLocaleString('ru-RU') : mode === 'risk' ? percent(row.overdue_rate) : duration(row.median_duration_seconds)}</strong></div>)}</div> : <p className="insight-empty">Недостаточно данных.</p>}</section>
}
