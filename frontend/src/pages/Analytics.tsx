import { AlertTriangle, CheckCircle2, ChevronDown, Clock3, GitBranch, Inbox, RotateCcw, SlidersHorizontal, X } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { api, formatNumber, percent } from '../lib/api'
import type { AnalyticsData, BreakdownRow, DatasetOptions } from '../types'

type Filters = { date_from: string; date_to: string; service: string; category: string; priority: string; support_line: string }
const emptyFilters: Filters = { date_from: '', date_to: '', service: '', category: '', priority: '', support_line: '' }

export function Analytics({ options }: { options: DatasetOptions }) {
  const [filters, setFilters] = useState<Filters>(emptyFilters)
  const [filtersOpen, setFiltersOpen] = useState(false)
  const [data, setData] = useState<AnalyticsData | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    const controller = new AbortController()
    const params = new URLSearchParams(Object.entries(filters).filter(([, value]) => value))
    api<AnalyticsData>(`/api/analytics/summary?${params}`, { signal: controller.signal })
      .then((next) => { setData(next); setError('') })
      .catch((reason: Error) => { if (reason.name !== 'AbortError') setError(reason.message) })
    return () => controller.abort()
  }, [filters])

  const set = (key: keyof Filters, value: string) => setFilters((current) => ({ ...current, [key]: value }))
  const active = useMemo(() => Object.entries(filters).filter(([, value]) => Boolean(value)) as [keyof Filters, string][], [filters])
  const period = filters.date_from || filters.date_to ? `${filters.date_from || 'начало'} — ${filters.date_to || 'сегодня'}` : 'Весь период'

  return <div className="page analytics-page-v6">
    <header className="monitor-heading analytics-heading">
      <div>
        <h1>Аналитика</h1>
        <p>Главное о нагрузке и сроках обработки обращений.</p>
      </div>
    </header>

    <div className="analytics-toolbar">
      <div className="period-control"><span>Период</span><strong>{period}</strong></div>
      <button className={filtersOpen ? 'secondary selected' : 'secondary'} onClick={() => setFiltersOpen((value) => !value)} aria-expanded={filtersOpen}>
        <SlidersHorizontal size={17} />Фильтры{active.length > 0 && <b>{active.length}</b>}
      </button>
      {active.length > 0 && <button className="toolbar-reset" onClick={() => setFilters(emptyFilters)}><RotateCcw size={16} />Сбросить</button>}
    </div>

    {filtersOpen && <section className="analytics-filter-drawer" aria-label="Фильтры аналитики">
      <label>Дата с<input aria-label="Дата с" type="date" value={filters.date_from} onChange={(event) => set('date_from', event.target.value)} /></label>
      <label>Дата по<input aria-label="Дата по" type="date" value={filters.date_to} onChange={(event) => set('date_to', event.target.value)} /></label>
      <FilterSelect label="Услуга" value={filters.service} values={options.services} onChange={(value) => set('service', value)} />
      <FilterSelect label="Категория" value={filters.category} values={options.categories} onChange={(value) => set('category', value)} />
      <FilterSelect label="Приоритет" value={filters.priority} values={options.priorities} onChange={(value) => set('priority', value)} />
      <FilterSelect label="Линия" value={filters.support_line} values={options.support_lines} onChange={(value) => set('support_line', value)} />
    </section>}

    {active.length > 0 && <div className="filter-chips analytics-chips">{active.map(([key, value]) => <button key={key} onClick={() => set(key, '')}>{filterName(key)}: {value}<X size={14} /></button>)}</div>}
    {error && <div className="alert error"><AlertTriangle size={18} />{error}</div>}

    {!data ? <div className="loading-block">Загружаем аналитику…</div> : <AnalyticsContent data={data} />}
  </div>
}

function AnalyticsContent({ data }: { data: AnalyticsData }) {
  const inTimeShare = data.kpis.tickets ? 1 - data.kpis.overdue_share : 0
  const multiLineShare = data.kpis.tickets ? data.kpis.multi_line / data.kpis.tickets : 0
  const attention = [...data.breakdowns.categories]
    .filter((row) => row.sample_n >= 10 && row.overdue > 0)
    .sort((a, b) => b.overdue - a.overdue || b.overdue_share - a.overdue_share)
    .slice(0, 6)
  const riskyCategories = riskRows(data.breakdowns.categories)
  const riskyLines = riskRows(data.breakdowns.support_lines)
  const clarifications = [...data.breakdowns.categories]
    .filter((row) => row.sample_n >= 10 && row.avg_clarifications > 0)
    .sort((a, b) => b.avg_clarifications - a.avg_clarifications)
    .slice(0, 3)
  const slowCategories = durationRows(data.breakdowns.categories)
  const slowServices = durationRows(data.breakdowns.services)
  const slowPriorities = durationRows(data.breakdowns.priorities)
  const slowLines = durationRows(data.breakdowns.support_lines)

  return <>
    <section className="analytics-kpis" aria-label="Ключевые показатели">
      <KpiCard icon={<Inbox size={19} />} value={formatNumber(data.kpis.tickets)} label="Обращений" />
      <KpiCard icon={<CheckCircle2 size={19} />} value={percent(inTimeShare)} label="В срок" sub={data.kpis.overdue ? `${formatNumber(data.kpis.overdue)} просрочено` : 'Просрочек нет'} tone={data.kpis.overdue_share > .05 ? 'warning' : 'success'} />
      <KpiCard icon={<Clock3 size={19} />} value={data.kpis.median_duration || '—'} label="Медиана обработки" sub={data.kpis.p90_duration ? `90-й перцентиль: ${data.kpis.p90_duration}` : undefined} />
      <KpiCard icon={<GitBranch size={19} />} value={formatNumber(data.kpis.multi_line)} label="Несколько линий" sub={`${percent(multiLineShare)} от всех обращений`} />
    </section>

    <section className="analytics-section analytics-trend-section">
      <div className="section-title-row"><div><h2>Динамика обращений</h2><p>Количество зарегистрированных обращений по месяцам.</p></div></div>
      <TrendChart rows={data.time} />
    </section>

    <section className="analytics-dashboard-grid">
      <div className="analytics-section analytics-topic-card">
        <div className="section-title-row"><div><h2>Основные темы</h2><p>Категории, которые встречаются чаще всего.</p></div></div>
        <HorizontalBars rows={data.breakdowns.categories.slice(0, 7)} value={(row) => row.count} format={formatNumber} />
      </div>

      <div className="analytics-section analytics-lines-card">
        <div className="section-title-row"><div><h2>Нагрузка по линиям</h2><p>Сколько обращений закрывает каждая линия поддержки.</p></div></div>
        <HorizontalBars rows={data.breakdowns.support_lines} value={(row) => row.count} format={formatNumber} />
      </div>
    </section>

    <section className="analytics-section analytics-attention">
      <div className="section-title-row">
        <div><h2>Требуют внимания</h2><p>Категории с наибольшим числом просроченных обращений.</p></div>
      </div>
      <AttentionList rows={attention} />
    </section>

    <details className="analytics-more">
      <summary>Сложные обращения <ChevronDown size={16} /></summary>
      <div className="analytics-more-grid analytics-more-grid-four">
        <div className="analytics-detail-card">
          <h3>Риск SLA по категориям</h3>
          <SimpleMetricList rows={riskyCategories} value={(row) => percent(row.smoothed_risk)} />
        </div>
        <div className="analytics-detail-card">
          <h3>Риск SLA по линиям</h3>
          <SimpleMetricList rows={riskyLines} value={(row) => percent(row.smoothed_risk)} />
        </div>
        <div className="analytics-detail-card">
          <h3>Чаще требуют уточнений</h3>
          <SimpleMetricList rows={clarifications} value={(row) => clarificationLabel(row.avg_clarifications)} />
        </div>
        <div className="analytics-detail-highlight">
          <strong>{formatNumber(data.kpis.high_clarifications)}</strong>
          <span>обращений потребовали {data.kpis.clarification_threshold} и более уточнений</span>
          <small>Порог рассчитан по полному набору данных.</small>
        </div>
      </div>
    </details>

    <details className="analytics-more">
      <summary>Сроки обработки <ChevronDown size={16} /></summary>
      <div className="analytics-more-grid analytics-more-grid-four">
        <div className="analytics-detail-card"><h3>Категории</h3><SimpleMetricList rows={slowCategories} value={(row) => durationLabel(row.median_duration_seconds)} /></div>
        <div className="analytics-detail-card"><h3>Услуги</h3><SimpleMetricList rows={slowServices} value={(row) => durationLabel(row.median_duration_seconds)} /></div>
        <div className="analytics-detail-card"><h3>Приоритеты</h3><SimpleMetricList rows={slowPriorities} value={(row) => durationLabel(row.median_duration_seconds)} /></div>
        <div className="analytics-detail-card"><h3>Линии поддержки</h3><SimpleMetricList rows={slowLines} value={(row) => durationLabel(row.median_duration_seconds)} /></div>
      </div>
    </details>
  </>
}

function KpiCard({ icon, value, label, sub, tone }: { icon: React.ReactNode; value: string; label: string; sub?: string; tone?: 'success' | 'warning' }) {
  return <div className={`analytics-kpi${tone ? ` ${tone}` : ''}`}><div className="analytics-kpi-icon">{icon}</div><div><strong>{value}</strong><span>{label}</span>{sub && <small>{sub}</small>}</div></div>
}

function TrendChart({ rows }: { rows: AnalyticsData['time'] }) {
  const series = fillMissingMonths(rows)
  if (!series.length) return <div className="chart-empty">Нет данных для выбранного периода.</div>
  const width = 980
  const height = 300
  const left = 48
  const right = 22
  const top = 18
  const bottom = 42
  const innerW = width - left - right
  const innerH = height - top - bottom
  const maxCount = Math.max(...series.map((row) => row.count), 1)
  const x = (index: number) => left + (series.length === 1 ? innerW / 2 : index * innerW / (series.length - 1))
  const y = (value: number) => top + innerH - value / maxCount * innerH
  const points = series.map((row, index) => `${x(index)},${y(row.count)}`).join(' ')
  const area = `${left},${top + innerH} ${points} ${x(series.length - 1)},${top + innerH}`
  const ticks = [...new Set([0, Math.round(maxCount / 2), maxCount])]
  const labelStep = Math.max(1, Math.ceil(series.length / 6))

  return <div className="trend-chart-wrap"><svg className="trend-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Количество обращений по месяцам">
    {ticks.map((tick) => <g key={tick}><line x1={left} x2={width - right} y1={y(tick)} y2={y(tick)} className="chart-grid-line" /><text x={left - 10} y={y(tick) + 4} textAnchor="end" className="chart-axis-text">{tick}</text></g>)}
    <polygon points={area} className="trend-area" />
    <polyline points={points} className="trend-line" />
    {series.map((row, index) => <g key={row.month} className="trend-point">
      {row.count > 0 && <circle cx={x(index)} cy={y(row.count)} r="4"><title>{monthLabel(row.month)}: {formatNumber(row.count)} обращений</title></circle>}
      {(index === 0 || index === series.length - 1 || index % labelStep === 0) && <text x={x(index)} y={height - 13} textAnchor="middle" className="chart-axis-text">{monthLabel(row.month)}</text>}
    </g>)}
  </svg></div>
}

function HorizontalBars({ rows, value, format }: { rows: BreakdownRow[]; value: (row: BreakdownRow) => number; format: (value: number) => string }) {
  if (!rows.length) return <div className="chart-empty compact">Нет данных.</div>
  const max = Math.max(...rows.map(value), 1)
  return <div className="hbar-chart">{rows.map((row) => <div className="hbar-row" key={row.name}><div className="hbar-label"><span title={row.name}>{cleanLine(row.name)}</span><strong>{format(value(row))}</strong></div><div className="hbar-track"><span style={{ width: `${Math.max(2, value(row) / max * 100)}%` }} /></div></div>)}</div>
}

function AttentionList({ rows }: { rows: BreakdownRow[] }) {
  if (!rows.length) return <div className="analytics-empty-state"><CheckCircle2 size={19} /><span>Нет категорий с заметным числом просрочек.</span></div>
  const max = Math.max(...rows.map((row) => row.overdue), 1)
  return <div className="attention-list">{rows.map((row) => <div className="attention-row" key={row.name}>
    <div><strong title={row.name}>{row.name}</strong><span>{formatNumber(row.overdue)} из {formatNumber(row.count)} просрочено</span></div>
    <div className="attention-track"><i style={{ width: `${Math.max(8, row.overdue / max * 100)}%` }} /></div>
  </div>)}</div>
}

function SimpleMetricList({ rows, value }: { rows: BreakdownRow[]; value: (row: BreakdownRow) => string }) {
  if (!rows.length) return <p className="compact-empty">Недостаточно данных.</p>
  return <div className="simple-metric-list">{rows.map((row) => <div key={row.name}><span title={row.name}>{row.name}</span><strong>{value(row)}</strong></div>)}</div>
}

function fillMissingMonths(rows: AnalyticsData['time']) {
  if (!rows.length) return []
  const sorted = [...rows].sort((a, b) => a.month.localeCompare(b.month))
  const byMonth = new Map(sorted.map((row) => [row.month, row]))
  const [startYear, startMonth] = sorted[0].month.split('-').map(Number)
  const [endYear, endMonth] = sorted[sorted.length - 1].month.split('-').map(Number)
  const cursor = new Date(startYear, startMonth - 1, 1)
  const end = new Date(endYear, endMonth - 1, 1)
  const result: AnalyticsData['time'] = []
  while (cursor <= end) {
    const month = `${cursor.getFullYear()}-${String(cursor.getMonth() + 1).padStart(2, '0')}`
    result.push(byMonth.get(month) ?? { month, count: 0, overdue: 0, overdue_share: 0, smoothed_risk: 0 })
    cursor.setMonth(cursor.getMonth() + 1)
  }
  return result
}

function FilterSelect({ label, value, values, onChange }: { label: string; value: string; values: string[]; onChange: (value: string) => void }) { return <label>{label}<select aria-label={label} value={value} onChange={(event) => onChange(event.target.value)}><option value="">Все</option>{values.map((item) => <option key={item}>{item}</option>)}</select></label> }
function filterName(key: keyof Filters) { return ({ date_from: 'С', date_to: 'По', service: 'Услуга', category: 'Категория', priority: 'Приоритет', support_line: 'Линия' } as Record<keyof Filters, string>)[key] }
function cleanLine(value: string) { return value.replace(/^\(/, '').replace(/\)$/, '') }
function monthLabel(value: string) { const [year, month] = value.split('-'); const names = ['янв', 'фев', 'мар', 'апр', 'май', 'июн', 'июл', 'авг', 'сен', 'окт', 'ноя', 'дек']; return `${names[Number(month) - 1]} ${year.slice(2)}` }
function clarificationLabel(value: number) { return `${value.toLocaleString('ru-RU', { maximumFractionDigits: 1 })} уточнения` }
function durationLabel(seconds: number) {
  if (!seconds) return '—'
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.round((seconds % 3600) / 60)
  return hours ? `${hours} ч ${minutes} мин` : `${minutes} мин`
}
function riskRows(rows: BreakdownRow[]) {
  return [...rows]
    .filter((row) => row.sample_n >= 10)
    .sort((a, b) => b.smoothed_risk - a.smoothed_risk || b.overdue_share - a.overdue_share)
    .slice(0, 4)
}
function durationRows(rows: BreakdownRow[]) {
  const reliable = rows.filter((row) => row.sample_n >= 5 && row.median_duration_seconds > 0)
  const source = reliable.length ? reliable : rows.filter((row) => row.median_duration_seconds > 0)
  return [...source].sort((a, b) => b.median_duration_seconds - a.median_duration_seconds).slice(0, 4)
}
