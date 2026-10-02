import { AlertTriangle, CheckCircle2, Database, Gauge, ServerCog } from 'lucide-react'
import { useEffect, useState } from 'react'
import { api, formatNumber, percent } from '../lib/api'

type StatusData = {
  data: { ticket_count: number; dataset_hash?: string; source_path?: string; imported_at?: string; schema_version?: string }
  models: { active_runtime?: unknown } & Record<string, unknown>
}
type Metrics = {
  category: { accuracy: number; macro_f1: number; accepted_accuracy?: number; accepted_coverage?: number; test_size: number; model?: string }
  routing: { accuracy: number; macro_f1: number; accepted_accuracy?: number; accepted_coverage?: number; test_size: number; model?: string }
  retrieval: { same_category_at_1: number; same_category_at_3: number; same_category_at_5: number; queries: number; method: string }
  limitations: Record<string, string>
}

function stringField(value: unknown, names: string[]): string | null {
  if (!value || typeof value !== 'object') return null
  const record = value as Record<string, unknown>
  for (const name of names) if (typeof record[name] === 'string' && record[name]) return record[name] as string
  for (const child of Object.values(record)) {
    const nested = stringField(child, names)
    if (nested) return nested
  }
  return null
}

export function SystemPage() {
  const [status, setStatus] = useState<StatusData | null>(null)
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [error, setError] = useState('')
  useEffect(() => {
    Promise.all([api<StatusData>('/api/system/status'), api<Metrics>('/api/model/metrics')])
      .then(([nextStatus, nextMetrics]) => { setStatus(nextStatus); setMetrics(nextMetrics) })
      .catch((reason: Error) => setError(reason.message))
  }, [])
  if (error) return <div className="page"><div className="alert error"><AlertTriangle size={18} />{error}</div></div>
  if (!status || !metrics) return <div className="page"><div className="loading-block">Проверяем состояние системы…</div></div>
  const activeModel = stringField(status.models.active_runtime, ['candidate_id', 'model_family', 'model', 'name', 'family']) ?? 'Активная модель не определена'
  return <div className="page system-page">
    <div className="page-heading"><div><h1>Система</h1><p>Состояние приложения, активная модель и техническая информация для администраторов и проверки решения.</p></div></div>
    <section className="system-overview">
      <article><CheckCircle2 size={23} /><div><span>Состояние системы</span><strong>Система готова</strong><small>API, база и локальные модели доступны</small></div></article>
      <article><ServerCog size={23} /><div><span>Активная модель</span><strong>{activeModel}</strong><small>используется для рекомендаций оператору</small></div></article>
      <article><Database size={23} /><div><span>Данных в подтверждённой базе</span><strong>{formatNumber(status.data.ticket_count)}</strong><small>история + сохранённые оператором обращения</small></div></article>
      <article><Gauge size={23} /><div><span>Последнее обновление данных</span><strong>{status.data.imported_at ? new Date(status.data.imported_at).toLocaleDateString('ru-RU') : '—'}</strong><small>схема v{status.data.schema_version ?? '—'}</small></div></article>
    </section>

    <details className="system-details">
      <summary>Показать технические детали</summary>
      <div className="system-tech-grid">
        <section><h2>Качество категоризации</h2><dl><dt>Accuracy</dt><dd>{percent(metrics.category.accuracy)}</dd><dt>Macro-F1</dt><dd>{percent(metrics.category.macro_f1)}</dd><dt>TOP / accepted accuracy</dt><dd>{metrics.category.accepted_accuracy != null ? percent(metrics.category.accepted_accuracy) : '—'}</dd><dt>Test N</dt><dd>{metrics.category.test_size}</dd></dl></section>
        <section><h2>Качество маршрутизации</h2><dl><dt>Accuracy</dt><dd>{percent(metrics.routing.accuracy)}</dd><dt>Macro-F1</dt><dd>{percent(metrics.routing.macro_f1)}</dd><dt>Accepted coverage</dt><dd>{metrics.routing.accepted_coverage != null ? percent(metrics.routing.accepted_coverage) : '—'}</dd><dt>Test N</dt><dd>{metrics.routing.test_size}</dd></dl></section>
        <section><h2>Retrieval quality</h2><dl><dt>Same category @1</dt><dd>{percent(metrics.retrieval.same_category_at_1)}</dd><dt>@3</dt><dd>{percent(metrics.retrieval.same_category_at_3)}</dd><dt>@5</dt><dd>{percent(metrics.retrieval.same_category_at_5)}</dd><dt>Queries</dt><dd>{metrics.retrieval.queries}</dd></dl></section>
        <section><h2>Dataset / provenance</h2><dl><dt>Dataset SHA-256</dt><dd className="mono">{status.data.dataset_hash ?? '—'}</dd><dt>Источник</dt><dd className="mono break-all">{status.data.source_path ?? '—'}</dd><dt>Метод retrieval</dt><dd>{metrics.retrieval.method}</dd></dl></section>
      </div>
      <section className="system-limitations"><h2>Технические ограничения</h2>{Object.entries(metrics.limitations).map(([key, value]) => <p key={key}><strong>{key}</strong>{value}</p>)}</section>
    </details>
  </div>
}
