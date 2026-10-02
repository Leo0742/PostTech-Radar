import { AlertTriangle, CheckCircle2, ShieldAlert } from 'lucide-react'
import { useEffect, useState } from 'react'
import { api, percent } from '../lib/api'

interface ClassMetric { precision: number; recall: number; f1: number; support: number }
interface ModelMetric { model: string; accuracy: number; macro_f1: number; weighted_f1: number; test_size: number; per_class: Record<string, ClassMetric>; line4?: string }
interface CategoryMetric extends ModelMetric {
  accepted_coverage: number
  accepted_accuracy: number
  confidence_threshold: number
  margin_threshold: number
  calibration: { brier: number; ece: number; temperature: number }
  out_of_scope: { rejection_rate: number; false_acceptance_rate?: number }
}
interface Metrics {
  category: CategoryMetric
  routing: ModelMetric & { line4: string; confidence_threshold: number; accepted_coverage: number; accepted_accuracy: number; calibration: { brier: number; ece: number; temperature: number } }
  retrieval: { same_category_at_1: number; same_category_at_3: number; same_category_at_5: number; mrr_proxy: number; queries: number; method: string; duplicate_groups_excluded: boolean }
  sla: { overdue_count: number; median_duration_seconds: number; p90_duration_seconds: number; limitation: string }
  limitations: Record<string, string>
}

function Score({ label, value, hint }: { label: string; value: number; hint?: string }) {
  return <div className="quality-score"><span>{label}</span><strong>{percent(value)}</strong>{hint && <small>{hint}</small>}</div>
}

export function Quality() {
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    api<Metrics>('/api/model/metrics').then(setMetrics).catch((reason: Error) => setError(reason.message))
  }, [])

  if (error) return <div className="page"><div className="alert error"><AlertTriangle size={18} />{error}</div></div>
  if (!metrics) return <div className="page"><div className="loading-block">Загружаем артефакты качества…</div></div>

  return (
    <div className="page">
      <div className="page-heading">
        <div>
          <span className="eyebrow">Контроль качества</span>
          <h1>Качество модели</h1>
          <p>Результаты независимой проверки и ограничения, которые оператору важно учитывать при работе с рекомендациями.</p>
        </div>
      </div>

      <div className="quality-layout">
        <section className="quality-card">
          <div className="panel-heading"><h2>Категоризация TOP‑15</h2><span>{metrics.category.model}</span></div>
          <div className="score-row">
            <Score label="Точность" value={metrics.category.accuracy} />
            <Score label="Macro‑F1" value={metrics.category.macro_f1} />
            <Score label="Точность принятых" value={metrics.category.accepted_accuracy} hint="без ручной проверки" />
            <Score label="Доля без проверки" value={metrics.category.accepted_coverage} />
          </div>
          <p className="quality-summary">Проверка выполнена на <strong>{metrics.category.test_size}</strong> обращениях, отделённых от обучения по группам. Если уверенность или разрыв между двумя лучшими вариантами недостаточны, интерфейс просит оператора проверить рекомендацию.</p>
          <details className="technical-details">
            <summary>Технические параметры проверки</summary>
            <p>Порог уверенности: <strong>{percent(metrics.category.confidence_threshold)}</strong> · разрыв TOP‑1/TOP‑2: <strong>{percent(metrics.category.margin_threshold)}</strong> · ECE: <strong>{metrics.category.calibration.ece}</strong> · Brier: <strong>{metrics.category.calibration.brier}</strong> · OOS отклонено: <strong>{percent(metrics.category.out_of_scope.rejection_rate)}</strong>.</p>
          </details>
          <img className="confusion" src="/api/model/figures/category_confusion_matrix.png" alt="Матрица ошибок категоризации" />
          <MetricTable values={metrics.category.per_class} />
        </section>

        <section className="quality-card">
          <div className="panel-heading"><h2>Маршрутизация</h2><span>{metrics.routing.model}</span></div>
          <div className="score-row">
            <Score label="Точность" value={metrics.routing.accuracy} />
            <Score label="Macro‑F1" value={metrics.routing.macro_f1} />
            <Score label="Точность принятых" value={metrics.routing.accepted_accuracy} hint="без ручной проверки" />
            <Score label="Доля без проверки" value={metrics.routing.accepted_coverage} />
          </div>
          <p className="quality-summary">Маршрут проверен на <strong>{metrics.routing.test_size}</strong> обращениях. При слабой уверенности рекомендация линии явно помечается как требующая проверки.</p>
          <details className="technical-details">
            <summary>Технические параметры проверки</summary>
            <p>Порог уверенности: <strong>{percent(metrics.routing.confidence_threshold)}</strong> · ECE: <strong>{metrics.routing.calibration.ece}</strong> · Brier: <strong>{metrics.routing.calibration.brier}</strong>.</p>
          </details>
          <div className="alert warning"><ShieldAlert size={18} />{metrics.routing.line4}</div>
          <img className="confusion" src="/api/model/figures/routing_confusion_matrix.png" alt="Матрица ошибок маршрутизации" />
          <MetricTable values={metrics.routing.per_class} />
        </section>

        <section className="quality-card compact">
          <h2>Поиск похожих обращений</h2>
          <div className="score-row quality-score-row">
            <Score label="Та же категория @1" value={metrics.retrieval.same_category_at_1} />
            <Score label="Та же категория @3" value={metrics.retrieval.same_category_at_3} />
            <Score label="Та же категория @5" value={metrics.retrieval.same_category_at_5} />
            <Score label="MRR proxy" value={metrics.retrieval.mrr_proxy} />
          </div>
          <p>Проверено <strong>{metrics.retrieval.queries}</strong> запросов. Собственное обращение и точные дубли исключены: <strong>{metrics.retrieval.duplicate_groups_excluded ? 'да' : 'нет'}</strong>.</p>
          <small>{metrics.retrieval.method}</small>
        </section>

        <section className="quality-card compact">
          <h2>Историческая SLA‑аналитика</h2>
          <p><strong>{metrics.sla.overdue_count}</strong> просроченных обращений · медиана {Math.round(metrics.sla.median_duration_seconds / 3600)} ч · P90 {Math.round(metrics.sla.p90_duration_seconds / 3600)} ч.</p>
          <small>{metrics.sla.limitation}</small>
        </section>
      </div>

      <section className="limitations">
        <h2>Ограничения и условия применения</h2>
        {Object.entries(metrics.limitations).map(([key, value]) => <div key={key}><CheckCircle2 size={18} /><p><strong>{key}</strong>{value}</p></div>)}
      </section>
    </div>
  )
}

function MetricTable({ values }: { values: Record<string, ClassMetric> }) {
  return <div className="metric-table"><div><span>Класс</span><span>Precision</span><span>Recall</span><span>F1</span><span>N</span></div>{Object.entries(values).map(([label, row]) => <div key={label}><strong>{label}</strong><span>{percent(row.precision)}</span><span>{percent(row.recall)}</span><span>{percent(row.f1)}</span><span>{row.support}</span></div>)}</div>
}
