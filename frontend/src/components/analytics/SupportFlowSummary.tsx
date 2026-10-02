import { AlertTriangle, ArrowRight } from 'lucide-react'
import { percent } from '../../lib/api'
import type { ProcessData } from '../../types'

function duration(seconds: number) {
  if (!seconds) return '—'
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.round((seconds % 3600) / 60)
  return hours ? `${hours} ч ${minutes} мин` : `${minutes} мин`
}

function lineLabel(value: string) {
  const match = String(value).match(/[1-4]/)
  return match ? `${match[0]} линия` : value || 'Не определена'
}

export function SupportFlowSummary({ data, error }: { data: ProcessData | null; error: string }) {
  const participation = data?.participation
  const share = participation?.ticket_count ? participation.multi_line_count / participation.ticket_count : 0
  const edges = participation
    ? [...participation.resolver_edges].sort((a, b) => b.count - a.count).slice(0, 6)
    : []
  const risky = participation
    ? [...participation.combinations]
        .filter((item) => item.count >= 5 && item.combination.includes('+'))
        .sort((a, b) => b.overdue_rate - a.overdue_rate)
        .slice(0, 4)
    : []

  return (
    <section className="analytics-section support-flow-summary" aria-label="Маршрутизация между линиями">
      <div className="section-title-row">
        <div>
          <h2>Маршрутизация между линиями</h2>
          <p>Где к обращению подключается несколько линий поддержки.</p>
        </div>
      </div>

      {error ? (
        <div className="inline-state error"><AlertTriangle size={17} />Не удалось загрузить данные маршрутизации.</div>
      ) : !participation ? (
        <div className="inline-state">Загружаем маршрутизацию…</div>
      ) : (
        <>
          <div className="support-flow-stats">
            <div><strong>{percent(share)}</strong><span>обращений затронули несколько линий</span></div>
            <div><strong>{duration(participation.duration_groups.single_line.median_seconds)}</strong><span>медиана · одна линия</span></div>
            <div><strong>{duration(participation.duration_groups.multi_line.median_seconds)}</strong><span>медиана · несколько линий</span></div>
          </div>

          <div className="support-flow-grid">
            <div className="support-flow-card">
              <h3>Частые направления</h3>
              {edges.length ? <div className="support-edge-list">{edges.map((edge) => (
                <div key={`${edge.participant}-${edge.resolver}`}>
                  <span>{lineLabel(edge.participant)}</span>
                  <ArrowRight size={15} aria-hidden="true" />
                  <span>{lineLabel(edge.resolver)}</span>
                  <strong>{edge.count.toLocaleString('ru-RU')}</strong>
                </div>
              ))}</div> : <p className="compact-empty">Недостаточно данных о линиях.</p>}
            </div>

            <div className="support-flow-card">
              <h3>Комбинации с риском SLA</h3>
              {risky.length ? <div className="support-risk-list">{risky.map((item) => (
                <div key={item.combination}>
                  <span>{item.combination.split('+').join(' → ')} линии</span>
                  <small>{item.count.toLocaleString('ru-RU')} обращений</small>
                  <strong>{percent(item.overdue_rate)}</strong>
                </div>
              ))}</div> : <p className="compact-empty">Нет устойчивых multi-line комбинаций.</p>}
            </div>
          </div>

          <details className="support-flow-help">
            <summary>Как читать данные</summary>
            <p>{participation.edge_semantics}. Это участие линий и финальный исполнитель, а не хронологическая последовательность статусов.</p>
          </details>
        </>
      )}
    </section>
  )
}
