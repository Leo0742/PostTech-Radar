import { ArrowRight, BarChart3, BrainCircuit, Database, Layers3, Route, Search } from 'lucide-react'
import type { DatasetSummary, NavPage } from '../types'

export function Overview({ summary, onNavigate }: { summary: DatasetSummary | null; onNavigate: (page: NavPage) => void }) {
  const top = summary?.top15?.slice(0, 6) ?? []

  return (
    <div className="page overview">
      <div className="page-heading overview-heading">
        <div>
          <span className="eyebrow">Сводка</span>
          <h1>Обзор работы с обращениями</h1>
          <p>История обращений, рекомендации модели и операционная аналитика в одном локальном контуре.</p>
        </div>
        <button className="primary compact-action" onClick={() => onNavigate('new')}>Новое обращение <ArrowRight size={17} /></button>
      </div>

      <section className="overview-kpis" aria-label="Сводные показатели">
        <article><Database size={19} /><span>Обращений в базе</span><strong>{summary?.records?.toLocaleString('ru-RU') ?? '—'}</strong><small>исторические записи</small></article>
        <article><BrainCircuit size={19} /><span>Категорий в данных</span><strong>{summary?.distinct_categories ?? '—'}</strong><small>основные классы</small></article>
        <article><Layers3 size={19} /><span>Записей в TOP‑15</span><strong>{summary?.top15_records?.toLocaleString('ru-RU') ?? '—'}</strong><small>наиболее частые категории</small></article>
      </section>

      <div className="overview-workspace">
        <section className="overview-primary">
          <div className="section-heading">
            <div><h2>Что помогает оператору</h2><p>Каждый вывод модели остаётся рекомендацией и сопровождается проверяемым контекстом.</p></div>
          </div>
          <div className="capability-list">
            <article><BrainCircuit /><div><h3>Категория и варианты</h3><p>Показываем основной вариант, ближайшие альтернативы и явно отмечаем случаи, где нужна ручная проверка.</p></div></article>
            <article><Route /><div><h3>Маршрут поддержки</h3><p>Рекомендуем линию только по данным, доступным на момент регистрации обращения.</p></div></article>
            <article><Search /><div><h3>Похожие обращения</h3><p>Открываем реальные закрытые случаи, чтобы оператор мог сравнить контекст и результат.</p></div></article>
            <article><BarChart3 /><div><h3>Аналитика процесса</h3><p>Фильтры, SLA, длительность и участие линий рассчитываются по рабочей базе, без витринных чисел.</p></div></article>
          </div>
        </section>

        <section className="overview-data">
          <div className="section-heading"><div><h2>Частые категории</h2><p>Первые шесть категорий по текущему набору данных.</p></div></div>
          {top.length ? (
            <ol>{top.map((item) => <li key={item.name}><span>{item.name}</span><strong>{item.count.toLocaleString('ru-RU')}</strong></li>)}</ol>
          ) : (
            <div className="compact-empty">Сводка набора ещё загружается.</div>
          )}
        </section>
      </div>
    </div>
  )
}
