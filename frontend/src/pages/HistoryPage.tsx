import {
  AlertTriangle,
  ArrowLeft,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Database,
  Pencil,
  RotateCcw,
  Search,
  SlidersHorizontal,
  X,
} from 'lucide-react'
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { api } from '../lib/api'
import type { DatasetOptions } from '../types'
import { ToastStack, type ToastMessage } from '../components/ToastStack'

interface TicketListItem {
  request_id: string
  registration_date: string
  service: string
  category: string
  request_type: string
  description: string
  priority: string
  final_line: string
  overdue: boolean
  duration: string | null
  clarifications_count: number
}

interface TicketDetail extends TicketListItem {
  user_name?: string | null
  component?: string | null
  request_type: string
  criticality: string
  urgency: string
  service_class: string
  timezone: string
  status: string
  result: string
  actual_duration: string | null
  actual_duration_seconds: number | null
  raw: Record<string, unknown>
}

interface TicketRevision {
  revision_id: number
  request_id: string
  changed_at: string
  changed_by: string
  changed_fields: string[]
  before: Record<string, unknown>
  after: Record<string, unknown>
}

interface PageResult { items: TicketListItem[]; total: number; page: number; page_size: number; total_pages: number }
const initialFilters = { query: '', service: '', category: '', priority: '', support_line: '', date_from: '', date_to: '' }

type Filters = typeof initialFilters

type TicketDraft = {
  registration_date: string
  user_name: string
  service: string
  component: string
  category: string
  request_type: string
  description: string
  criticality: string
  urgency: string
  priority: string
  service_class: string
  timezone: string
  final_line: string
  result: string
  overdue: boolean
  actual_duration_minutes: string
  clarifications_count: string
}

const fieldNames: Record<string, string> = {
  registration_date: 'Дата регистрации',
  user_name: 'Пользователь',
  service: 'Услуга',
  component: 'Компонент',
  category: 'Категория',
  request_type: 'Тип запроса',
  description: 'Описание',
  criticality: 'Критичность',
  urgency: 'Срочность',
  priority: 'Приоритет',
  service_class: 'Класс обслуживания',
  timezone: 'Часовой пояс',
  final_line: 'Линия поддержки',
  result: 'Результат работ',
  overdue: 'SLA',
  actual_duration_seconds: 'Фактическая длительность',
  clarifications_count: 'Количество уточнений',
}

function detailToDraft(ticket: TicketDetail): TicketDraft {
  return {
    registration_date: ticket.registration_date?.slice(0, 16) ?? '',
    user_name: ticket.user_name ?? '',
    service: ticket.service ?? '',
    component: ticket.component ?? '',
    category: ticket.category ?? '',
    request_type: ticket.request_type ?? '',
    description: ticket.description ?? '',
    criticality: ticket.criticality ?? '',
    urgency: ticket.urgency ?? '',
    priority: ticket.priority ?? '',
    service_class: ticket.service_class ?? '',
    timezone: ticket.timezone ?? '',
    final_line: ticket.final_line ?? '',
    result: ticket.result ?? '',
    overdue: ticket.overdue,
    actual_duration_minutes: ticket.actual_duration_seconds == null ? '' : String(Math.round(ticket.actual_duration_seconds / 60)),
    clarifications_count: String(ticket.clarifications_count ?? 0),
  }
}

function filterLabel(key: keyof Filters, value: string) {
  const labels: Record<keyof Filters, string> = {
    query: 'Поиск', service: 'Услуга', category: 'Категория', priority: 'Приоритет', support_line: 'Линия', date_from: 'С', date_to: 'По',
  }
  return `${labels[key]}: ${value}`
}

export function HistoryPage({ options, selectedId, onSelected }: { options: DatasetOptions; selectedId: string | null; onSelected: (id: string | null) => void }) {
  const [filters, setFilters] = useState(initialFilters)
  const [filtersOpen, setFiltersOpen] = useState(false)
  const [page, setPage] = useState(1)
  const [data, setData] = useState<PageResult>({ items: [], total: 0, page: 1, page_size: 30, total_pages: 0 })
  const [detail, setDetail] = useState<TicketDetail | null>(null)
  const [revisions, setRevisions] = useState<TicketRevision[]>([])
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState<TicketDraft | null>(null)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [refreshKey, setRefreshKey] = useState(0)
  const [toasts, setToasts] = useState<ToastMessage[]>([])

  useEffect(() => {
    const controller = new AbortController()
    const params = new URLSearchParams({ page: String(page), page_size: '30' })
    Object.entries(filters).forEach(([key, value]) => { if (value) params.set(key, value) })
    api<PageResult>(`/api/tickets?${params}`, { signal: controller.signal })
      .then(setData)
      .catch((reason: Error) => { if (reason.name !== 'AbortError') setError(reason.message) })
    return () => controller.abort()
  }, [filters, page, refreshKey])

  useEffect(() => {
    if (!selectedId) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- clearing the detail view when selection is reset
      setDetail(null); setRevisions([]); setEditing(false); return
    }
    let active = true
    Promise.all([
      api<TicketDetail>(`/api/tickets/${encodeURIComponent(selectedId)}`),
      api<TicketRevision[]>(`/api/tickets/${encodeURIComponent(selectedId)}/revisions`),
    ]).then(([nextDetail, nextRevisions]) => {
      if (!active) return
      setDetail(nextDetail)
      setRevisions(nextRevisions)
      setDraft(detailToDraft(nextDetail))
      setError('')
    }).catch((reason: Error) => { if (active) setError(reason.message) })
    return () => { active = false }
  }, [selectedId, refreshKey])

  const set = (key: keyof Filters, value: string) => { setPage(1); setFilters((current) => ({ ...current, [key]: value })) }
  const activeFilters = useMemo(() => Object.entries(filters).filter(([, value]) => Boolean(value)) as [keyof Filters, string][], [filters])

  function notify(text: string, tone: ToastMessage['tone'] = 'success') {
    const id = Date.now() + Math.random()
    setToasts((current) => [...current, { id, text, tone }])
  }

  async function saveTicket() {
    if (!detail || !draft) return
    setSaving(true)
    setError('')
    const payload = {
      registration_date: draft.registration_date,
      user_name: draft.user_name || null,
      service: draft.service,
      component: draft.component || null,
      category: draft.category,
      request_type: draft.request_type,
      description: draft.description,
      criticality: draft.criticality,
      urgency: draft.urgency,
      priority: draft.priority,
      service_class: draft.service_class,
      timezone: draft.timezone,
      final_line: draft.final_line,
      result: draft.result,
      overdue: draft.overdue,
      actual_duration_seconds: draft.actual_duration_minutes === '' ? null : Math.max(0, Number(draft.actual_duration_minutes) * 60),
      clarifications_count: Math.max(0, Number(draft.clarifications_count || 0)),
    }
    try {
      const updated = await api<TicketDetail>(`/api/tickets/${encodeURIComponent(detail.request_id)}`, {
        method: 'PATCH',
        body: JSON.stringify(payload),
      })
      setDetail(updated)
      setDraft(detailToDraft(updated))
      setEditing(false)
      setRefreshKey((current) => current + 1)
      notify('Изменения сохранены')
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : 'Не удалось сохранить изменения'
      setError(message)
      notify(message, 'error')
    } finally {
      setSaving(false)
    }
  }

  if (selectedId) {
    return <div className="page ticket-detail-page">
      <button className="back-link" onClick={() => onSelected(null)}><ArrowLeft size={18} />База обращений</button>
      {error && !detail && <div className="alert error"><AlertTriangle size={18} />{error}</div>}
      {!detail ? <div className="loading-block">Загружаем обращение…</div> : <>
        <header className="ticket-detail-header">
          <div>
            <div className="ticket-identity-row">
              <strong className="ticket-number">#{detail.request_id}</strong>
              <span className="ticket-identity-chip"><small>Линия</small>{detail.final_line || 'Не указана'}</span>
              <span className="ticket-identity-chip"><small>Тип</small>{detail.request_type || 'Не указан'}</span>
            </div>
            <h1>{detail.category || 'Без категории'}</h1>
            <p>{detail.service || 'Услуга не указана'}</p>
          </div>
          <div className="ticket-detail-actions">
            <span className={detail.overdue ? 'status overdue' : 'status ok'}>{detail.overdue ? 'Просрочено' : 'В срок'}</span>
            <button className="secondary" onClick={() => { setDraft(detailToDraft(detail)); setEditing(true) }}><Pencil size={17} />Редактировать</button>
          </div>
        </header>

        <section className="ticket-overview-card" aria-label="Основные данные обращения">
          <dl className="ticket-summary-grid">
            <Fact label="Дата регистрации" value={new Date(detail.registration_date).toLocaleString('ru-RU')} />
            <Fact label="Пользователь" value={detail.user_name || 'Не указан'} />
            <Fact label="Приоритет" value={detail.priority || 'Не указан'} />
            <Fact label="Длительность" value={detail.actual_duration ?? 'Нет данных'} />
          </dl>
          <div className="ticket-story">
            <section>
              <h2>Описание</h2>
              <p className="detail-description">{detail.description || 'Описание отсутствует.'}</p>
            </section>
            <section>
              <h2>Результат</h2>
              <p className="detail-description">{detail.result || 'Результат не указан.'}</p>
            </section>
          </div>
        </section>

        <div className="ticket-detail-lower">
          <details className="detail-disclosure">
            <summary>Дополнительные поля <ChevronDown size={17} /></summary>
            <div className="canonical-grid">
              <Fact label="Компонент" value={detail.component || 'Не указан'} />
              <Fact label="Критичность" value={detail.criticality || 'Не указана'} />
              <Fact label="Срочность" value={detail.urgency || 'Не указана'} />
              <Fact label="Класс обслуживания" value={detail.service_class || 'Не указан'} />
              <Fact label="Часовой пояс" value={detail.timezone || 'Не указан'} />
              <Fact label="Количество уточнений" value={String(detail.clarifications_count)} />
            </div>
            <details className="source-snapshot"><summary>Исходная запись импорта</summary><dl>{Object.entries(detail.raw).map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{String(value ?? '—')}</dd></div>)}</dl></details>
          </details>

          <section className="revision-section">
            <div className="section-title-row"><div><h2>История изменений</h2><p>{revisions.length ? `${revisions.length} ${revisions.length === 1 ? 'изменение' : 'изменений'}` : 'Ручных изменений пока нет'}</p></div></div>
            {revisions.length ? <div className="revision-list">{revisions.map((revision) => <RevisionRow revision={revision} key={revision.revision_id} />)}</div> : <div className="revision-empty"><Check size={18} />Запись соответствует сохранённой версии.</div>}
          </section>
        </div>
      </>}

      {editing && detail && draft && <TicketEditDrawer
        ticket={detail}
        draft={draft}
        options={options}
        saving={saving}
        onChange={(key, value) => setDraft((current) => current ? { ...current, [key]: value } : current)}
        onCancel={() => { setDraft(detailToDraft(detail)); setEditing(false); setError('') }}
        onSave={() => void saveTicket()}
        error={error}
      />}
      <ToastStack messages={toasts} onDismiss={(id) => setToasts((current) => current.filter((item) => item.id !== id))} />
    </div>
  }

  return <div className="page history-page">
    <div className="database-title-row">
      <div><h1>База обращений</h1><p>Подтверждённые обращения и история операторских решений.</p></div>
      <div className="record-count"><strong>{data.total.toLocaleString('ru-RU')}</strong><span>записей</span></div>
    </div>

    <div className="data-toolbar">
      <label className="search-field database-search"><Search size={18} /><input aria-label="Поиск" value={filters.query} onChange={(event) => set('query', event.target.value)} placeholder="Номер или текст обращения" /></label>
      <button className={filtersOpen ? 'secondary selected' : 'secondary'} onClick={() => setFiltersOpen((value) => !value)} aria-expanded={filtersOpen}><SlidersHorizontal size={17} />Фильтры{activeFilters.filter(([key]) => key !== 'query').length > 0 && <b>{activeFilters.filter(([key]) => key !== 'query').length}</b>}</button>
      {activeFilters.length > 0 && <button className="toolbar-reset" onClick={() => { setFilters(initialFilters); setPage(1) }}><RotateCcw size={16} />Сбросить</button>}
    </div>

    {filtersOpen && <section className="database-filter-drawer" aria-label="Фильтры базы обращений">
      <FilterSelect label="Услуга" value={filters.service} values={options.services} onChange={(value) => set('service', value)} />
      <FilterSelect label="Категория" value={filters.category} values={options.categories} onChange={(value) => set('category', value)} />
      <FilterSelect label="Приоритет" value={filters.priority} values={options.priorities} onChange={(value) => set('priority', value)} />
      <FilterSelect label="Линия" value={filters.support_line} values={options.support_lines} onChange={(value) => set('support_line', value)} />
      <label>Дата с<input aria-label="Дата с" type="date" value={filters.date_from} onChange={(event) => set('date_from', event.target.value)} /></label>
      <label>Дата по<input aria-label="Дата по" type="date" value={filters.date_to} onChange={(event) => set('date_to', event.target.value)} /></label>
    </section>}

    {activeFilters.length > 0 && <div className="filter-chips" aria-label="Активные фильтры">{activeFilters.map(([key, value]) => <button key={key} onClick={() => set(key, '')}>{filterLabel(key, value)}<X size={14} /></button>)}</div>}

    {error && <div className="alert error"><AlertTriangle size={18} />{error}</div>}
    {data.items.length ? <div className="database-table-wrap"><div className="database-table" role="table" aria-label="Подтверждённые обращения">
      <div className="database-head" role="row"><span>Обращение</span><span>Дата</span><span>Тип</span><span>Категория</span><span>Линия</span><span>SLA</span></div>
      {data.items.map((ticket) => <button className="database-row" role="row" key={ticket.request_id} onClick={() => onSelected(ticket.request_id)}>
        <span className="database-ticket-cell"><strong className="database-id">#{ticket.request_id}</strong><span className="database-description"><b>{ticket.description}</b><small>{ticket.service || 'Услуга не указана'}</small></span></span>
        <span className="database-date">{new Date(ticket.registration_date).toLocaleDateString('ru-RU')}</span>
        <span className="database-type">{ticket.request_type || '—'}</span>
        <span>{ticket.category}</span>
        <span className="database-line">{ticket.final_line || '—'}</span>
        <span className={ticket.overdue ? 'status overdue' : 'status ok'}>{ticket.overdue ? 'Просрочено' : 'В срок'}</span>
      </button>)}
    </div></div> : <div className="history-empty"><Database size={28} /><h2>Ничего не найдено</h2><p>Измените фильтры или поисковый запрос.</p></div>}

    <div className="pagination"><button disabled={page <= 1} onClick={() => setPage((current) => current - 1)}><ChevronLeft size={16} />Предыдущая</button><span>Страница <strong>{page}</strong> из <strong>{Math.max(data.total_pages, 1)}</strong></span><button aria-label="Следующая" disabled={page >= data.total_pages} onClick={() => setPage((current) => current + 1)}>Следующая<ChevronRight size={16} /></button></div>
    <ToastStack messages={toasts} onDismiss={(id) => setToasts((current) => current.filter((item) => item.id !== id))} />
  </div>
}

function TicketEditDrawer({ ticket, draft, options, saving, onChange, onCancel, onSave, error }: {
  ticket: TicketDetail
  draft: TicketDraft
  options: DatasetOptions
  saving: boolean
  onChange: <K extends keyof TicketDraft>(key: K, value: TicketDraft[K]) => void
  onCancel: () => void
  onSave: () => void
  error: string
}) {
  const firstFieldRef = useRef<HTMLTextAreaElement>(null)
  const returnFocusRef = useRef<HTMLElement | null>(null)
  const routeChanged = draft.final_line !== ticket.final_line
  const categoryChanged = draft.category !== ticket.category
  const saveDisabled = saving || !draft.description.trim() || !draft.category.trim() || !draft.final_line.trim() || !draft.registration_date.trim()
  useEffect(() => {
    returnFocusRef.current = document.activeElement instanceof HTMLElement ? document.activeElement : null
    firstFieldRef.current?.focus()
    return () => returnFocusRef.current?.focus()
  }, [])

  return <div className="drawer-layer" role="dialog" aria-modal="true" aria-label="Редактирование обращения" onKeyDown={(event) => { if (event.key === 'Escape' && !saving) onCancel() }}>
    <button className="drawer-backdrop" aria-label="Закрыть редактирование" onClick={onCancel} />
    <aside className="ticket-edit-drawer">
      <header><div><span>Обращение #{ticket.request_id}</span><h2>Редактирование записи</h2></div><button className="icon-button" aria-label="Закрыть" onClick={onCancel}><X size={20} /></button></header>
      <div className="edit-scroll">
        {error && <div className="alert error"><AlertTriangle size={18} />{error}</div>}
        <EditGroup title="Основное">
          <label className="edit-field full"><span>Описание</span><textarea ref={firstFieldRef} value={draft.description} onChange={(event) => onChange('description', event.target.value)} /></label>
          <label className="edit-field"><span>Дата регистрации</span><input type="datetime-local" value={draft.registration_date} onChange={(event) => onChange('registration_date', event.target.value)} /></label>
          <label className="edit-field"><span>Пользователь</span><input value={draft.user_name} onChange={(event) => onChange('user_name', event.target.value)} /></label>
          <EditSelect label="Услуга" value={draft.service} values={options.services} onChange={(value) => onChange('service', value)} />
          <EditSelect label="Компонент" value={draft.component} values={options.components} allowEmpty onChange={(value) => onChange('component', value)} />
          <EditSelect label="Тип запроса" value={draft.request_type} values={options.request_types} onChange={(value) => onChange('request_type', value)} />
        </EditGroup>

        <EditGroup title="Классификация и маршрут">
          <EditSelect label="Категория" value={draft.category} values={options.categories} onChange={(value) => onChange('category', value)} />
          <EditSelect label="Линия поддержки" value={draft.final_line} values={options.support_lines} onChange={(value) => onChange('final_line', value)} />
          {(categoryChanged || routeChanged) && <div className="edit-change-callout full"><AlertTriangle size={17} /><div><strong>Изменяется подтверждённое решение</strong><span>{categoryChanged && `Категория: ${ticket.category} → ${draft.category}`}{categoryChanged && routeChanged ? ' · ' : ''}{routeChanged && `Линия: ${ticket.final_line} → ${draft.final_line}`}</span></div></div>}
        </EditGroup>

        <EditGroup title="Результат и SLA">
          <label className="edit-field full"><span>Результат работ</span><textarea value={draft.result} onChange={(event) => onChange('result', event.target.value)} /></label>
          <label className="edit-field"><span>Фактическая длительность, мин</span><input type="number" min="0" value={draft.actual_duration_minutes} onChange={(event) => onChange('actual_duration_minutes', event.target.value)} /></label>
          <label className="edit-field"><span>Количество уточнений</span><input type="number" min="0" value={draft.clarifications_count} onChange={(event) => onChange('clarifications_count', event.target.value)} /></label>
          <label className="sla-toggle full"><input type="checkbox" checked={draft.overdue} onChange={(event) => onChange('overdue', event.target.checked)} /><span><strong>Просрочено по SLA</strong><small>Снимите отметку, если обращение закрыто в срок.</small></span></label>
        </EditGroup>

        <EditGroup title="Дополнительные данные">
          <EditSelect label="Критичность" value={draft.criticality} values={options.criticalities} onChange={(value) => onChange('criticality', value)} />
          <EditSelect label="Срочность" value={draft.urgency} values={options.urgencies} onChange={(value) => onChange('urgency', value)} />
          <EditSelect label="Приоритет" value={draft.priority} values={options.priorities} onChange={(value) => onChange('priority', value)} />
          <EditSelect label="Класс обслуживания" value={draft.service_class} values={options.service_classes} onChange={(value) => onChange('service_class', value)} />
          <EditSelect label="Часовой пояс" value={draft.timezone} values={options.timezones} onChange={(value) => onChange('timezone', value)} />
        </EditGroup>
      </div>
      <footer><button className="secondary" onClick={onCancel} disabled={saving}>Отмена</button><button className="primary" onClick={onSave} disabled={saveDisabled}>{saving ? 'Сохраняем…' : 'Сохранить изменения'}</button></footer>
    </aside>
  </div>
}

function EditGroup({ title, children }: { title: string; children: ReactNode }) { return <section className="edit-group"><h3>{title}</h3><div className="edit-grid">{children}</div></section> }
function EditSelect({ label, value, values, onChange, allowEmpty = false }: { label: string; value: string; values: string[]; onChange: (value: string) => void; allowEmpty?: boolean }) {
  const all = value && !values.includes(value) ? [value, ...values] : values
  return <label className="edit-field"><span>{label}</span><select value={value} onChange={(event) => onChange(event.target.value)}>{allowEmpty && <option value="">Не указано</option>}{all.map((item) => <option key={item} value={item}>{item}</option>)}</select></label>
}
function Fact({ label, value, tone }: { label: string; value: string; tone?: 'success' | 'danger' }) { return <div className={`fact${tone ? ` ${tone}` : ''}`}><dt>{label}</dt><dd>{value}</dd></div> }
function FilterSelect({ label, value, values, onChange }: { label: string; value: string; values: string[]; onChange: (value: string) => void }) { return <label>{label}<select aria-label={label} value={value} onChange={(event) => onChange(event.target.value)}><option value="">Все</option>{values.map((item) => <option key={item}>{item}</option>)}</select></label> }
function RevisionRow({ revision }: { revision: TicketRevision }) {
  return <article className="revision-row">
    <div className="revision-marker" />
    <div className="revision-content"><div className="revision-meta"><strong>{revision.changed_by}</strong><span>{new Date(revision.changed_at).toLocaleString('ru-RU')}</span></div>
      <div className="revision-fields">{revision.changed_fields.map((field) => <div key={field}><span>{fieldNames[field] ?? field}</span><strong>{formatRevisionValue(field, revision.before[field])}</strong><ChevronRight size={14} /><b>{formatRevisionValue(field, revision.after[field])}</b></div>)}</div>
    </div>
  </article>
}
function formatRevisionValue(field: string, value: unknown) {
  if (field === 'overdue') return value ? 'Просрочено' : 'В срок'
  if (field === 'actual_duration_seconds' && typeof value === 'number') return `${Math.round(value / 60)} мин`
  return value == null || value === '' ? '—' : String(value)
}
