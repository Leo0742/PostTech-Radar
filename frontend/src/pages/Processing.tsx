import {
  AlertTriangle,
  ArrowLeft,
  Check,
  CheckCircle2,
  ChevronRight,
  FileSpreadsheet,
  Info,
  LoaderCircle,
  PanelLeftOpen,
  Plus,
  Search,
  Upload,
  X,
} from 'lucide-react'
import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react'
import { QwenRecheck } from '../components/tickets/QwenRecheck'
import { TicketQueue, type QueueFilter } from '../components/tickets/TicketQueue'
import { TicketSource } from '../components/tickets/TicketSource'
import { TicketWorkspace } from '../components/tickets/TicketWorkspace'
import { api, formatNumber, uploadExcel } from '../lib/api'
import type { DatasetOptions, IncomingBatch, IncomingTicket } from '../types'

type ManualForm = {
  request_id: string
  registration_date: string
  user: string
  service: string
  component: string
  request_type: string
  description: string
  criticality: string
  urgency: string
  priority: string
  service_class: string
  timezone: string
}

const emptyManual: ManualForm = {
  request_id: '', registration_date: '', user: '', service: '', component: '', request_type: '',
  description: '', criticality: '', urgency: '', priority: '', service_class: '', timezone: '',
}

function manualHasAnalyzableInput(form: ManualForm) {
  return [form.description, form.service, form.component, form.request_type, form.criticality, form.urgency, form.priority, form.service_class, form.timezone]
    .some((value) => value.trim().length > 0)
}

function ticketToForm(ticket: IncomingTicket): ManualForm {
  return {
    request_id: ticket.request_id ?? '',
    registration_date: ticket.registration_date ?? '',
    user: ticket.user_name ?? '',
    service: ticket.service ?? '',
    component: ticket.component ?? '',
    request_type: ticket.request_type ?? '',
    description: ticket.description ?? '',
    criticality: ticket.criticality ?? '',
    urgency: ticket.urgency ?? '',
    priority: ticket.priority ?? '',
    service_class: ticket.service_class ?? '',
    timezone: ticket.timezone ?? '',
  }
}

function formChanged(ticket: IncomingTicket, form: ManualForm) {
  const current = ticketToForm(ticket)
  return (Object.keys(current) as (keyof ManualForm)[]).some((key) => current[key] !== form[key])
}

function requestNoun(count: number) {
  const lastTwo = count % 100
  if (lastTwo >= 11 && lastTwo <= 14) return 'обращений'
  const last = count % 10
  if (last === 1) return 'обращение'
  if (last >= 2 && last <= 4) return 'обращения'
  return 'обращений'
}

export function Processing({ options, onOpenTicket }: { options: DatasetOptions; onOpenTicket: (id: string) => void }) {
  const inputRef = useRef<HTMLInputElement>(null)
  const queueRef = useRef<HTMLElement>(null)
  const [batch, setBatch] = useState<IncomingBatch | null>(null)
  const [tickets, setTickets] = useState<IncomingTicket[]>([])
  const [selectedId, setSelectedId] = useState<number | null>(null)
  const [filter, setFilter] = useState<QueueFilter>('pending')
  const [uploading, setUploading] = useState(false)
  const [analyzing, setAnalyzing] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [manualOpen, setManualOpen] = useState(false)
  const [manual, setManual] = useState<ManualForm>(emptyManual)
  const [categoryEdit, setCategoryEdit] = useState(false)
  const [routeEdit, setRouteEdit] = useState(false)
  const [categoryChoice, setCategoryChoice] = useState('')
  const [routeChoice, setRouteChoice] = useState('')
  const [sourceEditing, setSourceEditing] = useState(false)
  const [sourceDraft, setSourceDraft] = useState<ManualForm>(emptyManual)
  const [confirmSourceReset, setConfirmSourceReset] = useState(false)
  const [initialLoading, setInitialLoading] = useState(true)
  const [queueView, setQueueView] = useState<'split' | 'hidden' | 'full'>(() => {
    const layoutVersion = localStorage.getItem('posttech-processing-layout')
    if (layoutVersion !== 'workspace-v2') {
      localStorage.setItem('posttech-processing-layout', 'workspace-v2')
      localStorage.setItem('posttech-queue-view', 'split')
      return 'split'
    }
    const remembered = localStorage.getItem('posttech-queue-view')
    return remembered === 'hidden' || remembered === 'full' ? remembered : 'split'
  })
  const [mobileTicketOpen, setMobileTicketOpen] = useState(false)
  const queueCollapsed = queueView === 'hidden'
  const queueExpanded = queueView === 'full'

  async function loadBatches(preferred?: string) {
    const list = await api<IncomingBatch[]>('/api/incoming/batches')
    const remembered = preferred ?? localStorage.getItem('posttech-active-batch') ?? ''
    const next = list.find((item) => item.id === remembered) ?? list[0]
    if (next) await loadBatch(next.id)
  }

  async function loadBatch(id: string) {
    const [nextBatch, nextTickets] = await Promise.all([
      api<IncomingBatch>(`/api/incoming/batches/${id}`),
      api<IncomingTicket[]>(`/api/incoming/batches/${id}/tickets`),
    ])
    setBatch(nextBatch)
    setTickets(nextTickets)
    localStorage.setItem('posttech-active-batch', id)
    setSelectedId((current) => {
      if (current && nextTickets.some((item) => item.id === current)) return current
      const first = [...nextTickets].sort(ticketOrder).find((item) => item.state === 'waiting_review')
      return first?.id ?? nextTickets[0]?.id ?? null
    })
    return nextTickets
  }

  useEffect(() => {
    api<IncomingBatch[]>('/api/incoming/batches').then(async (list) => {
      const remembered = localStorage.getItem('posttech-active-batch') ?? ''
      const next = list.find((item) => item.id === remembered) ?? list[0]
      if (!next) return
      const [nextBatch, nextTickets] = await Promise.all([
        api<IncomingBatch>(`/api/incoming/batches/${next.id}`),
        api<IncomingTicket[]>(`/api/incoming/batches/${next.id}/tickets`),
      ])
      setBatch(nextBatch)
      setTickets(nextTickets)
      localStorage.setItem('posttech-active-batch', next.id)
      const first = [...nextTickets].sort(ticketOrder).find((item) => item.state === 'waiting_review')
      setSelectedId(first?.id ?? nextTickets[0]?.id ?? null)
    }).catch((reason: Error) => setError(reason.message)).finally(() => setInitialLoading(false))
  }, [])

  const selected = tickets.find((ticket) => ticket.id === selectedId) ?? null

  function selectTicket(ticket: IncomingTicket) {
    setSelectedId(ticket.id)
    setMobileTicketOpen(true)
    if (queueView === 'full') {
      setQueueView('split')
      localStorage.setItem('posttech-queue-view', 'split')
    }
    setCategoryEdit(false)
    setRouteEdit(false)
    setCategoryChoice(ticket.operator_category ?? ticket.model_category ?? '')
    setRouteChoice(ticket.operator_route ?? ticket.model_route ?? '')
    setSourceEditing(false)
    setConfirmSourceReset(false)
    setSourceDraft(ticketToForm(ticket))
  }

  function backToQueue() {
    setMobileTicketOpen(false)
    requestAnimationFrame(() => queueRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }))
  }

  function toggleQueueView() {
    setQueueView((current) => {
      const next = current === 'hidden' ? 'split' : 'hidden'
      localStorage.setItem('posttech-queue-view', next)
      return next
    })
  }

  function toggleQueueExpanded() {
    setQueueView((current) => {
      const next = current === 'full' ? 'split' : 'full'
      localStorage.setItem('posttech-queue-view', next)
      return next
    })
  }

  const sortedTickets = useMemo(() => [...tickets].sort(ticketOrder), [tickets])
  const visibleTickets = sortedTickets.filter((ticket) => {
    if (filter === 'reviewed') return ticket.state === 'saved'
    return ticket.state !== 'saved'
  })
  function changeQueueFilter(nextFilter: QueueFilter) {
    setFilter(nextFilter)
    const next = sortedTickets.find((ticket) => nextFilter === 'reviewed'
      ? ticket.state === 'saved'
      : ticket.state !== 'saved')
    setSelectedId(next?.id ?? null)
    setCategoryEdit(false)
    setRouteEdit(false)
    setSourceEditing(false)
    setConfirmSourceReset(false)
  }

  async function handleFile(file: File) {
    setError(''); setNotice(''); setUploading(true)
    try {
      const created = await uploadExcel<IncomingBatch>(file)
      await loadBatches(created.id)
      setNotice(`Файл прочитан: найдено ${created.total_rows} обращений.`)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Не удалось загрузить Excel')
    } finally {
      setUploading(false)
    }
  }

  async function analyzeBatch() {
    if (!batch) return
    setError(''); setNotice(''); setAnalyzing(true)
    try {
      await api<IncomingBatch>(`/api/incoming/batches/${batch.id}/analyze`, { method: 'POST' })
      await loadBatch(batch.id)
      setNotice('Анализ завершён. Обращения готовы к проверке оператором.')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Не удалось выполнить анализ')
    } finally {
      setAnalyzing(false)
    }
  }

  async function submitManual() {
    setError(''); setNotice(''); setAnalyzing(true)
    try {
      const created = await api<IncomingBatch>('/api/incoming/manual', { method: 'POST', body: JSON.stringify(manual) })
      await api<IncomingBatch>(`/api/incoming/batches/${created.id}/analyze`, { method: 'POST' })
      await loadBatches(created.id)
      setSelectedId(created.ticket_id ?? null)
      setMobileTicketOpen(true)
      setManual(emptyManual)
      setManualOpen(false)
      setNotice('Обращение добавлено в общую очередь и проанализировано.')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Не удалось добавить обращение')
    } finally {
      setAnalyzing(false)
    }
  }

  function replaceTicket(next: IncomingTicket) {
    setTickets((current) => current.map((item) => item.id === next.id ? next : item))
  }

  function startSourceEdit() {
    if (!selected || selected.state === 'saved') return
    setSourceDraft(ticketToForm(selected))
    setSourceEditing(true)
    setConfirmSourceReset(false)
  }

  async function persistSourceEdit() {
    if (!selected || !formChanged(selected, sourceDraft) || !manualHasAnalyzableInput(sourceDraft)) return
    setSaving(true); setError(''); setNotice('')
    try {
      const patched = await api<IncomingTicket>(`/api/incoming/tickets/${selected.id}`, {
        method: 'PATCH',
        body: JSON.stringify(sourceDraft),
      })
      replaceTicket(patched)
      const analyzed = await api<IncomingTicket>(`/api/incoming/tickets/${selected.id}/analyze`, { method: 'POST' })
      replaceTicket(analyzed)
      setSourceDraft(ticketToForm(analyzed))
      setSourceEditing(false)
      setConfirmSourceReset(false)
      setCategoryEdit(false)
      setRouteEdit(false)
      setCategoryChoice(analyzed.model_category ?? analyzed.analysis?.category.label ?? '')
      setRouteChoice(analyzed.model_route ?? analyzed.analysis?.routing.label ?? '')
      setNotice('Данные обращения обновлены. Рекомендация пересчитана.')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Не удалось сохранить данные и пересчитать рекомендацию')
    } finally {
      setSaving(false)
    }
  }

  function requestSourceSave() {
    if (!selected || !formChanged(selected, sourceDraft)) return
    if (selected.category_confirmed || selected.route_confirmed) {
      setConfirmSourceReset(true)
      return
    }
    void persistSourceEdit()
  }

  async function confirmCategory(value: string) {
    if (!selected || !value.trim()) return
    setSaving(true); setError('')
    try {
      const next = await api<IncomingTicket>(`/api/incoming/tickets/${selected.id}/confirm-category`, {
        method: 'POST', body: JSON.stringify({ category: value.trim() }),
      })
      replaceTicket(next)
      setRouteChoice(next.model_route ?? next.analysis?.routing.label ?? '')
      setCategoryEdit(false)
      setRouteEdit(false)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Не удалось сохранить категорию')
    } finally { setSaving(false) }
  }

  async function confirmRoute(value: string) {
    if (!selected || !value.trim()) return
    setSaving(true); setError('')
    try {
      const next = await api<IncomingTicket>(`/api/incoming/tickets/${selected.id}/confirm-route`, {
        method: 'POST', body: JSON.stringify({ route: value.trim() }),
      })
      replaceTicket(next); setRouteEdit(false)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Не удалось сохранить линию')
    } finally { setSaving(false) }
  }

  async function completeSelected() {
    if (!selected || !batch) return
    setSaving(true); setError(''); setNotice('')
    try {
      const result = await api<{ request_id: string }>(`/api/incoming/tickets/${selected.id}/complete`, { method: 'POST' })
      const fresh = await loadBatch(batch.id)
      const next = [...fresh].sort(ticketOrder).find((item) => item.state !== 'saved')
      setFilter('pending')
      setSelectedId(next?.id ?? null)
      setNotice(`Обращение #${result.request_id} сохранено в базе.`)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Не удалось сохранить решение')
    } finally { setSaving(false) }
  }

  return (
    <div className="page processing-page">
      <div className="page-heading processing-heading">
        <div className="processing-title-block">
          <h1>Обращения</h1>
          {!batch && <p>Загрузите обращения и проверьте рекомендации.</p>}
        </div>
        {batch && <div className="processing-header-actions">
          <button className="secondary compact-header-action" onClick={() => inputRef.current?.click()}><Upload size={16} />Загрузить</button>
          <button className="secondary compact-header-action" onClick={() => setManualOpen(true)}><Plus size={16} />Обращение</button>
        </div>}
      </div>

      {error && <div className="alert error" aria-live="polite"><AlertTriangle size={18} />{error}</div>}
      {notice && <div className="alert success" aria-live="polite"><CheckCircle2 size={18} />{notice}</div>}

      {initialLoading && !manualOpen && <div className="processing-loading"><LoaderCircle className="spin" size={18} />Загружаем обращения…</div>}

      {!initialLoading && !batch && !manualOpen && (
        <section className="processing-empty" onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); const file = event.dataTransfer.files[0]; if (file) void handleFile(file) }}>
          <div className="upload-icon"><FileSpreadsheet size={32} /></div>
          <h2>Добавьте обращения</h2>
          <p>Загрузите Excel или добавьте одно обращение вручную. Для анализа достаточно описания или доступных регистрационных данных.</p>
          <div className="empty-actions">
            <button className="primary" onClick={() => inputRef.current?.click()} disabled={uploading}>{uploading ? <LoaderCircle className="spin" size={18} /> : <Upload size={18} />}{uploading ? 'Читаем файл…' : 'Загрузить Excel'}</button>
            <button className="secondary" onClick={() => setManualOpen(true)}><Plus size={18} />Добавить вручную</button>
          </div>
          <small>Поддерживается .xlsx. Для .xls приложение предложит сохранить файл в новом формате.</small>
        </section>
      )}

      <input ref={inputRef} className="visually-hidden" type="file" accept=".xlsx,.xls" onChange={(event) => { const file = event.target.files?.[0]; if (file) void handleFile(file); event.currentTarget.value = '' }} />

      {manualOpen && <ManualFormPanel form={manual} options={options} busy={analyzing} onChange={(key, value) => setManual((current) => ({ ...current, [key]: value }))} onClose={() => setManualOpen(false)} onSubmit={submitManual} />}

      {!initialLoading && batch && batch.status === 'uploaded' && !manualOpen && (
        <section className="batch-validation">
          <div className="batch-title-row"><div><FileSpreadsheet size={23} /><div><h2>{batch.filename}</h2><p>{batch.total_rows} обращений</p></div></div><button className="secondary" onClick={() => inputRef.current?.click()}>Загрузить другой файл</button></div>
          <div className="validation-list">
            <div><Check size={18} /><span><strong>Файл готов к анализу</strong></span></div>
            <div><Check size={18} /><span>{formatNumber(batch.total_rows)} {requestNoun(batch.total_rows)}</span></div>
            {batch.labeled_historical && <div className="validation-info"><Info size={18} /><span>Историческая разметка найдена и не используется как подсказка модели.</span></div>}
            {(batch.invalid_rows > 0 || !!batch.duplicate_ids?.length || !batch.description_column) && <details className="file-details"><summary>Подробнее о файле</summary><div>
              {!batch.description_column && <p>Текст обращения не найден — анализ будет опираться на доступные регистрационные признаки.</p>}
              {batch.invalid_rows > 0 && <p>{batch.invalid_rows} строк требуют исправления и будут отмечены в очереди.</p>}
              {!!batch.duplicate_ids?.length && <p>Есть повторяющиеся номера обращений. Строки сохранены в очереди.</p>}
            </div></details>}
          </div>
          <div className="batch-actions"><button className="primary" disabled={analyzing || batch.valid_rows === 0} onClick={analyzeBatch}>{analyzing ? <LoaderCircle className="spin" size={18} /> : <ChevronRight size={18} />}{analyzing ? 'Анализируем обращения…' : 'Анализировать обращения'}</button><button className="secondary" onClick={() => setManualOpen(true)}><Plus size={18} />Добавить одно обращение</button></div>
          {analyzing && <div className="indeterminate" role="progressbar" aria-label="Анализ обращений"><span /></div>}
        </section>
      )}

      {!initialLoading && batch && batch.status !== 'uploaded' && !manualOpen && (
        <>
          <div className={`review-workspace${queueCollapsed ? ' queue-collapsed' : ''}${queueExpanded ? ' queue-expanded' : ''}${mobileTicketOpen ? ' mobile-ticket-open' : ''}`}>
            <TicketQueue sectionRef={queueRef} tickets={visibleTickets} selectedId={selectedId} filter={filter} onFilterChange={changeQueueFilter} onSelect={selectTicket} onCollapse={toggleQueueView} expanded={queueExpanded} onExpandedChange={toggleQueueExpanded} />
            <TicketWorkspace>
              <ReviewPanel ticket={selected} options={options} saving={saving} categoryEdit={categoryEdit} routeEdit={routeEdit} categoryChoice={categoryChoice} routeChoice={routeChoice} sourceEditing={sourceEditing} sourceDraft={sourceDraft} confirmSourceReset={confirmSourceReset} queueCollapsed={queueCollapsed} onToggleQueue={toggleQueueView} onBackToQueue={backToQueue} onCategoryEdit={setCategoryEdit} onRouteEdit={setRouteEdit} onCategoryChoice={setCategoryChoice} onRouteChoice={setRouteChoice} onSourceEdit={startSourceEdit} onSourceChange={(key, value) => setSourceDraft((current) => ({ ...current, [key]: value }))} onSourceCancel={() => { if (selected) setSourceDraft(ticketToForm(selected)); setSourceEditing(false); setConfirmSourceReset(false) }} onSourceSave={requestSourceSave} onSourceResetCancel={() => setConfirmSourceReset(false)} onSourceResetConfirm={() => { setConfirmSourceReset(false); void persistSourceEdit() }} onConfirmCategory={confirmCategory} onConfirmRoute={confirmRoute} onComplete={completeSelected} onOpenTicket={onOpenTicket} />
            </TicketWorkspace>
          </div>
        </>
      )}
    </div>
  )
}

function ticketOrder(a: IncomingTicket, b: IncomingTicket) {
  const rank = (ticket: IncomingTicket) => ticket.state === 'failed' ? 0 : ticket.state === 'waiting_review' && ticket.attention ? 1 : ticket.state === 'waiting_review' ? 2 : ticket.state === 'uploaded' ? 3 : 4
  return rank(a) - rank(b) || a.id - b.id
}

function ManualFormPanel({ form, options, busy, onChange, onClose, onSubmit }: { form: ManualForm; options: DatasetOptions; busy: boolean; onChange: (key: keyof ManualForm, value: string) => void; onClose: () => void; onSubmit: () => void }) {
  return <section className="manual-panel"><div className="manual-heading"><div><h2>Добавить одно обращение</h2><p>После анализа оно попадёт в ту же очередь проверки.</p></div><button aria-label="Закрыть" className="icon-button" onClick={onClose}><X size={20} /></button></div><label className="field primary-field"><span>Описание</span><textarea value={form.description} onChange={(event) => onChange('description', event.target.value)} placeholder="Опишите проблему пользователя" /><small>Короткий текст допустим. Если текста нет, укажите доступные регистрационные признаки.</small></label><div className="manual-grid"><TextField label="Номер запроса" value={form.request_id} onChange={(value) => onChange('request_id', value)} /><TextField label="Дата регистрации" type="datetime-local" value={form.registration_date} onChange={(value) => onChange('registration_date', value)} /><TextField label="Пользователь" value={form.user} onChange={(value) => onChange('user', value)} /><SelectField label="Услуга" value={form.service} values={options.services} onChange={(value) => onChange('service', value)} /><SelectField label="Компонент" value={form.component} values={options.components} onChange={(value) => onChange('component', value)} /><SelectField label="Тип запроса" value={form.request_type} values={options.request_types} onChange={(value) => onChange('request_type', value)} /><SelectField label="Приоритет" value={form.priority} values={options.priorities} onChange={(value) => onChange('priority', value)} /><SelectField label="Критичность" value={form.criticality} values={options.criticalities} onChange={(value) => onChange('criticality', value)} /><SelectField label="Срочность" value={form.urgency} values={options.urgencies} onChange={(value) => onChange('urgency', value)} /><SelectField label="Класс обслуживания" value={form.service_class} values={options.service_classes} onChange={(value) => onChange('service_class', value)} /><SelectField label="Часовой пояс" value={form.timezone} values={options.timezones} onChange={(value) => onChange('timezone', value)} /></div><div className="manual-actions"><button className="secondary" onClick={onClose}>Отмена</button><button className="primary" disabled={busy || !manualHasAnalyzableInput(form)} onClick={onSubmit}>{busy ? <LoaderCircle className="spin" size={18} /> : <ChevronRight size={18} />}{busy ? 'Анализируем…' : 'Добавить и анализировать'}</button></div></section>
}

function TextField({ label, value, onChange, type = 'text' }: { label: string; value: string; onChange: (value: string) => void; type?: string }) { return <label className="field"><span>{label}</span><input type={type} value={value} onChange={(event) => onChange(event.target.value)} /></label> }
function SelectField({ label, value, values, onChange }: { label: string; value: string; values: string[]; onChange: (value: string) => void }) { return <label className="field"><span>{label}</span><select value={value} onChange={(event) => onChange(event.target.value)}><option value="">Не указано</option>{values.map((item) => <option key={item}>{item}</option>)}</select></label> }

type ReviewPanelProps = {
  ticket: IncomingTicket | null
  options: DatasetOptions
  saving: boolean
  categoryEdit: boolean
  routeEdit: boolean
  categoryChoice: string
  routeChoice: string
  sourceEditing: boolean
  sourceDraft: ManualForm
  confirmSourceReset: boolean
  queueCollapsed: boolean
  onToggleQueue: () => void
  onBackToQueue: () => void
  onCategoryEdit: (value: boolean) => void
  onRouteEdit: (value: boolean) => void
  onCategoryChoice: (value: string) => void
  onRouteChoice: (value: string) => void
  onSourceEdit: () => void
  onSourceChange: (key: keyof ManualForm, value: string) => void
  onSourceCancel: () => void
  onSourceSave: () => void
  onSourceResetCancel: () => void
  onSourceResetConfirm: () => void
  onConfirmCategory: (value: string) => Promise<void>
  onConfirmRoute: (value: string) => Promise<void>
  onComplete: () => Promise<void>
  onOpenTicket: (id: string) => void
}

function ReviewPanel({
  ticket, options, saving, categoryEdit, routeEdit, categoryChoice, routeChoice,
  sourceEditing, sourceDraft, confirmSourceReset, queueCollapsed, onToggleQueue, onBackToQueue, onCategoryEdit, onRouteEdit,
  onCategoryChoice, onRouteChoice, onSourceEdit, onSourceChange, onSourceCancel,
  onSourceSave, onSourceResetCancel, onSourceResetConfirm, onConfirmCategory,
  onConfirmRoute, onComplete, onOpenTicket,
}: ReviewPanelProps) {
  if (!ticket) {
    return <aside className="review-panel review-empty"><Search size={28} /><h3>Выберите обращение</h3><p>Справа появятся данные, рекомендация и действия оператора.</p></aside>
  }

  const saved = ticket.state === 'saved'
  const analysis = ticket.analysis
  const categoryFinal = ticket.operator_category ?? ticket.model_category ?? analysis?.category.label ?? ''
  const routeFinal = ticket.operator_route ?? ticket.model_route ?? analysis?.routing.label ?? ''
  const suggestedCategories = analysis
    ? [analysis.category.label, ...analysis.category.alternatives.map((item) => item.label)]
        .filter((item, index, items) => item && items.indexOf(item) === index)
        .slice(0, 3)
    : []
  const showSlaRisk = analysis
    ? !['низкий', 'low'].includes(analysis.sla_risk.level.trim().toLowerCase())
    : false

  return (
    <aside className="review-panel" aria-live="polite">
      <button className="tertiary mobile-back-to-queue" onClick={onBackToQueue}><ArrowLeft size={16} />Назад к очереди</button>
      <div className="review-header">
        <div className="review-identity">
          {queueCollapsed && <button className="tertiary queue-view-toggle queue-show-button" aria-label="Показать очередь" onClick={onToggleQueue}><PanelLeftOpen size={17} />Показать очередь</button>}
          <span>{ticket.request_id ? 'Обращение #' + ticket.request_id : 'Строка ' + ticket.row_number}</span>
        </div>
      </div>

      {ticket.low_information && !sourceEditing && <div className="alert warning"><AlertTriangle size={18} />Мало исходной информации — проверьте данные обращения.</div>}
      {ticket.state === 'failed' && !sourceEditing && <div className="alert error"><AlertTriangle size={18} />{ticket.error_text ?? 'Не удалось выполнить анализ.'}</div>}

      <div className="review-layout">
      {sourceEditing
        ? <SourceEditor
            form={sourceDraft}
            options={options}
            busy={saving}
            changed={formChanged(ticket, sourceDraft)}
            showResetConfirm={confirmSourceReset}
            onChange={onSourceChange}
            onCancel={onSourceCancel}
            onSave={onSourceSave}
            onResetCancel={onSourceResetCancel}
            onResetConfirm={onSourceResetConfirm}
          />
        : <TicketSource ticket={ticket} editable={!saved} onEdit={onSourceEdit} />}

      {analysis && !sourceEditing && <>
        <section className="decision-card" aria-labelledby={`decision-${ticket.id}`}>
          <h2 id={`decision-${ticket.id}`}>Решение</h2>

          <div className="decision-row category-decision">
            <div className="decision-main">
              {ticket.category_confirmed === 1 && <div className="decision-label-row"><span className="decision-confirmed"><CheckCircle2 size={14} />Подтверждено</span></div>}
              <h3>{ticket.category_confirmed === 1 ? categoryFinal : (analysis.category.label === 'UNKNOWN_NEW_ISSUE' ? 'Новая / неизвестная проблема' : analysis.category.label)}</h3>
              {ticket.category_confirmed !== 1 && <p className="decision-confidence">Уверенность модели: {Math.round(analysis.category.confidence * 100)}%</p>}
              {ticket.category_confirmed !== 1 && <details className="decision-why">
                <summary>Почему эта категория?</summary>
                <div>
                  {analysis.category.explanation && <p>{analysis.category.explanation}</p>}
                  <small>Уверенность — калиброванная оценка модели. Она показывает силу рекомендации и помогает определить, нужна ли ручная проверка; это не гарантия правильности.</small>
                </div>
              </details>}
              {suggestedCategories.length > 1 && <details className="recommendations-disclosure">
                <summary>Рекомендации системы <span>{suggestedCategories.length}</span></summary>
                <div className="alternative-options">{suggestedCategories.map((item, index) => <span key={item + '-' + index}>{item === 'UNKNOWN_NEW_ISSUE' ? 'Новая / неизвестная проблема' : item}</span>)}</div>
              </details>}
            </div>
            {ticket.category_confirmed === 1 && !saved && !categoryEdit && <button className="tertiary decision-inline-edit" onClick={() => { onCategoryChoice(categoryFinal); onCategoryEdit(true) }}>Изменить</button>}
          </div>

          {!saved && ticket.category_confirmed !== 1 && <QwenRecheck
            key={`${ticket.id}-${ticket.prediction_timestamp ?? ''}`}
            liteLabel={analysis.category.label}
            compact
            payload={{
              description: ticket.description ?? '', registration_date: ticket.registration_date ?? '', user: ticket.user_name ?? '',
              service: ticket.service ?? '', component: ticket.component ?? '', request_type: ticket.request_type ?? '', criticality: ticket.criticality ?? '',
              urgency: ticket.urgency ?? '', priority: ticket.priority ?? '', service_class: ticket.service_class ?? '', timezone: ticket.timezone ?? '',
            }}
          />}

          {!saved && !ticket.category_confirmed && !categoryEdit && <div className="decision-actions current-action"><button className="primary" disabled={saving} onClick={() => void onConfirmCategory(analysis.category.label)}><Check size={18} />Подтвердить категорию</button><button className="secondary" onClick={() => { onCategoryChoice(categoryFinal); onCategoryEdit(true) }}>Изменить</button></div>}
          {!saved && categoryEdit && <div className="choice-editor compact-choice-editor">
            <CategoryCombobox value={categoryChoice} options={options.categories} onChange={onCategoryChoice} />
            <div><button className="secondary" onClick={() => onCategoryEdit(false)}>Отмена</button><button className="primary" disabled={saving || !categoryChoice.trim()} onClick={() => void onConfirmCategory(categoryChoice)}>Подтвердить</button></div>
          </div>}

          <div className="decision-row route-decision">
            <div className="decision-main">
              {ticket.route_confirmed === 1 && <div className="decision-label-row"><span className="decision-confirmed"><CheckCircle2 size={14} />Подтверждено</span></div>}
              <h3>{ticket.route_confirmed === 1 ? routeFinal : analysis.routing.label}</h3>
              {ticket.route_confirmed !== 1 && analysis.routing.explanation && <details className="decision-why">
                <summary>Почему эта линия?</summary>
                <div><p>{analysis.routing.explanation}</p></div>
              </details>}
              {ticket.category_confirmed !== 1 && !saved && <p className="decision-pending-copy">Может обновиться после изменения категории.</p>}
            </div>
            {ticket.route_confirmed === 1 && !saved && !routeEdit && <button className="tertiary decision-inline-edit" onClick={() => { onRouteChoice(routeFinal); onRouteEdit(true) }}>Изменить</button>}
          </div>

          {!saved && ticket.category_confirmed === 1 && !ticket.route_confirmed && !routeEdit && <div className="decision-actions current-action"><button className="primary" disabled={saving} onClick={() => void onConfirmRoute(analysis.routing.label)}><Check size={18} />Подтвердить линию</button><button className="secondary" onClick={() => { onRouteChoice(routeFinal); onRouteEdit(true) }}>Изменить</button></div>}
          {!saved && routeEdit && <div className="choice-editor compact-choice-editor">
            <fieldset className="route-options"><legend>Линия оператора</legend>{options.support_lines.map((item) => <label key={item} className={routeChoice === item ? 'selected' : ''}><input type="radio" name={'route-' + ticket.id} value={item} checked={routeChoice === item} onChange={() => onRouteChoice(item)} /><span>{item}</span></label>)}</fieldset>
            <div><button className="secondary" onClick={() => onRouteEdit(false)}>Отмена</button><button className="primary" disabled={saving || !routeChoice} onClick={() => void onConfirmRoute(routeChoice)}>Подтвердить</button></div>
          </div>}

          {ticket.category_confirmed === 1 && ticket.route_confirmed === 1 && !saved && <div className="review-sticky-actions"><button className="primary final-save" disabled={saving} onClick={() => void onComplete()}>{saving ? <LoaderCircle className="spin" size={18} /> : <CheckCircle2 size={18} />}{saving ? 'Сохраняем…' : 'Сохранить и перейти к следующему'}</button></div>}
        </section>

        {(analysis.similar.length > 0 || showSlaRisk) && <details className="review-context">
          <summary>
            <span>{analysis.similar.length > 0 ? 'Похожие обращения' : 'Дополнительная информация'}</span>
            <span className="review-context-meta">
              {analysis.similar.length > 0 && <small>{analysis.similar.length}</small>}
              {showSlaRisk && <small className="sla-attention">SLA · {analysis.sla_risk.level}</small>}
            </span>
          </summary>
          {showSlaRisk && <div className="review-risk-note"><span>Риск SLA</span><strong>{analysis.sla_risk.level}</strong></div>}
          {analysis.similar.length > 0 && <div className="review-secondary">
            <div className="similar-compact">{analysis.similar.slice(0, 5).map((item) => <button key={item.request_id} onClick={() => onOpenTicket(item.request_id)}><span>#{item.request_id}</span><strong>{item.description}</strong><small>{item.category} · {item.final_line}{item.result ? ` · ${item.result}` : ''}</small><ChevronRight size={15} aria-hidden /></button>)}</div>
          </div>}
        </details>}

        {saved && ticket.saved_request_id && <button className="secondary saved-link" onClick={() => onOpenTicket(ticket.saved_request_id!)}>Открыть в базе обращений <ChevronRight size={17} /></button>}
      </>}
      </div>
    </aside>
  )
}

function SourceEditor({
  form, options, busy, changed, showResetConfirm, onChange, onCancel, onSave, onResetCancel, onResetConfirm,
}: {
  form: ManualForm
  options: DatasetOptions
  busy: boolean
  changed: boolean
  showResetConfirm: boolean
  onChange: (key: keyof ManualForm, value: string) => void
  onCancel: () => void
  onSave: () => void
  onResetCancel: () => void
  onResetConfirm: () => void
}) {
  return <section className="source-editor">
    <div className="source-heading"><div><h3>Данные обращения</h3><p>После сохранения рекомендация будет пересчитана для этого обращения.</p></div></div>
    <label className="field primary-field"><span>Описание</span><textarea value={form.description} onChange={(event) => onChange('description', event.target.value)} /></label>
    <div className="source-edit-grid">
      <TextField label="Номер запроса" value={form.request_id} onChange={(value) => onChange('request_id', value)} />
      <TextField label="Дата регистрации" value={form.registration_date} onChange={(value) => onChange('registration_date', value)} />
      <TextField label="Пользователь" value={form.user} onChange={(value) => onChange('user', value)} />
      <SelectField label="Услуга" value={form.service} values={options.services} onChange={(value) => onChange('service', value)} />
      <SelectField label="Компонент" value={form.component} values={options.components} onChange={(value) => onChange('component', value)} />
      <SelectField label="Тип запроса" value={form.request_type} values={options.request_types} onChange={(value) => onChange('request_type', value)} />
      <SelectField label="Критичность" value={form.criticality} values={options.criticalities} onChange={(value) => onChange('criticality', value)} />
      <SelectField label="Срочность" value={form.urgency} values={options.urgencies} onChange={(value) => onChange('urgency', value)} />
      <SelectField label="Приоритет" value={form.priority} values={options.priorities} onChange={(value) => onChange('priority', value)} />
      <SelectField label="Класс обслуживания" value={form.service_class} values={options.service_classes} onChange={(value) => onChange('service_class', value)} />
      <SelectField label="Часовой пояс" value={form.timezone} values={options.timezones} onChange={(value) => onChange('timezone', value)} />
    </div>
    <div className="source-edit-actions"><button className="secondary" disabled={busy} onClick={onCancel}>Отмена</button><button className="primary" disabled={busy || !changed || !manualHasAnalyzableInput(form)} onClick={onSave}>{busy ? <LoaderCircle className="spin" size={18} /> : <ChevronRight size={18} />}{busy ? 'Пересчитываем…' : 'Сохранить и пересчитать'}</button></div>
    {showResetConfirm && <div className="reset-confirm" role="dialog" aria-modal="true" aria-label="Подтверждение повторного анализа"><p>Изменение исходных данных потребует повторного анализа. Подтверждённые категория и линия будут сброшены.</p><div><button className="secondary" onClick={onResetCancel}>Отмена</button><button className="primary" onClick={onResetConfirm}>Изменить и пересчитать</button></div></div>}
  </section>
}

function CategoryCombobox({ value, options, onChange }: { value: string; options: string[]; onChange: (value: string) => void }) {
  const [query, setQuery] = useState(value)
  const [open, setOpen] = useState(false)
  const [activeIndex, setActiveIndex] = useState(0)
  const [customMode, setCustomMode] = useState(() => Boolean(value) && !options.includes(value))
  const inputRef = useRef<HTMLInputElement>(null)
  const inputId = 'category-combobox'
  const listId = 'category-listbox'

  const filtered = useMemo(() => {
    if (customMode) return []
    const needle = query.trim().toLocaleLowerCase('ru')
    if (!needle || query === value) return options
    return options.filter((item) => item.toLocaleLowerCase('ru').includes(needle))
  }, [customMode, options, query, value])

  function choose(item: string) {
    setCustomMode(false)
    onChange(item)
    setQuery(item)
    setOpen(false)
  }

  function chooseOther() {
    setCustomMode(true)
    onChange('')
    setQuery('')
    setOpen(false)
  }

  function backToList() {
    setCustomMode(false)
    onChange('')
    setQuery('')
    setActiveIndex(0)
    setOpen(true)
    inputRef.current?.focus()
  }

  function keyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (customMode) {
      if (event.key === 'Escape') event.currentTarget.blur()
      return
    }
    const optionCount = filtered.length + 1
    if (event.key === 'ArrowDown') {
      event.preventDefault()
      setOpen(true)
      setActiveIndex((current) => Math.min(current + 1, Math.max(0, optionCount - 1)))
    } else if (event.key === 'ArrowUp') {
      event.preventDefault()
      setOpen(true)
      setActiveIndex((current) => Math.max(current - 1, 0))
    } else if (event.key === 'Enter' && open) {
      event.preventDefault()
      if (activeIndex < filtered.length) choose(filtered[activeIndex])
      else chooseOther()
    } else if (event.key === 'Escape') {
      event.preventDefault()
      setOpen(false)
    }
  }

  const activeDescendant = open
    ? activeIndex < filtered.length ? 'category-option-' + activeIndex : 'category-option-other'
    : undefined

  return <div className="category-combobox">
    <label htmlFor={inputId}>Категория оператора</label>
    <input ref={inputRef} id={inputId} role="combobox" aria-autocomplete={customMode ? 'none' : 'list'} aria-expanded={open} aria-controls={customMode ? undefined : listId} aria-activedescendant={activeDescendant} value={query} maxLength={500}
      onFocus={(event) => { if (!customMode) { event.currentTarget.select(); setOpen(true); setActiveIndex(0) } }}
      onChange={(event) => {
        setQuery(event.target.value)
        if (customMode) onChange(event.target.value)
        else { onChange(''); setOpen(true); setActiveIndex(0) }
      }}
      onKeyDown={keyDown} placeholder={customMode ? 'Введите свою категорию' : 'Выберите или найдите категорию'} />
    {open && <div id={listId} className="category-listbox" role="listbox">
      {filtered.length
        ? filtered.map((item, index) => <button type="button" id={'category-option-' + index} role="option" aria-selected={value === item} className={index === activeIndex ? 'active' : ''} key={item} onMouseDown={(event) => event.preventDefault()} onMouseEnter={() => setActiveIndex(index)} onClick={() => choose(item)}>{item}{value === item && <Check size={16} />}</button>)
        : <div className="combobox-empty">Совпадений нет</div>}
      <button type="button" id="category-option-other" role="option" aria-selected={false} className={`category-other-option${activeIndex === filtered.length ? ' active' : ''}`} onMouseDown={(event) => event.preventDefault()} onMouseEnter={() => setActiveIndex(filtered.length)} onClick={chooseOther}>Другое…</button>
    </div>}
    {customMode && <button type="button" className="tertiary category-back-to-list" onClick={backToList}>Выбрать из списка</button>}
  </div>
}
