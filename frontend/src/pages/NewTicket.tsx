import { AlertTriangle, ArrowUpRight, CheckCircle2, Clock3, LoaderCircle, Route, Search, ShieldCheck, Sparkles } from 'lucide-react'
import { useState } from 'react'
import { QwenRecheck } from '../components/tickets/QwenRecheck'
import { api } from '../lib/api'
import type { AnalysisResult, DatasetOptions } from '../types'

interface FormState {
  description: string
  service: string
  component: string
  request_type: string
  criticality: string
  urgency: string
  priority: string
  service_class: string
  timezone: string
}

type ReviewState = 'pending' | 'confirmed' | 'correcting' | 'corrected'

const initial: FormState = {
  description: '',
  service: '',
  component: '',
  request_type: '',
  criticality: '',
  urgency: '',
  priority: '',
  service_class: '',
  timezone: '',
}

function SelectField({ label, value, values, onChange }: { label: string; value: string; values: string[]; onChange: (value: string) => void }) {
  return (
    <label className="field">
      <span>{label} <small>необязательно</small></span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        <option value="">Не указано</option>
        {values.map((item) => <option key={item}>{item}</option>)}
      </select>
    </label>
  )
}

export function NewTicket({ options, onOpenTicket }: { options: DatasetOptions; onOpenTicket: (id: string) => void }) {
  const [form, setForm] = useState<FormState>(initial)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [reviewState, setReviewState] = useState<ReviewState>('pending')
  const [correctedCategory, setCorrectedCategory] = useState('')

  const update = (key: keyof FormState, value: string) => setForm((current) => ({ ...current, [key]: value }))
  const metadataCount = Object.entries(form).filter(([key, value]) => key !== 'description' && value.trim()).length
  const descriptionReady = form.description.trim().length >= 10
  const alternatives = result
    ? result.category.alternatives.filter((item) => item.label !== result.category.label).slice(0, 2)
    : []

  async function submit() {
    setError('')
    setResult(null)
    setReviewState('pending')
    setCorrectedCategory('')
    setLoading(true)
    try {
      setResult(await api<AnalysisResult>('/api/tickets/analyze', { method: 'POST', body: JSON.stringify(form) }))
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Не удалось выполнить анализ')
    } finally {
      setLoading(false)
    }
  }

  function clear() {
    setForm(initial)
    setResult(null)
    setError('')
    setReviewState('pending')
    setCorrectedCategory('')
  }

  return (
    <div className="page page-new">
      <div className="page-heading">
        <div>
          <span className="eyebrow">Рабочий сценарий</span>
          <h1>Новое обращение</h1>
          <p>Введите известные данные. Модель предложит варианты, а решение остаётся за оператором.</p>
        </div>
        <span className="privacy-note"><ShieldCheck size={15} /> Локальный анализ</span>
      </div>

      <div className="analysis-layout">
        <section className="form-panel">
          <div className="panel-heading form-panel-heading">
            <div>
              <h2>Данные обращения</h2>
              <p>Описание обязательно. Остальные поля улучшают контекст, если они уже известны.</p>
            </div>
            <span>{metadataCount}/8 доп. полей</span>
          </div>

          <label className="field primary-field">
            <span>Описание обращения <b>обязательно</b></span>
            <textarea
              aria-label="Описание обращения"
              maxLength={5000}
              value={form.description}
              onChange={(event) => update('description', event.target.value)}
              placeholder="Например: не подключается QR-код при получении отправления, повторный вход не помог"
            />
            <small>{form.description.length} / 5000 {form.description.length > 0 && !descriptionReady ? '· минимум 10 символов' : ''}</small>
          </label>

          <div className="form-grid">
            <SelectField label="Услуга" value={form.service} values={options.services} onChange={(value) => update('service', value)} />
            <SelectField label="Компонент" value={form.component} values={options.components} onChange={(value) => update('component', value)} />
            <SelectField label="Тип запроса" value={form.request_type} values={options.request_types} onChange={(value) => update('request_type', value)} />
            <SelectField label="Приоритет" value={form.priority} values={options.priorities} onChange={(value) => update('priority', value)} />
            <SelectField label="Критичность" value={form.criticality} values={options.criticalities} onChange={(value) => update('criticality', value)} />
            <SelectField label="Срочность" value={form.urgency} values={options.urgencies} onChange={(value) => update('urgency', value)} />
            <SelectField label="Класс обслуживания" value={form.service_class} values={options.service_classes} onChange={(value) => update('service_class', value)} />
            <SelectField label="Часовой пояс" value={form.timezone} values={options.timezones} onChange={(value) => update('timezone', value)} />
          </div>

          {error && <div className="alert error"><AlertTriangle size={18} />{error}</div>}
          <div className="form-actions">
            <button className="primary" disabled={loading || !descriptionReady} onClick={submit}>
              {loading ? <LoaderCircle className="spin" size={18} /> : <Sparkles size={18} />}
              {loading ? 'Анализируем…' : 'Анализировать'}
            </button>
            <button className="secondary" onClick={clear}>Очистить</button>
          </div>
        </section>

        <section className="result-panel" aria-live="polite">
          <div className="result-title">
            <div>
              <span className="eyebrow">Поддержка решения</span>
              <h2>Рекомендация модели</h2>
            </div>
            {result && (
              <span className={result.category.review_required ? 'result-state review' : 'result-state ready'}>
                {result.category.review_required ? <AlertTriangle size={16} /> : <CheckCircle2 size={16} />}
                {result.category.review_required ? 'нужна проверка' : 'расчёт завершён'}
              </span>
            )}
          </div>

          {!result && !loading && (
            <div className="empty-result">
              <Search size={30} />
              <h3>Результат появится здесь</h3>
              <p>После анализа покажем основную категорию, другие подходящие варианты, маршрут и похожие обращения.</p>
            </div>
          )}

          {loading && (
            <div className="empty-result">
              <LoaderCircle className="spin" size={30} />
              <h3>Сопоставляем с историей</h3>
              <p>Проверяем категорию, маршрут и похожие случаи по накопленной истории.</p>
            </div>
          )}

          {result && (
            <>
              <article className={result.category.review_required ? 'prediction review' : 'prediction accepted'}>
                <div className="prediction-copy">
                  <span>Рекомендация категории</span>
                  <h3>{result.category.label === 'UNKNOWN_NEW_ISSUE' ? 'Новая / неизвестная проблема' : result.category.label}</h3>
                  <p className="prediction-confidence">Уверенность модели: {Math.round(result.category.confidence * 100)}%</p>
                  <p>
                    {result.category.review_required
                      ? 'Рекомендацию стоит проверить перед подтверждением.'
                      : 'Основной вариант найден. Итоговое решение подтверждает оператор.'}
                  </p>
                </div>
              </article>

              {alternatives.length > 0 && <section className="top3-block" aria-label="Другие подходящие категории">
                <div className="section-heading">
                  <div><h3>Другие подходящие варианты</h3><p>Используйте их, если основной вариант не подходит по смыслу.</p></div>
                </div>
                <div className="top3-list">
                  {alternatives.map((item, index) => (
                    <div className="top3-row" key={item.label + '-' + index}>
                      <strong>{item.label}</strong>
                    </div>
                  ))}
                </div>
              </section>}

              <QwenRecheck
                liteLabel={result.category.label}
                payload={{
                  description: form.description,
                  service: form.service,
                  component: form.component,
                  request_type: form.request_type,
                  criticality: form.criticality,
                  urgency: form.urgency,
                  priority: form.priority,
                  service_class: form.service_class,
                  timezone: form.timezone,
                }}
              />

              <section className="operator-review">
                <div className="panel-heading">
                  <div><h3>Решение оператора</h3><p>Модель помогает принять решение, но не заменяет его.</p></div>
                  <span>локальный черновик</span>
                </div>
                <div className="review-actions">
                  <button className={reviewState === 'confirmed' ? 'secondary selected' : 'secondary'} onClick={() => { setReviewState('confirmed'); setCorrectedCategory('') }}>
                    <CheckCircle2 size={17} /> Рекомендация подходит
                  </button>
                  <button className={reviewState === 'correcting' || reviewState === 'corrected' ? 'secondary selected' : 'secondary'} onClick={() => setReviewState('correcting')}>
                    Указать другую категорию
                  </button>
                </div>
                {(reviewState === 'correcting' || reviewState === 'corrected') && (
                  <label className="field review-select">
                    <span>Категория оператора</span>
                    <select
                      aria-label="Категория оператора"
                      value={correctedCategory}
                      onChange={(event) => { setCorrectedCategory(event.target.value); setReviewState(event.target.value ? 'corrected' : 'correcting') }}
                    >
                      <option value="">Выберите категорию</option>
                      {options.categories.map((item) => <option key={item}>{item}</option>)}
                    </select>
                  </label>
                )}
                {reviewState !== 'pending' && (
                  <div className="draft-note">
                    <AlertTriangle size={16} />
                    <span>
                      {reviewState === 'confirmed'
                        ? 'В текущем окне отмечено: оператор согласен с рекомендацией.'
                        : correctedCategory ? 'В текущем окне выбрано: ' + correctedCategory + '.' : 'Выберите категорию оператора.'}
                      {' '}API сохранения решения пока не подключён — в историю это действие не записывается.
                    </span>
                  </div>
                )}
              </section>

              <div className="result-grid">
                <article className={result.routing.review_required ? 'result-section route review-route' : 'result-section route'}>
                  <Route size={21} />
                  <div>
                    <span>Рекомендация маршрута</span>
                    <h3>{result.routing.label}</h3>
                    {result.routing.explanation && <p>{result.routing.explanation}</p>}
                    {result.routing.review_required && <small>Линию стоит проверить перед подтверждением.</small>}
                  </div>
                </article>
                <article className="result-section risk">
                  <Clock3 size={21} />
                  <div>
                    <span>Исторический риск SLA</span>
                    <h3>{result.sla_risk.level}</h3>
                    <p>По истории {result.sla_risk.support_n ?? result.sla_risk.sample_size} похожих обращений</p>
                  </div>
                </article>
              </div>

              <details className="evidence-details">
                <summary>Что учтено в рекомендации</summary>
                <div className="signals">
                  {result.category.signals.length
                    ? result.category.signals.map((signal) => <span key={signal}>{signal}</span>)
                    : <span>Доступные данные обращения</span>}
                </div>
                <p className="method-note">Использованы текст обращения и доступные регистрационные данные.</p>
              </details>

              <section className="similar-section">
                <div className="section-heading">
                  <div><h3>Похожие обращения</h3><p>Реальные закрытые случаи из истории — для проверки контекста и результата.</p></div>
                </div>
                {result.retrieval?.rejected ? (
                  <div className="retrieval-empty">
                    <Search size={24} />
                    <div><h3>{result.retrieval.reason}</h3><p>Достаточно близкого исторического примера не найдено.</p></div>
                  </div>
                ) : result.similar.length ? (
                  <div className="similar-list">
                    {result.similar.map((ticket) => (
                      <button className="similar-row" key={ticket.request_id} onClick={() => onOpenTicket(ticket.request_id)}>
                        <span className="ticket-id">#{ticket.request_id}</span>
                        <span className="ticket-summary"><strong>{ticket.description}</strong><small>{ticket.category} · {ticket.final_line}</small></span>
                        <span className="ticket-outcome">{ticket.result}</span>
                        <ArrowUpRight size={17} />
                      </button>
                    ))}
                  </div>
                ) : (
                  <div className="retrieval-empty"><Search size={24} /><div><h3>Похожие обращения не найдены</h3><p>Результат анализа можно оценить без исторического примера.</p></div></div>
                )}
              </section>
            </>
          )}
        </section>
      </div>
    </div>
  )
}
