import { AlertTriangle, LoaderCircle, RefreshCw } from 'lucide-react'
import { useState } from 'react'
import { api } from '../../lib/api'
import type { QwenRecheckResult } from '../../types'

export type QwenRecheckPayload = Record<string, string>

function operatorErrorMessage(reason: unknown) {
  const detail = reason instanceof Error ? reason.message.toLowerCase() : ''
  if (detail.includes('timeout') || detail.includes('timed out') || detail.includes('exceeded')) {
    return 'Дополнительная проверка не завершилась. Основная рекомендация остаётся доступной.'
  }
  return 'Дополнительная проверка временно недоступна. Основная рекомендация остаётся доступной.'
}

export function QwenRecheck({
  payload,
  liteLabel,
  compact = false,
}: {
  payload: QwenRecheckPayload
  liteLabel: string
  compact?: boolean
}) {
  const [result, setResult] = useState<QwenRecheckResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const alternatives = result
    ? result.category.alternatives.filter((item) => item.label !== result.category.label).slice(0, 2)
    : []

  async function run() {
    setLoading(true)
    setError('')
    setResult(null)
    try {
      setResult(await api<QwenRecheckResult>('/api/tickets/recheck-qwen', {
        method: 'POST',
        body: JSON.stringify(payload),
      }))
    } catch (reason) {
      setError(operatorErrorMessage(reason))
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className={`qwen-recheck${compact ? ' compact-recheck' : ''}`} aria-label="Дополнительная проверка">
      <button className="secondary qwen-recheck-button" disabled={loading} onClick={() => void run()}>
        {loading ? <LoaderCircle className="spin" size={16} /> : <RefreshCw size={16} />}
        {loading ? 'Проверяем…' : 'Дополнительная проверка'}
      </button>

      {loading && <div className="qwen-progress" aria-hidden><span /></div>}

      {error && <div className="qwen-error" role="status"><AlertTriangle size={16} /><span>{error}</span></div>}

      {result && (
        <div className="qwen-operator-result recheck-comparison">
          <span>Результат дополнительной проверки</span>
          <div className="recheck-comparison-grid">
            <div><small>Основная рекомендация</small><strong>{liteLabel}</strong></div>
            <div><small>Дополнительная проверка</small><h3>{result.category.label}</h3></div>
          </div>
          {alternatives.length > 0 && <div className="qwen-alternatives">
            <small>Другие варианты</small>
            <div className="compact-top3">
              {alternatives.map((item, index) => (
                <div key={item.label + '-qwen-' + index}><strong>{item.label}</strong></div>
              ))}
            </div>
          </div>}
          <small>{result.category.label === liteLabel ? 'Результаты совпадают.' : 'Получен другой вариант. Итоговое решение выбирает оператор.'}</small>
        </div>
      )}
    </section>
  )
}
