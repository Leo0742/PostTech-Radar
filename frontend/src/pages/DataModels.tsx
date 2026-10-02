import { Database, Info, ShieldCheck } from 'lucide-react'
import { useEffect, useState } from 'react'
import { api, formatNumber } from '../lib/api'

interface ModelStatus { trained_at?: string; training_record_count?: number; dataset_sha256?: string; package_versions?: Record<string, string> }
interface StatusData { data: { ticket_count: number; dataset_hash?: string; source_path?: string; imported_at?: string; schema_version?: string; last_import?: { mode: string; inserted: number; updated: number; unchanged: number; total: number } }; models: Record<string, ModelStatus | null> }

export function DataModels() {
  const [status, setStatus] = useState<StatusData | null>(null)
  const [error, setError] = useState('')
  useEffect(() => { api<StatusData>('/api/system/status').then(setStatus).catch((reason: Error) => setError(reason.message)) }, [])
  return <div className="page"><div className="page-heading"><div><h1>Данные и модели</h1><p>Происхождение набора, версии артефактов и безопасный локальный контур</p></div><span className="privacy-note">Только для просмотра</span></div>
    {error && <div className="alert error">{error}</div>}
    {!status ? <div className="loading-block">Читаем метаданные…</div> : <>
      <div className="status-grid"><article className="status-card"><Database /><span>Строк в рабочей базе</span><strong>{formatNumber(status.data.ticket_count)}</strong><small>схема v{status.data.schema_version ?? '—'}</small></article><article className="status-card"><ShieldCheck /><span>SHA‑256 набора</span><strong className="mono">{status.data.dataset_hash?.slice(0, 16) ?? '—'}…</strong><small>{status.data.imported_at ? new Date(status.data.imported_at).toLocaleString('ru-RU') : 'нет даты'}</small></article></div>
      {status.data.last_import && <section className="chart-panel"><div className="panel-heading"><h2>Последний импорт</h2><span>{status.data.last_import.mode}</span></div><p>Добавлено: <strong>{status.data.last_import.inserted}</strong> · обновлено: <strong>{status.data.last_import.updated}</strong> · без изменений: <strong>{status.data.last_import.unchanged}</strong> · всего: <strong>{status.data.last_import.total}</strong></p><small className="muted">Источник: {status.data.source_path}</small></section>}
      <section className="model-status-grid">{Object.entries(status.models).map(([name, model]) => <article className="quality-card" key={name}><div className="panel-heading"><h2>{name}</h2><span>{model ? 'готова' : 'нет артефакта'}</span></div>{model ? <dl className="status-list"><dt>Обучающих строк</dt><dd>{model.training_record_count ?? '—'}</dd><dt>Обучено</dt><dd>{model.trained_at ? new Date(model.trained_at).toLocaleString('ru-RU') : '—'}</dd><dt>Хэш данных</dt><dd className="mono">{model.dataset_sha256?.slice(0, 12) ?? '—'}</dd></dl> : <p>Метаданные отсутствуют.</p>}</article>)}</section>
      <div className="alert info"><Info size={18} />Изменяющие действия намеренно не имитируются. Импорт и переобучение выполняются воспроизводимо командой <code>./scripts/prepare.sh</code>; статус здесь только для просмотра.</div>
    </>}
  </div>
}
