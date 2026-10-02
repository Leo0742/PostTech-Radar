import type { Ref } from 'react'
import { Maximize2, Minimize2, PanelLeftClose } from 'lucide-react'
import type { IncomingTicket } from '../../types'
import { ticketStateClass, ticketStateLabel } from './ticketState'

export type QueueFilter = 'pending' | 'reviewed'

export function TicketQueue({
  tickets,
  selectedId,
  filter,
  onFilterChange,
  onSelect,
  onCollapse,
  expanded,
  onExpandedChange,
  sectionRef,
}: {
  tickets: IncomingTicket[]
  selectedId: number | null
  filter: QueueFilter
  onFilterChange: (filter: QueueFilter) => void
  onSelect: (ticket: IncomingTicket) => void
  onCollapse: () => void
  expanded: boolean
  onExpandedChange: () => void
  sectionRef?: Ref<HTMLElement>
}) {
  const filters: [QueueFilter, string][] = [
    ['pending', 'Не проверено'],
    ['reviewed', 'Проверено'],
  ]

  return (
    <section ref={sectionRef} className={expanded ? 'queue-panel ticket-queue queue-expanded-panel' : 'queue-panel ticket-queue'} aria-label="Очередь обращений">
      <div className="queue-toolbar">
        <div className="queue-toolbar-title">
          <div><strong>Очередь</strong><span>{tickets.length}</span></div>
          <div className="queue-toolbar-actions">
            <button className="tertiary queue-expand-button" aria-label={expanded ? 'Свернуть очередь' : 'Развернуть очередь на весь экран'} onClick={onExpandedChange}>
              {expanded ? <Minimize2 size={16} /> : <Maximize2 size={16} />}{expanded ? 'Свернуть' : 'На весь экран'}
            </button>
            {!expanded && <button className="tertiary queue-collapse-button" aria-label="Скрыть очередь" onClick={onCollapse}><PanelLeftClose size={16} />Скрыть</button>}
          </div>
        </div>
        <div className="queue-filters" aria-label="Фильтр очереди">
          {filters.map(([id, label]) => (
            <button key={id} className={filter === id ? 'active' : ''} onClick={() => onFilterChange(id)}>{label}</button>
          ))}
        </div>
      </div>
      <div className="queue-table" role="table" aria-label="Список обращений">
        {expanded && <div className="queue-expanded-head" role="row"><span>Обращение</span><span>Текст</span><span>Категория</span><span>Статус</span></div>}
        {tickets.map((ticket) => (
          <button
            role="row"
            key={ticket.id}
            className={selectedId === ticket.id ? 'queue-row selected' : 'queue-row'}
            onClick={() => onSelect(ticket)}
          >
            {expanded ? <>
              <span className="queue-expanded-id"><strong>{ticket.request_id ? `#${ticket.request_id}` : `Строка ${ticket.row_number}`}</strong></span>
              <small className="queue-description">{ticket.description || 'Без описания'}</small>
              <span className="queue-category">{ticket.model_category ?? '—'}</span>
              <span className={ticketStateClass(ticket)}>{ticketStateLabel(ticket)}</span>
            </> : <>
              <span className="queue-row-top">
                <strong>{ticket.request_id ? `#${ticket.request_id}` : `Строка ${ticket.row_number}`}</strong>
                <span className={ticketStateClass(ticket)}>{ticketStateLabel(ticket)}</span>
              </span>
              <span className="queue-category">{ticket.model_category ?? '—'}</span>
            </>}
          </button>
        ))}
        {!tickets.length && <div className="queue-empty">В этом фильтре обращений нет.</div>}
      </div>
    </section>
  )
}
