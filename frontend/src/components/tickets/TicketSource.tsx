import type { IncomingTicket } from '../../types'

const fieldLabels: Record<string, string> = {
  description: 'Описание',
  service: 'Услуга',
  component: 'Компонент',
  request_type: 'Тип запроса',
  priority: 'Приоритет',
  registration_date: 'Дата регистрации',
  user_name: 'Пользователь',
  criticality: 'Критичность',
  urgency: 'Срочность',
  service_class: 'Класс обслуживания',
  timezone: 'Часовой пояс',
}

const optionalRegistrationFields: [keyof IncomingTicket, string][] = [
  ['registration_date', 'Дата регистрации'],
  ['user_name', 'Пользователь'],
  ['service', 'Услуга'],
  ['component', 'Компонент'],
  ['request_type', 'Тип запроса'],
  ['criticality', 'Критичность'],
  ['urgency', 'Срочность'],
  ['priority', 'Приоритет'],
  ['service_class', 'Класс обслуживания'],
  ['timezone', 'Часовой пояс'],
]

export function TicketSource({ ticket, editable, onEdit }: { ticket: IncomingTicket; editable: boolean; onEdit: () => void }) {
  const primary = [
    ['service', ticket.service],
    ['component', ticket.component],
    ['request_type', ticket.request_type],
    ['priority', ticket.priority],
  ].filter(([, value]) => Boolean(value)) as [string, string][]
  const additional = [
    ['registration_date', ticket.registration_date],
    ['user_name', ticket.user_name],
    ['criticality', ticket.criticality],
    ['urgency', ticket.urgency],
    ['service_class', ticket.service_class],
    ['timezone', ticket.timezone],
  ].filter(([, value]) => Boolean(value)) as [string, string][]
  const missing = optionalRegistrationFields.filter(([key]) => !ticket[key]).map(([, label]) => label)

  return (
    <section className="source-fields ticket-source">
      {ticket.description
        ? <div className="source-description">
            <div className="source-description-head">
              <span>Описание</span>
              {editable && <button className="tertiary compact-button" onClick={onEdit}>Изменить</button>}
            </div>
            <p>{ticket.description}</p>
          </div>
        : <p className="empty-source">Описание не заполнено.</p>}
      {primary.length > 0 && <dl className="source-core-line">{primary.map(([key, value]) => <div key={key}><dt>{fieldLabels[key]}</dt><dd>{value}</dd></div>)}</dl>}
      {additional.length > 0 && (
        <details>
          <summary>Ещё {additional.length} полей</summary>
          <dl className="source-additional-grid">{additional.map(([key, value]) => <div key={key}><dt>{fieldLabels[key]}</dt><dd>{value}</dd></div>)}</dl>
        </details>
      )}
      {missing.length > 0 && <p className="missing-source">Не заполнено полей: {missing.length}</p>}
    </section>
  )
}
