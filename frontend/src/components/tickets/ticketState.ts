import type { IncomingTicket } from '../../types'

export function ticketStateLabel(ticket: IncomingTicket) {
  if (ticket.state === 'saved') return 'Проверено'
  return 'Не проверено'
}

export function ticketStateClass(ticket: IncomingTicket) {
  if (ticket.state === 'saved') return 'queue-state saved'
  return 'queue-state pending'
}
