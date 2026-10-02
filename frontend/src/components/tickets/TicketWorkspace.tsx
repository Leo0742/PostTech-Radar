import type { ReactNode } from 'react'

export function TicketWorkspace({ children }: { children: ReactNode }) {
  return <section className="ticket-workspace" aria-label="Работа с обращением">{children}</section>
}
