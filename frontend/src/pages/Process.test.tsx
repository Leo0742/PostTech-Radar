import { render, screen } from '@testing-library/react'
import { Process } from './Process'

const emptyDurationGroups = {
  single_line: { sample_n: 0, median_seconds: 0, p90_seconds: 0 },
  multi_line: { sample_n: 0, median_seconds: 0, p90_seconds: 0 },
}

beforeEach(() => vi.restoreAllMocks())

test('shows an honest empty state when status history is unavailable', async () => {
  vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({
    status_history: { available: false, message: 'Журнал отсутствует', required_columns: ['request_id', 'status', 'entered_at'], transitions: [], dwell: [] },
    participation: { edge_semantics: 'участие → финальная линия; не хронологический переход', combinations: [], resolver_edges: [], co_participation: [], duration_groups: emptyDurationGroups, multi_line_count: 0, ticket_count: 1931 },
  }), { status: 200 }))
  render(<Process />)
  expect(await screen.findByText('Недоступно для текущей выгрузки')).toBeInTheDocument()
  expect(screen.getByText(/request_id, status, entered_at/)).toBeInTheDocument()
})

test('labels participation edges without claiming chronological transitions', async () => {
  vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({
    status_history: { available: true, message: 'Только события', required_columns: [], event_count: 2, request_count: 1, transitions: [{ from: 'Открыто', to: 'Закрыто', count: 1, share: 1 }], dwell: [] },
    participation: { edge_semantics: 'участие → финальная линия; не хронологический переход', combinations: [{ combination: '1+2', count: 7, overdue_rate: 0.2, median_duration_seconds: 3600 }], resolver_edges: [{ participant: '1', resolver: '2', count: 7 }], co_participation: [], duration_groups: { single_line: { sample_n: 0, median_seconds: 0, p90_seconds: 0 }, multi_line: { sample_n: 7, median_seconds: 3600, p90_seconds: 7200 } }, multi_line_count: 7, ticket_count: 10 },
  }), { status: 200 }))
  render(<Process />)
  expect(await screen.findByText(/не хронологический переход/)).toBeInTheDocument()
  expect(screen.getAllByText('Участвовали: 1, 2 линии').length).toBeGreaterThan(0)
  expect(screen.getByText('2 событий доступно')).toBeInTheDocument()
})
