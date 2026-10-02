import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { HistoryPage } from './HistoryPage'
import type { DatasetOptions } from '../types'

const options: DatasetOptions = { services: ['Почта'], components: [], request_types: [], criticalities: [], urgencies: [], priorities: ['Высокий'], service_classes: [], timezones: [], categories: ['Сбой'], support_lines: ['2 линия'] }

beforeEach(() => vi.restoreAllMocks())

test('sends all filters and supports server-side pagination', async () => {
  const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({ items: [], total: 61, page: 1, page_size: 30, total_pages: 3 }), { status: 200 }))
  render(<HistoryPage options={options} selectedId={null} onSelected={() => undefined} />)
  await screen.findByText('61')
  expect(screen.getByText('записей')).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: /Фильтры/ }))
  fireEvent.change(screen.getByLabelText('Услуга'), { target: { value: 'Почта' } })
  fireEvent.change(screen.getByLabelText('Приоритет'), { target: { value: 'Высокий' } })
  fireEvent.change(screen.getByLabelText('Дата по'), { target: { value: '2026-09-16' } })
  await waitFor(() => expect(fetchMock.mock.calls.at(-1)?.[0]).toContain('service=%D0%9F%D0%BE%D1%87%D1%82%D0%B0'))
  expect(String(fetchMock.mock.calls.at(-1)?.[0])).toContain('priority=%D0%92%D1%8B%D1%81%D0%BE%D0%BA%D0%B8%D0%B9')
  expect(String(fetchMock.mock.calls.at(-1)?.[0])).toContain('date_to=2026-09-16')
  fireEvent.click(screen.getByRole('button', { name: 'Следующая' }))
  await waitFor(() => expect(String(fetchMock.mock.calls.at(-1)?.[0])).toContain('page=2'))
})

test('opens an alphanumeric ticket id', async () => {
  const onSelected = vi.fn()
  vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({ items: [{ request_id: 'INC-A12', registration_date: '2026-09-16T12:00:00', service: 'Почта', category: 'Сбой', request_type: 'Инцидент', description: 'Описание', priority: 'Высокий', final_line: '2 линия', overdue: false, duration: '1 ч', clarifications_count: 0 }], total: 1, page: 1, page_size: 30, total_pages: 1 }), { status: 200 }))
  render(<HistoryPage options={options} selectedId={null} onSelected={onSelected} />)
  expect(await screen.findByText('Инцидент')).toBeInTheDocument()
  expect(screen.getByText('2 линия')).toBeInTheDocument()
  fireEvent.click(await screen.findByText('#INC-A12'))
  expect(onSelected).toHaveBeenCalledWith('INC-A12')
})

test('edit drawer manages keyboard focus and closes with Escape', async () => {
  const detail = {
    request_id: 'INC-A12', registration_date: '2026-09-16T12:00:00', user_name: 'Иван', service: 'Почта', component: null,
    category: 'Сбой', request_type: 'Инцидент', description: 'Не работает отслеживание', criticality: 'Средняя', urgency: 'Средняя',
    priority: 'Высокий', service_class: 'Стандарт', timezone: 'Москва', final_line: '2 линия', overdue: false, duration: '1 ч',
    clarifications_count: 0, status: 'closed', result: 'Исправлено', actual_duration: '1 ч', actual_duration_seconds: 3600, raw: {},
  }
  vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
    const url = String(input)
    if (url.includes('/revisions')) return new Response(JSON.stringify([]), { status: 200 })
    if (url === '/api/tickets/INC-A12') return new Response(JSON.stringify(detail), { status: 200 })
    if (url.startsWith('/api/tickets?')) return new Response(JSON.stringify({ items: [], total: 0, page: 1, page_size: 30, total_pages: 0 }), { status: 200 })
    return new Response('{}', { status: 404 })
  })

  render(<HistoryPage options={options} selectedId="INC-A12" onSelected={() => undefined} />)
  const editButton = await screen.findByRole('button', { name: /Редактировать/ })
  expect(screen.getByText('#INC-A12')).toBeInTheDocument()
  expect(screen.getByText('2 линия')).toBeInTheDocument()
  expect(screen.getByText('Инцидент')).toBeInTheDocument()
  editButton.focus()
  fireEvent.click(editButton)

  const dialog = screen.getByRole('dialog', { name: 'Редактирование обращения' })
  expect(within(dialog).getByLabelText('Описание')).toHaveFocus()
  fireEvent.keyDown(dialog, { key: 'Escape' })
  expect(screen.queryByRole('dialog', { name: 'Редактирование обращения' })).not.toBeInTheDocument()
  expect(editButton).toHaveFocus()
})

test('keeps database controls visible when the server request fails', async () => {
  vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({ detail: 'Сервис временно недоступен' }), { status: 503 }))
  render(<HistoryPage options={options} selectedId={null} onSelected={() => undefined} />)

  expect(await screen.findByText('Сервис временно недоступен')).toBeInTheDocument()
  expect(screen.getByRole('heading', { name: 'База обращений' })).toBeInTheDocument()
  expect(screen.getByLabelText('Поиск')).toBeInTheDocument()
  expect(screen.getByRole('button', { name: /Фильтры/ })).toBeInTheDocument()
})
