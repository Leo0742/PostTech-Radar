import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { vi } from 'vitest'
import type { DatasetOptions, IncomingBatch, IncomingTicket } from '../types'
import { Processing } from './Processing'

const options: DatasetOptions = {
  services: ['Почта', 'Логистика'], components: ['Получение'], request_types: ['Инцидент'], criticalities: ['Высокая'], urgencies: ['Высокая'], priorities: ['Высокий'], service_classes: ['Стандарт'], timezones: ['Москва'],
  categories: ['Категория модели', 'Исправленная категория'], support_lines: ['2 линия', '3 линия'],
}

const batch: IncomingBatch = {
  id: 'batch-1', filename: 'incoming.xlsx', status: 'waiting_review', total_rows: 1, valid_rows: 1, invalid_rows: 0,
  recognized_columns: ['Описание'], missing_optional: [], created_at: '2026-09-18T10:00:00Z', state_counts: { waiting_review: 1 },
}

const baseTicket: IncomingTicket = {
  id: 11, batch_id: 'batch-1', row_number: 2, request_id: 'A-1', description: 'Не работает QR-код для получения отправления',
  state: 'waiting_review', category_confirmed: 0, route_confirmed: 0, attention: true, raw: {},
  model_category: 'Категория модели', model_category_confidence: 0.72, model_route: '2 линия', model_route_confidence: 0.81,
  analysis: {
    category: { label: 'Категория модели', confidence: 0.72, accepted: false, review_required: true, threshold: 0.8, margin_threshold: 0.1, alternatives: [{ label: 'Исправленная категория', confidence: 0.18 }], signals: ['qr код'], explanation: 'Исторические признаки.' },
    routing: { label: '2 линия', confidence: 0.81, accepted: true, review_required: false, threshold: 0.5, alternatives: [], signals: [], explanation: 'Категория чаще решается второй линией.' },
    sla_risk: { risk: 0.18, sample_size: 84, level: 'Низкий', explanation: '', support_n: 84 },
    similar: [],
  },
}

beforeEach(() => {
  localStorage.clear()
  vi.restoreAllMocks()
})

test('shows the primary Excel upload empty state', async () => {
  vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify([]), { status: 200 }))
  render(<Processing options={options} onOpenTicket={() => undefined} />)
  expect(await screen.findByRole('heading', { name: 'Добавьте обращения' })).toBeInTheDocument()
  expect(screen.getByRole('button', { name: /Загрузить Excel/ })).toBeInTheDocument()
  expect(screen.getByRole('button', { name: /Добавить вручную/ })).toBeInTheDocument()
})

test('presents the selected ticket as a queue and operator workspace', async () => {
  vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
    const url = String(input)
    if (url === '/api/incoming/batches') return new Response(JSON.stringify([batch]), { status: 200 })
    if (url === '/api/incoming/batches/batch-1') return new Response(JSON.stringify(batch), { status: 200 })
    if (url === '/api/incoming/batches/batch-1/tickets') return new Response(JSON.stringify([baseTicket]), { status: 200 })
    return new Response('{}', { status: 500 })
  })

  render(<Processing options={options} onOpenTicket={() => undefined} />)

  expect(await screen.findByRole('region', { name: 'Очередь обращений' })).toBeInTheDocument()
  expect(within(screen.getByRole('region', { name: 'Очередь обращений' })).queryByText(baseTicket.description)).not.toBeInTheDocument()
  expect(screen.getByRole('region', { name: 'Работа с обращением' })).toBeInTheDocument()
  expect(screen.getAllByText(baseTicket.description).length).toBeGreaterThan(0)
  expect(screen.getByRole('heading', { name: 'Решение' })).toBeInTheDocument()
  expect(screen.getByRole('heading', { name: '2 линия' })).toBeInTheDocument()
  expect(screen.queryByText('Линия поддержки')).not.toBeInTheDocument()
  expect(screen.getByText('Может обновиться после изменения категории.')).toBeInTheDocument()
  expect(screen.getByText('Почему эта категория?')).toBeInTheDocument()
  expect(screen.getByText('Почему эта линия?')).toBeInTheDocument()
  expect(screen.getByRole('button', { name: /Подтвердить категорию/ })).toBeInTheDocument()
  expect(screen.getByRole('button', { name: 'Назад к очереди', hidden: true })).toBeInTheDocument()
  expect(screen.getByRole('button', { name: 'Скрыть очередь' })).toBeInTheDocument()
  expect(screen.queryByText('Требует внимания')).not.toBeInTheDocument()
  expect(screen.getAllByText('Не проверено').length).toBeGreaterThan(0)
})

test('expands the queue to a full-width list with descriptions', async () => {
  vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
    const url = String(input)
    if (url === '/api/incoming/batches') return new Response(JSON.stringify([batch]), { status: 200 })
    if (url === '/api/incoming/batches/batch-1') return new Response(JSON.stringify(batch), { status: 200 })
    if (url === '/api/incoming/batches/batch-1/tickets') return new Response(JSON.stringify([baseTicket]), { status: 200 })
    return new Response('{}', { status: 500 })
  })

  render(<Processing options={options} onOpenTicket={() => undefined} />)
  const queue = await screen.findByRole('region', { name: 'Очередь обращений' })
  const workspace = queue.parentElement

  fireEvent.click(screen.getByRole('button', { name: 'Развернуть очередь на весь экран' }))
  expect(workspace).toHaveClass('queue-expanded')
  expect(queue).toHaveClass('queue-expanded-panel')
  expect(within(queue).getByText(baseTicket.description)).toBeInTheDocument()
  expect(localStorage.getItem('posttech-queue-view')).toBe('full')

  fireEvent.click(screen.getByRole('button', { name: 'Свернуть очередь' }))
  expect(workspace).not.toHaveClass('queue-expanded')
  expect(within(queue).queryByText(baseTicket.description)).not.toBeInTheDocument()
  expect(localStorage.getItem('posttech-queue-view')).toBe('split')
})

test('toggles the desktop queue view and remembers the choice', async () => {
  vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
    const url = String(input)
    if (url === '/api/incoming/batches') return new Response(JSON.stringify([batch]), { status: 200 })
    if (url === '/api/incoming/batches/batch-1') return new Response(JSON.stringify(batch), { status: 200 })
    if (url === '/api/incoming/batches/batch-1/tickets') return new Response(JSON.stringify([baseTicket]), { status: 200 })
    return new Response('{}', { status: 500 })
  })

  render(<Processing options={options} onOpenTicket={() => undefined} />)

  const queue = await screen.findByRole('region', { name: 'Очередь обращений' })
  const workspace = queue.parentElement
  expect(workspace).not.toHaveClass('queue-collapsed')

  fireEvent.click(screen.getByRole('button', { name: 'Скрыть очередь' }))
  expect(workspace).toHaveClass('queue-collapsed')
  expect(screen.getByRole('button', { name: 'Показать очередь' })).toBeInTheDocument()
  expect(localStorage.getItem('posttech-queue-view')).toBe('hidden')

  fireEvent.click(screen.getByRole('button', { name: 'Показать очередь' }))
  expect(workspace).not.toHaveClass('queue-collapsed')
  expect(localStorage.getItem('posttech-queue-view')).toBe('split')
})

test('keeps optional review context compact and hides explanatory clutter', async () => {
  const contextualTicket: IncomingTicket = {
    ...baseTicket,
    analysis: {
      ...baseTicket.analysis!,
      similar: [{
        request_id: '40602830',
        similarity: 0.82,
        description: 'Не могу войти в личный кабинет',
        category: 'Личный кабинет',
        final_line: '1 линия',
        result: 'Решено',
        overdue: false,
        duration: null,
        clarifications: 0,
      }],
    },
  }
  vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
    const url = String(input)
    if (url === '/api/incoming/batches') return new Response(JSON.stringify([batch]), { status: 200 })
    if (url === '/api/incoming/batches/batch-1') return new Response(JSON.stringify(batch), { status: 200 })
    if (url === '/api/incoming/batches/batch-1/tickets') return new Response(JSON.stringify([contextualTicket]), { status: 200 })
    return new Response('{}', { status: 500 })
  })

  render(<Processing options={options} onOpenTicket={() => undefined} />)

  expect(await screen.findByText('Похожие обращения')).toBeInTheDocument()
  expect(screen.queryByText('SLA · Низкий')).not.toBeInTheDocument()
  expect(screen.queryByText('Исторический риск SLA')).not.toBeInTheDocument()
  expect(screen.queryByText('Что учтено')).not.toBeInTheDocument()
  expect(screen.queryByText(/По истории/)).not.toBeInTheDocument()
})

test('keeps the review screen concise without duplicated section headings', async () => {
  const richTicket: IncomingTicket = {
    ...baseTicket,
    service: 'Портал pochta.ru',
    component: 'Личный кабинет',
    request_type: 'Инцидент',
    priority: '(2) Высокий',
    registration_date: '2026-09-18 10:20:00',
    user_name: 'Демо-клиент 5',
    criticality: 'Средняя',
    urgency: 'Высокая',
    service_class: 'Стандарт',
    timezone: 'Москва',
  }
  vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
    const url = String(input)
    if (url === '/api/incoming/batches') return new Response(JSON.stringify([batch]), { status: 200 })
    if (url === '/api/incoming/batches/batch-1') return new Response(JSON.stringify(batch), { status: 200 })
    if (url === '/api/incoming/batches/batch-1/tickets') return new Response(JSON.stringify([richTicket]), { status: 200 })
    return new Response('{}', { status: 500 })
  })

  render(<Processing options={options} onOpenTicket={() => undefined} />)

  expect(await screen.findByText('Обращение #A-1')).toBeInTheDocument()
  expect(screen.queryByText('Проверка обращения')).not.toBeInTheDocument()
  expect(screen.queryByText('Данные обращения')).not.toBeInTheDocument()
  expect(screen.getByText('Ещё 6 полей')).toBeInTheDocument()
})

test('shows required model confidence in the operator workflow', async () => {
  vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
    const url = String(input)
    if (url === '/api/incoming/batches') return new Response(JSON.stringify([batch]), { status: 200 })
    if (url === '/api/incoming/batches/batch-1') return new Response(JSON.stringify(batch), { status: 200 })
    if (url === '/api/incoming/batches/batch-1/tickets') return new Response(JSON.stringify([baseTicket]), { status: 200 })
    return new Response('{}', { status: 500 })
  })

  render(<Processing options={options} onOpenTicket={() => undefined} />)

  expect(await screen.findByRole('heading', { name: 'Категория модели' })).toBeInTheDocument()
  expect(screen.getByText('Исправленная категория')).toBeInTheDocument()
  expect(screen.getByText('Уверенность модели: 72%')).toBeInTheDocument()
  expect(screen.getByText('Почему эта линия?')).toBeInTheDocument()
  expect(screen.getByText('Категория чаще решается второй линией.')).not.toBeVisible()
})

test('shows calm upload success without schema aliases or duplicated technical details', async () => {
  const historicalBatch = {
    ...batch,
    status: 'uploaded',
    filename: 'Обращения_1931.xlsx',
    total_rows: 1931,
    valid_rows: 1931,
    invalid_rows: 0,
    recognized_columns: ['Номер запроса', 'Описание 2', 'Услуга', 'Приоритет'],
    missing_optional: ['component'],
    description_column: 'Описание 2',
    labeled_historical: true,
    state_counts: { uploaded: 1931 },
  } as IncomingBatch
  vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
    const url = String(input)
    if (url === '/api/incoming/batches') return new Response(JSON.stringify([historicalBatch]), { status: 200 })
    if (url === '/api/incoming/batches/batch-1') return new Response(JSON.stringify(historicalBatch), { status: 200 })
    if (url === '/api/incoming/batches/batch-1/tickets') return new Response(JSON.stringify([]), { status: 200 })
    return new Response('{}', { status: 404 })
  })

  render(<Processing options={options} onOpenTicket={() => undefined} />)
  expect(await screen.findByText('Файл готов к анализу')).toBeInTheDocument()
  expect(screen.getByText('1 931 обращение')).toBeInTheDocument()
  expect(screen.getByText(/Историческая разметка найдена/)).toBeInTheDocument()
  expect(screen.queryByText(/Описание 2/)).not.toBeInTheDocument()
  expect(screen.queryByText(/Поле текста/)).not.toBeInTheDocument()
  expect(screen.queryByText(/Регистрационные поля распознаны/)).not.toBeInTheDocument()
  expect(screen.queryByText(/Компонент/)).not.toBeInTheDocument()
})

test('runs category -> route -> save review loop with persisted server state', async () => {
  let ticket = { ...baseTicket }
  const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
    const url = String(input)
    if (url === '/api/incoming/batches') return new Response(JSON.stringify([batch]), { status: 200 })
    if (url === '/api/incoming/batches/batch-1') return new Response(JSON.stringify({ ...batch, status: ticket.state === 'saved' ? 'saved' : 'waiting_review' }), { status: 200 })
    if (url === '/api/incoming/batches/batch-1/tickets') return new Response(JSON.stringify([ticket]), { status: 200 })
    if (url.endsWith('/confirm-category') && init?.method === 'POST') {
      ticket = {
        ...ticket,
        category_confirmed: 1,
        operator_category: 'Категория модели',
        route_confirmed: 0,
        model_route: '3 линия',
        model_route_confidence: 0.74,
        analysis: {
          ...ticket.analysis!,
          routing: {
            ...ticket.analysis!.routing,
            label: '3 линия',
            confidence: 0.74,
            mode: 'confirmed_category',
            category_prior_used: true,
          },
        },
      }
      return new Response(JSON.stringify(ticket), { status: 200 })
    }
    if (url.endsWith('/confirm-route') && init?.method === 'POST') {
      ticket = { ...ticket, route_confirmed: 1, operator_route: '3 линия' }
      return new Response(JSON.stringify(ticket), { status: 200 })
    }
    if (url.endsWith('/complete') && init?.method === 'POST') {
      ticket = { ...ticket, state: 'saved', saved_request_id: 'A-1' }
      return new Response(JSON.stringify({ status: 'saved', request_id: 'A-1' }), { status: 200 })
    }
    return new Response(JSON.stringify({ detail: `unexpected ${url}` }), { status: 500 })
  })

  render(<Processing options={options} onOpenTicket={() => undefined} />)
  expect(await screen.findByRole('heading', { name: 'Категория модели' })).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: /Подтвердить категорию/ }))
  await waitFor(() => expect(screen.getAllByText('Подтверждено').length).toBeGreaterThan(0))
  const recommendations = screen.getByText('Рекомендации системы').closest('details')
  expect(recommendations).not.toHaveAttribute('open')
  expect(screen.getByText('Исправленная категория')).toBeInTheDocument()
  expect(screen.getByRole('heading', { name: '3 линия' })).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: /Подтвердить линию/ }))
  await waitFor(() => expect(screen.getByRole('button', { name: /Сохранить и перейти/ })).toBeInTheDocument())
  fireEvent.click(screen.getByRole('button', { name: /Сохранить и перейти/ }))
  await waitFor(() => expect(screen.getByText('Обращение #A-1 сохранено в базе.')).toBeInTheDocument())
  expect(screen.getByText('В этом фильтре обращений нет.')).toBeInTheDocument()
  expect(screen.queryByText(baseTicket.description)).not.toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: 'Проверено' }))
  expect(screen.getAllByText(baseTicket.description).length).toBeGreaterThan(0)
  expect(fetchMock).toHaveBeenCalledWith('/api/incoming/tickets/11/complete', expect.objectContaining({ method: 'POST' }))
})

test('shows missing optional source fields without blocking review', async () => {
  const partialTicket: IncomingTicket = {
    ...baseTicket,
    registration_date: null,
    user_name: null,
    service: 'Почта',
    component: null,
    request_type: null,
    criticality: null,
    urgency: null,
    priority: null,
    service_class: null,
    timezone: null,
  }
  vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
    const url = String(input)
    if (url === '/api/incoming/batches') return new Response(JSON.stringify([batch]), { status: 200 })
    if (url === '/api/incoming/batches/batch-1') return new Response(JSON.stringify(batch), { status: 200 })
    if (url === '/api/incoming/batches/batch-1/tickets') return new Response(JSON.stringify([partialTicket]), { status: 200 })
    return new Response('{}', { status: 500 })
  })

  render(<Processing options={options} onOpenTicket={() => undefined} />)
  expect(await screen.findByText('Не заполнено полей: 9')).toBeInTheDocument()
  expect(screen.getByRole('button', { name: /Подтвердить категорию/ })).toBeEnabled()
})

test('lets the operator run an additional recommendation check without exposing the model name', async () => {
  const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
    const url = String(input)
    if (url === '/api/incoming/batches') return new Response(JSON.stringify([batch]), { status: 200 })
    if (url === '/api/incoming/batches/batch-1') return new Response(JSON.stringify(batch), { status: 200 })
    if (url === '/api/incoming/batches/batch-1/tickets') return new Response(JSON.stringify([baseTicket]), { status: 200 })
    if (url === '/api/tickets/recheck-qwen' && init?.method === 'POST') {
      return new Response(JSON.stringify({
        category: {
          label: 'Исправленная категория',
          confidence: 0.84,
          alternatives: [
            { label: 'Исправленная категория', confidence: 0.84 },
            { label: 'Категория модели', confidence: 0.11 },
          ],
        },
        model_provenance: { model_id: 'Qwen/Qwen3-Embedding-4B', profile: 'qwen_recheck' },
        latency_seconds: 39.4,
      }), { status: 200 })
    }
    return new Response(JSON.stringify({ detail: `unexpected ${url}` }), { status: 500 })
  })

  render(<Processing options={options} onOpenTicket={() => undefined} />)
  expect(await screen.findByRole('heading', { name: 'Категория модели' })).toBeInTheDocument()

  expect(screen.queryByText(/Qwen/i)).not.toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: /Дополнительная проверка/ }))

  await waitFor(() => expect(screen.getByText('Результат дополнительной проверки')).toBeInTheDocument())
  expect(screen.getByRole('heading', { name: 'Исправленная категория' })).toBeInTheDocument()
  expect(screen.getByRole('heading', { name: 'Категория модели' })).toBeInTheDocument()
  expect(screen.getByText('Уверенность модели: 72%')).toBeInTheDocument()
  expect(fetchMock).toHaveBeenCalledWith('/api/tickets/recheck-qwen', expect.objectContaining({ method: 'POST' }))
})

test('keeps the main recommendation usable while the additional check is running', async () => {
  vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
    const url = String(input)
    if (url === '/api/incoming/batches') return new Response(JSON.stringify([batch]), { status: 200 })
    if (url === '/api/incoming/batches/batch-1') return new Response(JSON.stringify(batch), { status: 200 })
    if (url === '/api/incoming/batches/batch-1/tickets') return new Response(JSON.stringify([baseTicket]), { status: 200 })
    if (url === '/api/tickets/recheck-qwen') return new Promise<Response>(() => undefined)
    return new Response('{}', { status: 500 })
  })

  render(<Processing options={options} onOpenTicket={() => undefined} />)
  expect(await screen.findByRole('heading', { name: 'Категория модели' })).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: /Дополнительная проверка/ }))

  await waitFor(() => expect(screen.getByRole('button', { name: /Проверяем/ })).toBeDisabled())
  expect(screen.getByRole('heading', { name: 'Категория модели' })).toBeInTheDocument()
  expect(screen.getByRole('button', { name: /Подтвердить категорию/ })).toBeEnabled()
})

test('keeps the main workflow usable when the additional check fails', async () => {
  vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
    const url = String(input)
    if (url === '/api/incoming/batches') return new Response(JSON.stringify([batch]), { status: 200 })
    if (url === '/api/incoming/batches/batch-1') return new Response(JSON.stringify(batch), { status: 200 })
    if (url === '/api/incoming/batches/batch-1/tickets') return new Response(JSON.stringify([baseTicket]), { status: 200 })
    if (url === '/api/tickets/recheck-qwen') return new Response(JSON.stringify({ detail: 'Qwen recheck exceeded 180 seconds' }), { status: 503 })
    return new Response('{}', { status: 500 })
  })

  render(<Processing options={options} onOpenTicket={() => undefined} />)
  expect(await screen.findByRole('heading', { name: 'Категория модели' })).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: /Дополнительная проверка/ }))

  expect(await screen.findByText('Дополнительная проверка не завершилась. Основная рекомендация остаётся доступной.')).toBeInTheDocument()
  expect(screen.queryByText(/exceeded 180 seconds/)).not.toBeInTheDocument()
  expect(screen.getByRole('heading', { name: 'Категория модели' })).toBeInTheDocument()
  expect(screen.getByRole('button', { name: /Подтвердить категорию/ })).toBeEnabled()
})

test('edits every registration field and reanalyzes only the selected ticket', async () => {
  let ticket = {
    ...baseTicket,
    registration_date: '2026-09-18 10:00',
    user_name: 'Оператор',
    service: 'Почта',
    component: 'Получение',
    request_type: 'Инцидент',
    criticality: 'Высокая',
    urgency: 'Высокая',
    priority: 'Высокий',
    service_class: 'Стандарт',
    timezone: 'Москва',
  }
  const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
    const url = String(input)
    if (url === '/api/incoming/batches') return new Response(JSON.stringify([batch]), { status: 200 })
    if (url === '/api/incoming/batches/batch-1') return new Response(JSON.stringify(batch), { status: 200 })
    if (url === '/api/incoming/batches/batch-1/tickets') return new Response(JSON.stringify([ticket]), { status: 200 })
    if (url === '/api/incoming/tickets/11' && init?.method === 'PATCH') {
      const body = JSON.parse(String(init.body))
      ticket = {
        ...ticket,
        ...body,
        user_name: body.user,
        state: 'uploaded',
        analysis: null,
        model_category: null,
        model_route: null,
        category_confirmed: 0,
        route_confirmed: 0,
      }
      return new Response(JSON.stringify(ticket), { status: 200 })
    }
    if (url === '/api/incoming/tickets/11/analyze' && init?.method === 'POST') {
      ticket = {
        ...ticket,
        state: 'waiting_review',
        model_category: 'Исправленная категория',
        model_route: '3 линия',
        analysis: {
          ...baseTicket.analysis!,
          category: { ...baseTicket.analysis!.category, label: 'Исправленная категория', confidence: 0.91 },
          routing: { ...baseTicket.analysis!.routing, label: '3 линия', confidence: 0.88 },
        },
      }
      return new Response(JSON.stringify(ticket), { status: 200 })
    }
    return new Response(JSON.stringify({ detail: `unexpected ${url}` }), { status: 500 })
  })

  render(<Processing options={options} onOpenTicket={() => undefined} />)
  const editButtons = await screen.findAllByRole('button', { name: 'Изменить' })
  fireEvent.click(editButtons[0])

  for (const label of ['Описание', 'Номер запроса', 'Дата регистрации', 'Пользователь', 'Услуга', 'Компонент', 'Тип запроса', 'Критичность', 'Срочность', 'Приоритет', 'Класс обслуживания', 'Часовой пояс']) {
    expect(screen.getByLabelText(label)).toBeInTheDocument()
  }
  fireEvent.change(screen.getByLabelText('Описание'), { target: { value: 'Исправленное описание обращения для повторного анализа' } })
  fireEvent.click(screen.getByRole('button', { name: 'Сохранить и пересчитать' }))

  await waitFor(() => expect(screen.getByRole('heading', { name: 'Исправленная категория' })).toBeInTheDocument())
  expect(fetchMock).toHaveBeenCalledWith('/api/incoming/tickets/11', expect.objectContaining({ method: 'PATCH' }))
  expect(fetchMock).toHaveBeenCalledWith('/api/incoming/tickets/11/analyze', expect.objectContaining({ method: 'POST' }))
  expect(screen.getByText('Данные обращения обновлены. Рекомендация пересчитана.')).toBeInTheDocument()
  expect(screen.getByRole('button', { name: /Подтвердить категорию/ })).toBeInTheDocument()
})

test('cancel keeps source values and confirmed edits require explicit reset confirmation', async () => {
  const confirmedTicket = { ...baseTicket, category_confirmed: 1, route_confirmed: 1, operator_category: 'Категория модели', operator_route: '2 линия' }
  const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
    const url = String(input)
    if (url === '/api/incoming/batches') return new Response(JSON.stringify([batch]), { status: 200 })
    if (url === '/api/incoming/batches/batch-1') return new Response(JSON.stringify(batch), { status: 200 })
    if (url === '/api/incoming/batches/batch-1/tickets') return new Response(JSON.stringify([confirmedTicket]), { status: 200 })
    return new Response('{}', { status: 500 })
  })

  render(<Processing options={options} onOpenTicket={() => undefined} />)
  fireEvent.click((await screen.findAllByRole('button', { name: 'Изменить' }))[0])
  fireEvent.change(screen.getByLabelText('Описание'), { target: { value: 'Временная правка' } })
  fireEvent.click(screen.getByRole('button', { name: 'Отмена' }))
  expect(screen.getAllByText(baseTicket.description).length).toBeGreaterThan(0)

  fireEvent.click(screen.getAllByRole('button', { name: 'Изменить' })[0])
  fireEvent.change(screen.getByLabelText('Описание'), { target: { value: 'Новая исходная информация' } })
  fireEvent.click(screen.getByRole('button', { name: 'Сохранить и пересчитать' }))
  expect(screen.getByText('Изменение исходных данных потребует повторного анализа. Подтверждённые категория и линия будут сброшены.')).toBeInTheDocument()
  expect(screen.getByRole('button', { name: 'Изменить и пересчитать' })).toBeInTheDocument()
  expect(fetchMock).not.toHaveBeenCalledWith('/api/incoming/tickets/11', expect.objectContaining({ method: 'PATCH' }))
})

test('category editor shows all categories, supports search and offers a custom value', async () => {
  vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
    const url = String(input)
    if (url === '/api/incoming/batches') return new Response(JSON.stringify([batch]), { status: 200 })
    if (url === '/api/incoming/batches/batch-1') return new Response(JSON.stringify(batch), { status: 200 })
    if (url === '/api/incoming/batches/batch-1/tickets') return new Response(JSON.stringify([baseTicket]), { status: 200 })
    return new Response('{}', { status: 500 })
  })

  const { container } = render(<Processing options={options} onOpenTicket={() => undefined} />)
  fireEvent.click((await screen.findAllByRole('button', { name: 'Изменить' }))[1])
  const combo = screen.getByRole('combobox', { name: 'Категория оператора' })
  fireEvent.focus(combo)
  expect(screen.getByRole('option', { name: 'Категория модели' })).toBeInTheDocument()
  expect(screen.getByRole('option', { name: 'Исправленная категория' })).toBeInTheDocument()
  expect(screen.getByRole('option', { name: 'Другое…' })).toBeInTheDocument()
  fireEvent.change(combo, { target: { value: 'Исправ' } })
  expect(screen.getByRole('option', { name: 'Исправленная категория' })).toBeInTheDocument()
  fireEvent.keyDown(combo, { key: 'Enter' })
  expect(combo).toHaveValue('Исправленная категория')
  fireEvent.focus(combo)
  fireEvent.click(screen.getByRole('option', { name: 'Другое…' }))
  expect(combo).toHaveAttribute('placeholder', 'Введите свою категорию')
  fireEvent.change(combo, { target: { value: 'Новая категория оператора' } })
  expect(combo).toHaveValue('Новая категория оператора')
  expect(screen.getByRole('button', { name: 'Подтвердить' })).toBeEnabled()
  expect(screen.getByRole('button', { name: 'Выбрать из списка' })).toBeInTheDocument()
  expect(container.querySelector('datalist')).toBeNull()
})

test('sends a manually entered category to the server', async () => {
  const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
    const url = String(input)
    if (url === '/api/incoming/batches') return new Response(JSON.stringify([batch]), { status: 200 })
    if (url === '/api/incoming/batches/batch-1') return new Response(JSON.stringify(batch), { status: 200 })
    if (url === '/api/incoming/batches/batch-1/tickets') return new Response(JSON.stringify([baseTicket]), { status: 200 })
    if (url === '/api/incoming/tickets/11/confirm-category' && init?.method === 'POST') {
      return new Response(JSON.stringify({ ...baseTicket, category_confirmed: 1, operator_category: 'Новая категория оператора' }), { status: 200 })
    }
    return new Response('{}', { status: 500 })
  })

  render(<Processing options={options} onOpenTicket={() => undefined} />)
  fireEvent.click((await screen.findAllByRole('button', { name: 'Изменить' }))[1])
  const combo = screen.getByRole('combobox', { name: 'Категория оператора' })
  fireEvent.focus(combo)
  fireEvent.click(screen.getByRole('option', { name: 'Другое…' }))
  fireEvent.change(combo, { target: { value: 'Новая категория оператора' } })
  fireEvent.click(screen.getByRole('button', { name: 'Подтвердить' }))

  await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/api/incoming/tickets/11/confirm-category', expect.objectContaining({
    method: 'POST',
    body: JSON.stringify({ category: 'Новая категория оператора' }),
  })))
})
