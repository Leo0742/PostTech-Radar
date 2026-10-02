import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { vi } from 'vitest'

import { NewTicket } from './NewTicket'

const response = {
  category: {
    label: 'Проблема с QR-код(подключение/отключение)',
    confidence: 0.68,
    accepted: true,
    review_required: false,
    threshold: 0.35,
    alternatives: [],
    signals: ['qr код', 'получение'],
    explanation: 'Положительные признаки модели.',
  },
  routing: {
    label: '(1 линия)',
    confidence: 0.79,
    alternatives: [],
    signals: ['qr код'],
    explanation: 'Маршрут по регистрационным полям.',
  },
  sla_risk: { risk: 0.03, sample_size: 6, level: 'Низкий', explanation: 'Исторический индикатор.' },
  similar: [],
}

test('submits the primary flow and renders model output', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => response }))
  render(<NewTicket options={{ services: [], components: [], request_types: [], criticalities: [], urgencies: [], priorities: [], service_classes: [], timezones: [], categories: [], support_lines: [] }} onOpenTicket={() => undefined} />)
  fireEvent.change(screen.getByLabelText('Описание обращения'), {
    target: { value: 'Не подключается QR-код для получения отправления' },
  })
  fireEvent.click(screen.getByRole('button', { name: 'Анализировать' }))
  expect(screen.getByRole('button', { name: 'Анализируем…' })).toBeDisabled()
  expect(screen.getByText('Сопоставляем с историей')).toBeInTheDocument()
  await waitFor(() => expect(screen.getByRole('heading', { name: 'Проблема с QR-код(подключение/отключение)' })).toBeInTheDocument())
  expect(screen.getByText('(1 линия)')).toBeInTheDocument()
  expect(screen.getByText('Уверенность модели: 68%')).toBeInTheDocument()
})

test('shows a useful API error state', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, json: async () => ({ detail: 'Сервис недоступен' }) }))
  render(<NewTicket options={{ services: [], components: [], request_types: [], criticalities: [], urgencies: [], priorities: [], service_classes: [], timezones: [], categories: [], support_lines: [] }} onOpenTicket={() => undefined} />)
  fireEvent.change(screen.getByLabelText('Описание обращения'), {
    target: { value: 'Достаточно длинное описание обращения' },
  })
  fireEvent.click(screen.getByRole('button', { name: 'Анализировать' }))
  await waitFor(() => expect(screen.getByText('Сервис недоступен')).toBeInTheDocument())
})

test('shows an honest empty state when retrieval rejects weak matches', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({
      ...response,
      retrieval: {
        rejected: true,
        reason: 'Достаточно похожих исторических обращений не найдено.',
        threshold: 0.2,
        best_relevance: 0.04,
      },
      similar: [],
    }),
  }))
  render(<NewTicket options={{ services: [], components: [], request_types: [], criticalities: [], urgencies: [], priorities: [], service_classes: [], timezones: [], categories: [], support_lines: [] }} onOpenTicket={() => undefined} />)
  fireEvent.change(screen.getByLabelText('Описание обращения'), {
    target: { value: 'Квантовый телепорт сломал океан' },
  })
  fireEvent.click(screen.getByRole('button', { name: 'Анализировать' }))

  await waitFor(() => expect(screen.getByText('Достаточно похожих исторических обращений не найдено.')).toBeInTheDocument())
})

test('lets the operator explicitly run an additional category check', async () => {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce({ ok: true, json: async () => response })
    .mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        category: {
          label: 'Отслеживание отправлений',
          confidence: 0.81,
          alternatives: [
            { label: 'Отслеживание отправлений', confidence: 0.81 },
            { label: 'Личный кабинет', confidence: 0.12 },
          ],
        },
        model_provenance: { model_id: 'Qwen/Qwen3-Embedding-4B', profile: 'qwen_recheck' },
        latency_seconds: 41.2,
      }),
    })
  vi.stubGlobal('fetch', fetchMock)
  render(<NewTicket options={{ services: [], components: [], request_types: [], criticalities: [], urgencies: [], priorities: [], service_classes: [], timezones: [], categories: [], support_lines: [] }} onOpenTicket={() => undefined} />)
  fireEvent.change(screen.getByLabelText('Описание обращения'), {
    target: { value: 'Не отображается отправление в личном кабинете' },
  })
  fireEvent.click(screen.getByRole('button', { name: 'Анализировать' }))
  await waitFor(() => expect(screen.getByRole('button', { name: /Дополнительная проверка/ })).toBeInTheDocument())
  expect(screen.queryByText(/Qwen/i)).not.toBeInTheDocument()

  fireEvent.click(screen.getByRole('button', { name: /Дополнительная проверка/ }))

  await waitFor(() => expect(screen.getByText('Результат дополнительной проверки')).toBeInTheDocument())
  expect(screen.getByRole('heading', { name: 'Отслеживание отправлений' })).toBeInTheDocument()
  expect(screen.getByText('Уверенность модели: 68%')).toBeInTheDocument()
  expect(fetchMock).toHaveBeenLastCalledWith(
    '/api/tickets/recheck-qwen',
    expect.objectContaining({ method: 'POST' }),
  )
})
