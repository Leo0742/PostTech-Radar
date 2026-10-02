import { fireEvent, render, screen } from '@testing-library/react'
import { vi } from 'vitest'
import App from './App'

beforeEach(() => {
  vi.restoreAllMocks()
  vi.spyOn(window, 'scrollTo').mockImplementation(() => undefined)
  localStorage.clear()
  window.history.pushState({}, '', '/system')
  vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
    const url = String(input)
    if (url === '/api/dataset/options') return new Response(JSON.stringify({ services: [], components: [], request_types: [], criticalities: [], urgencies: [], priorities: [], service_classes: [], timezones: [], categories: [], support_lines: [] }), { status: 200 })
    if (url === '/api/incoming/batches') return new Response(JSON.stringify([]), { status: 200 })
    return new Response('{}', { status: 404 })
  })
})

afterEach(() => {
  window.history.pushState({}, '', '/')
})

test('legacy /system route returns to the operator workspace', async () => {
  render(<App />)
  expect(await screen.findByRole('heading', { name: 'Добавьте обращения' })).toBeInTheDocument()
  expect(window.location.pathname).toBe('/')
})

test('keeps the processing workspace mounted when navigating away and back', async () => {
  window.history.pushState({}, '', '/')
  const fetchMock = vi.mocked(globalThis.fetch)
  fetchMock.mockImplementation(async (input) => {
    const url = String(input)
    if (url === '/api/dataset/options') return new Response(JSON.stringify({ services: [], components: [], request_types: [], criticalities: [], urgencies: [], priorities: [], service_classes: [], timezones: [], categories: [], support_lines: [] }), { status: 200 })
    if (url === '/api/incoming/batches') return new Response(JSON.stringify([]), { status: 200 })
    return new Response('{}', { status: 404 })
  })

  render(<App />)
  expect(await screen.findByRole('heading', { name: 'Добавьте обращения' })).toBeInTheDocument()
  expect(fetchMock.mock.calls.filter(([input]) => String(input) === '/api/incoming/batches')).toHaveLength(1)

  fireEvent.click(screen.getByRole('button', { name: 'Аналитика' }))
  expect(await screen.findByRole('heading', { name: 'Аналитика' })).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: 'Обращения' }))
  expect(screen.getByRole('heading', { name: 'Добавьте обращения' })).toBeInTheDocument()
  expect(fetchMock.mock.calls.filter(([input]) => String(input) === '/api/incoming/batches')).toHaveLength(1)
})
