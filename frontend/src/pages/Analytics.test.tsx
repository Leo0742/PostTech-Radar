import { render, screen } from '@testing-library/react'
import { vi } from 'vitest'
import type { AnalyticsData, DatasetOptions } from '../types'
import { Analytics } from './Analytics'

const options: DatasetOptions = { services: [], components: [], request_types: [], criticalities: [], urgencies: [], priorities: [], service_classes: [], timezones: [], categories: [], support_lines: [] }

const analytics: AnalyticsData = {
  kpis: { tickets: 123, overdue: 7, overdue_share: 0.057, median_duration_seconds: 3600, median_duration: '1 ч', p90_duration_seconds: 7200, p90_duration: '2 ч', multi_line: 20, high_clarifications: 4, clarification_threshold: 3 },
  breakdowns: {
    categories: [
      { name: 'Отслеживание', count: 40, overdue: 3, sample_n: 40, raw_overdue_rate: .075, overdue_share: .075, smoothed_risk: .06, median_duration_seconds: 3600, p90_duration_seconds: 7200, avg_duration_seconds: 4000, avg_clarifications: 1.2 },
      { name: 'Личный кабинет', count: 25, overdue: 2, sample_n: 25, raw_overdue_rate: .08, overdue_share: .08, smoothed_risk: .06, median_duration_seconds: 5400, p90_duration_seconds: 9000, avg_duration_seconds: 6000, avg_clarifications: .8 },
    ],
    services: [],
    priorities: [],
    support_lines: [
      { name: '(1 линия)', count: 90, overdue: 4, sample_n: 90, raw_overdue_rate: .04, overdue_share: .04, smoothed_risk: .04, median_duration_seconds: 3000, p90_duration_seconds: 6000, avg_duration_seconds: 3500, avg_clarifications: .5 },
    ],
  },
  time: [
    { month: '2025-01', count: 10, overdue: 1, overdue_share: .1, smoothed_risk: .07 },
    { month: '2025-03', count: 20, overdue: 1, overdue_share: .05, smoothed_risk: .05 },
  ],
  methodology: {},
}

beforeEach(() => vi.restoreAllMocks())

test('shows a simplified decision-focused analytics dashboard', async () => {
  vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify(analytics), { status: 200 }))

  render(<Analytics options={options} />)

  expect(await screen.findByRole('heading', { name: 'Аналитика' })).toBeInTheDocument()
  expect(screen.getByText('123')).toBeInTheDocument()
  expect(screen.getByText(/94,3\s*%/)).toBeInTheDocument()
  expect(screen.getByText('Медиана обработки')).toBeInTheDocument()
  expect(screen.getByText('Несколько линий')).toBeInTheDocument()
  expect(screen.getByRole('heading', { name: 'Динамика обращений' })).toBeInTheDocument()
  expect(screen.getByRole('heading', { name: 'Основные темы' })).toBeInTheDocument()
  expect(screen.getByRole('heading', { name: 'Нагрузка по линиям' })).toBeInTheDocument()
  expect(screen.getByRole('heading', { name: 'Требуют внимания' })).toBeInTheDocument()
  expect(screen.getByText('Сложные обращения')).toBeInTheDocument()
  expect(screen.getByText('Чаще требуют уточнений')).toBeInTheDocument()
  expect(screen.getByText('Риск SLA по категориям')).toBeInTheDocument()
  expect(screen.getByText('Риск SLA по линиям')).toBeInTheDocument()
  expect(screen.getByText('Сроки обработки')).toBeInTheDocument()
  expect(screen.queryByText('Обычно решается')).not.toBeInTheDocument()
  expect(screen.queryByText('Дольше всего решаются')).not.toBeInTheDocument()
  expect(screen.queryByText(/N\s*=/)).not.toBeInTheDocument()
  expect(screen.queryByRole('heading', { name: 'Маршрутизация между линиями' })).not.toBeInTheDocument()
})

test('renders the main analytics without loading an unfiltered process summary', async () => {
  const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify(analytics), { status: 200 }))
  render(<Analytics options={options} />)
  expect(await screen.findByText('123')).toBeInTheDocument()
  expect(fetchMock).toHaveBeenCalledTimes(1)
  expect(String(fetchMock.mock.calls[0][0])).toMatch(/^\/api\/analytics\/summary\?/)
})
