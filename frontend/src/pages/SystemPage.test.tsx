import { render, screen } from '@testing-library/react'
import { vi } from 'vitest'
import { SystemPage } from './SystemPage'

beforeEach(() => {
  vi.restoreAllMocks()
})

test('shows active runtime provenance returned by backend instead of stale evaluation model name', async () => {
  vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
    const url = String(input)
    if (url === '/api/system/status') {
      return new Response(JSON.stringify({
        data: { ticket_count: 1931, schema_version: '4' },
        models: { active_runtime: { category: { model_family: 'qwen-live-runtime', trained_at: '2026-09-18T00:00:00Z' } } },
      }), { status: 200 })
    }
    if (url === '/api/model/metrics') {
      return new Response(JSON.stringify({
        category: { accuracy: 0.7, macro_f1: 0.6, test_size: 100, model: 'tuned_word_char_lr' },
        routing: { accuracy: 0.8, macro_f1: 0.7, test_size: 100 },
        retrieval: { same_category_at_1: 0.7, same_category_at_3: 0.8, same_category_at_5: 0.9, queries: 10, method: 'tfidf' },
        limitations: {},
      }), { status: 200 })
    }
    return new Response('{}', { status: 404 })
  })

  render(<SystemPage />)
  expect(await screen.findByText('qwen-live-runtime')).toBeInTheDocument()
  expect(screen.queryByText('tuned_word_char_lr')).not.toBeInTheDocument()
})
