import { render, screen } from '@testing-library/react'
import { DataModels } from './DataModels'

test('renders provenance and clearly read-only model controls', async () => {
  vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({
    data: { ticket_count: 1931, dataset_hash: 'abc123', source_path: '/data/Обращения.xlsx', imported_at: '2026-09-16T10:00:00', schema_version: '2', last_import: { mode: 'upsert', inserted: 0, updated: 2, unchanged: 1929, total: 1931 } },
    models: { category: { trained_at: '2026-09-16T11:00:00', training_record_count: 1511, dataset_sha256: 'data1', package_versions: { sklearn: '1.7' } }, routing: null, retrieval: null },
  }), { status: 200 }))
  render(<DataModels />)
  expect(await screen.findByText('1 931')).toBeInTheDocument()
  expect(screen.getByText(/upsert/)).toBeInTheDocument()
  expect(screen.getAllByText(/только для просмотра/i)).toHaveLength(2)
  expect(screen.queryByRole('button', { name: /переобучить/i })).not.toBeInTheDocument()
})
