import { fireEvent, render, screen } from '@testing-library/react'
import { vi } from 'vitest'
import { AppShell } from './AppShell'

beforeEach(() => {
  vi.restoreAllMocks()
  localStorage.clear()
})

test('shows only operator navigation without system controls', () => {
  const onNavigate = vi.fn()
  render(<AppShell page="processing" onNavigate={onNavigate}><div>content</div></AppShell>)

  expect(screen.getByRole('button', { name: 'Обращения' })).toBeInTheDocument()
  expect(screen.getByRole('button', { name: 'База обращений' })).toBeInTheDocument()
  expect(screen.getByRole('button', { name: 'Аналитика' })).toBeInTheDocument()
  expect(screen.queryByRole('button', { name: 'Процесс' })).not.toBeInTheDocument()
  expect(screen.queryByText('Решение подтверждает оператор')).not.toBeInTheDocument()
  expect(screen.queryByRole('button', { name: 'Система' })).not.toBeInTheDocument()
  expect(screen.queryByText('Система готова')).not.toBeInTheDocument()
  expect(screen.queryByRole('button', { name: 'О системе' })).not.toBeInTheDocument()
})

test('lets the operator collapse the desktop sidebar and remembers it', async () => {
  const onNavigate = vi.fn()
  const { container } = render(<AppShell page="processing" onNavigate={onNavigate}><div>content</div></AppShell>)

  fireEvent.click(screen.getByRole('button', { name: 'Свернуть боковое меню' }))
  expect(container.querySelector('.app-shell')).toHaveClass('sidebar-collapsed')
  expect(localStorage.getItem('posttech-sidebar-view')).toBe('collapsed')

  fireEvent.click(screen.getByRole('button', { name: 'Развернуть боковое меню' }))
  expect(container.querySelector('.app-shell')).not.toHaveClass('sidebar-collapsed')
  expect(localStorage.getItem('posttech-sidebar-view')).toBe('expanded')
})

test('lets the operator resize the sidebar and persists its width', async () => {
  const onNavigate = vi.fn()
  const { container } = render(<AppShell page="processing" onNavigate={onNavigate}><div>content</div></AppShell>)

  const resizer = screen.getByRole('separator', { name: 'Изменить ширину бокового меню' })
  fireEvent.keyDown(resizer, { key: 'ArrowRight' })
  expect(localStorage.getItem('posttech-sidebar-width')).toBe('232')
  expect(resizer).toHaveAttribute('aria-valuenow', '232')

  fireEvent.keyDown(resizer, { key: 'Home' })
  expect(container.querySelector('.app-shell')).toHaveClass('sidebar-collapsed')
  expect(localStorage.getItem('posttech-sidebar-width')).toBe('70')
})
