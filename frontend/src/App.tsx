import { useEffect, useState } from 'react'
import { AppShell } from './components/AppShell'
import { api } from './lib/api'
import { Analytics } from './pages/Analytics'
import { HistoryPage } from './pages/HistoryPage'
import { Processing } from './pages/Processing'
import { Process } from './pages/Process'
import { SystemPage } from './pages/SystemPage'
import type { DatasetOptions, NavPage } from './types'

const emptyOptions: DatasetOptions = { services: [], components: [], request_types: [], criticalities: [], urgencies: [], priorities: [], service_classes: [], timezones: [], categories: [], support_lines: [] }

function pageFromPath(): NavPage {
  return window.location.pathname === '/system' ? 'system' : 'processing'
}

export default function App() {
  const [page, setPage] = useState<NavPage>(pageFromPath)
  const [options, setOptions] = useState<DatasetOptions>(emptyOptions)
  const [selectedTicket, setSelectedTicket] = useState<string | null>(null)
  useEffect(() => { api<DatasetOptions>('/api/dataset/options').then(setOptions).catch(() => undefined) }, [])
  useEffect(() => {
    const handlePopState = () => setPage(pageFromPath())
    window.addEventListener('popstate', handlePopState)
    return () => window.removeEventListener('popstate', handlePopState)
  }, [])
  const navigate = (next: NavPage) => {
    setPage(next)
    if (next !== 'history') setSelectedTicket(null)
    const nextPath = next === 'system' ? '/system' : '/'
    if (window.location.pathname !== nextPath) window.history.pushState({}, '', nextPath)
    window.scrollTo({ top: 0 })
  }
  const openTicket = (id: string) => { setSelectedTicket(id); setPage('history'); window.scrollTo({ top: 0 }) }
  return <AppShell page={page} onNavigate={navigate}>
    {page === 'processing' && <Processing options={options} onOpenTicket={openTicket} />}
    {page === 'history' && <HistoryPage options={options} selectedId={selectedTicket} onSelected={setSelectedTicket} />}
    {page === 'analytics' && <Analytics options={options} />}
    {page === 'process' && <Process />}
    {page === 'system' && <SystemPage />}
  </AppShell>
}
