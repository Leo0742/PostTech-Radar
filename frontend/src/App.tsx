import { useEffect, useState } from 'react'
import { AppShell } from './components/AppShell'
import { api } from './lib/api'
import { Analytics } from './pages/Analytics'
import { HistoryPage } from './pages/HistoryPage'
import { Processing } from './pages/Processing'
import type { DatasetOptions, NavPage } from './types'

const emptyOptions: DatasetOptions = { services: [], components: [], request_types: [], criticalities: [], urgencies: [], priorities: [], service_classes: [], timezones: [], categories: [], support_lines: [] }

function pageFromPath(): NavPage {
  return 'processing'
}

export default function App() {
  const [page, setPage] = useState<NavPage>(pageFromPath)
  const [processingMounted, setProcessingMounted] = useState(() => pageFromPath() === 'processing')
  const [options, setOptions] = useState<DatasetOptions>(emptyOptions)
  const [selectedTicket, setSelectedTicket] = useState<string | null>(null)
  useEffect(() => { api<DatasetOptions>('/api/dataset/options').then(setOptions).catch(() => undefined) }, [])
  useEffect(() => {
    if (window.location.pathname === '/system') window.history.replaceState({}, '', '/')
  }, [])
  useEffect(() => {
    const handlePopState = () => {
      const next = pageFromPath()
      if (next === 'processing') setProcessingMounted(true)
      setPage(next)
    }
    window.addEventListener('popstate', handlePopState)
    return () => window.removeEventListener('popstate', handlePopState)
  }, [])
  const navigate = (next: NavPage) => {
    if (next === 'processing') setProcessingMounted(true)
    setPage(next)
    if (next !== 'history') setSelectedTicket(null)
    if (window.location.pathname !== '/') window.history.pushState({}, '', '/')
    window.scrollTo({ top: 0 })
  }
  const openTicket = (id: string) => { setSelectedTicket(id); setPage('history'); window.scrollTo({ top: 0 }) }
  return <AppShell page={page} onNavigate={navigate}>
    {processingMounted && <div className="persistent-page" hidden={page !== 'processing'}><Processing options={options} onOpenTicket={openTicket} /></div>}
    {page === 'history' && <HistoryPage options={options} selectedId={selectedTicket} onSelected={setSelectedTicket} />}
    {page === 'analytics' && <Analytics options={options} />}
  </AppShell>
}
