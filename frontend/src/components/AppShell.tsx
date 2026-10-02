import { BarChart3, Database, Inbox, PanelLeftClose, PanelLeftOpen, Radar } from 'lucide-react'
import { useRef, useState, type CSSProperties, type KeyboardEvent, type PointerEvent, type ReactNode } from 'react'
import type { NavPage } from '../types'

const primaryItems: { id: NavPage; label: string; icon: typeof Inbox }[] = [
  { id: 'processing', label: 'Обращения', icon: Inbox },
  { id: 'history', label: 'База обращений', icon: Database },
  { id: 'analytics', label: 'Аналитика', icon: BarChart3 },
]
const pageTitle: Partial<Record<NavPage, string>> = {
  processing: 'Обращения', history: 'База обращений', analytics: 'Аналитика',
}

const SIDEBAR_MIN_WIDTH = 70
const SIDEBAR_DEFAULT_WIDTH = 216
const SIDEBAR_MAX_WIDTH = 320
const SIDEBAR_COMPACT_THRESHOLD = 150

function clampSidebarWidth(value: number) {
  return Math.min(SIDEBAR_MAX_WIDTH, Math.max(SIDEBAR_MIN_WIDTH, value))
}

function initialSidebarWidth() {
  const stored = Number(localStorage.getItem('posttech-sidebar-width'))
  if (Number.isFinite(stored) && stored >= SIDEBAR_MIN_WIDTH && stored <= SIDEBAR_MAX_WIDTH) return stored
  return localStorage.getItem('posttech-sidebar-view') === 'collapsed' ? SIDEBAR_MIN_WIDTH : SIDEBAR_DEFAULT_WIDTH
}

export function AppShell({ page, onNavigate, children }: { page: NavPage; onNavigate: (page: NavPage) => void; children: ReactNode }) {
  const [sidebarWidth, setSidebarWidth] = useState(initialSidebarWidth)
  const [sidebarResizing, setSidebarResizing] = useState(false)
  const resizingRef = useRef(false)
  const sidebarCollapsed = sidebarWidth < SIDEBAR_COMPACT_THRESHOLD

  const updateSidebarWidth = (value: number) => {
    const next = clampSidebarWidth(Math.round(value))
    setSidebarWidth(next)
    localStorage.setItem('posttech-sidebar-width', String(next))
    localStorage.setItem('posttech-sidebar-view', next < SIDEBAR_COMPACT_THRESHOLD ? 'collapsed' : 'expanded')
    if (next >= SIDEBAR_COMPACT_THRESHOLD) localStorage.setItem('posttech-sidebar-expanded-width', String(next))
  }

  const toggleSidebar = () => {
    if (!sidebarCollapsed) {
      localStorage.setItem('posttech-sidebar-expanded-width', String(sidebarWidth))
      updateSidebarWidth(SIDEBAR_MIN_WIDTH)
      return
    }
    const remembered = Number(localStorage.getItem('posttech-sidebar-expanded-width'))
    updateSidebarWidth(Number.isFinite(remembered) && remembered >= SIDEBAR_COMPACT_THRESHOLD ? remembered : SIDEBAR_DEFAULT_WIDTH)
  }

  const beginSidebarResize = (event: PointerEvent<HTMLDivElement>) => {
    resizingRef.current = true
    setSidebarResizing(true)
    event.currentTarget.setPointerCapture?.(event.pointerId)
  }

  const resizeSidebar = (event: PointerEvent<HTMLDivElement>) => {
    if (!resizingRef.current) return
    updateSidebarWidth(event.clientX)
  }

  const endSidebarResize = (event: PointerEvent<HTMLDivElement>) => {
    resizingRef.current = false
    setSidebarResizing(false)
    if (event.currentTarget.hasPointerCapture?.(event.pointerId)) event.currentTarget.releasePointerCapture?.(event.pointerId)
  }

  const resizeSidebarWithKeyboard = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'ArrowLeft') updateSidebarWidth(sidebarWidth - 16)
    else if (event.key === 'ArrowRight') updateSidebarWidth(sidebarWidth + 16)
    else if (event.key === 'Home') updateSidebarWidth(SIDEBAR_MIN_WIDTH)
    else if (event.key === 'End') updateSidebarWidth(SIDEBAR_MAX_WIDTH)
    else return
    event.preventDefault()
  }

  const navButton = ({ id, label, icon: Icon }: { id: NavPage; label: string; icon: typeof Inbox }) => (
    <button key={id} className={page === id ? 'nav-item active' : 'nav-item'} onClick={() => onNavigate(id)} aria-current={page === id ? 'page' : undefined} aria-label={sidebarCollapsed ? label : undefined} title={sidebarCollapsed ? label : undefined}>
      <Icon size={19} aria-hidden="true" /><span>{label}</span>
    </button>
  )

  const shellStyle = { '--sidebar-width': `${sidebarWidth}px` } as CSSProperties

  return <div className={`app-shell${sidebarCollapsed ? ' sidebar-collapsed' : ''}${sidebarResizing ? ' sidebar-resizing' : ''}`} style={shellStyle}>
    <a className="skip-link" href="#main-content">Перейти к содержимому</a>
    <aside className="sidebar">
      <div className="brand"><span className="brand-mark" aria-hidden="true"><Radar size={21} /></span><div className="brand-copy"><strong>ПочтаТех Радар</strong></div></div>
      <nav aria-label="Основная навигация">
        <div className="nav-group">{primaryItems.map(navButton)}</div>
      </nav>
      <button className="sidebar-toggle" onClick={toggleSidebar} aria-label={sidebarCollapsed ? 'Развернуть боковое меню' : 'Свернуть боковое меню'} title={sidebarCollapsed ? 'Развернуть меню' : 'Свернуть меню'}>
        {sidebarCollapsed ? <PanelLeftOpen size={18} aria-hidden="true" /> : <PanelLeftClose size={18} aria-hidden="true" />}
        <span>{sidebarCollapsed ? 'Развернуть' : 'Свернуть меню'}</span>
      </button>
      <div
        className="sidebar-resizer"
        role="separator"
        aria-label="Изменить ширину бокового меню"
        aria-orientation="vertical"
        aria-valuemin={SIDEBAR_MIN_WIDTH}
        aria-valuemax={SIDEBAR_MAX_WIDTH}
        aria-valuenow={sidebarWidth}
        tabIndex={0}
        onPointerDown={beginSidebarResize}
        onPointerMove={resizeSidebar}
        onPointerUp={endSidebarResize}
        onPointerCancel={endSidebarResize}
        onKeyDown={resizeSidebarWithKeyboard}
      />
    </aside>
    <div className="main-column">
      <header className="topbar"><strong className="current-section">{pageTitle[page] ?? 'Service Desk'}</strong></header>
      <main id="main-content">{children}</main>
    </div>
  </div>
}
