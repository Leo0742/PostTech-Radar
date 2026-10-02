import type { LucideIcon } from 'lucide-react'

export function MetricCard({ label, value, detail, icon: Icon, tone = 'blue' }: { label: string; value: string; detail?: string; icon: LucideIcon; tone?: 'blue' | 'green' | 'amber' | 'navy' }) {
  return <section className={`metric-card tone-${tone}`}><div className="metric-icon"><Icon size={21} /></div><div><span>{label}</span><strong>{value}</strong>{detail && <small>{detail}</small>}</div></section>
}

