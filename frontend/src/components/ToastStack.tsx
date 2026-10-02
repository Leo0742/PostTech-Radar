import { AlertTriangle, CheckCircle2, X } from 'lucide-react'
import { useEffect } from 'react'

export type ToastMessage = { id: number; text: string; tone: 'success' | 'error' }

export function ToastStack({ messages, onDismiss }: { messages: ToastMessage[]; onDismiss: (id: number) => void }) {
  return <div className="toast-stack" aria-live="polite">{messages.map((message) => <Toast key={message.id} message={message} onDismiss={onDismiss} />)}</div>
}

function Toast({ message, onDismiss }: { message: ToastMessage; onDismiss: (id: number) => void }) {
  useEffect(() => {
    const timeout = window.setTimeout(() => onDismiss(message.id), 3600)
    return () => window.clearTimeout(timeout)
  }, [message.id, onDismiss])
  return <div className={`toast ${message.tone}`} role="status">{message.tone === 'success' ? <CheckCircle2 size={18} /> : <AlertTriangle size={18} />}<span>{message.text}</span><button aria-label="Закрыть уведомление" onClick={() => onDismiss(message.id)}><X size={15} /></button></div>
}
