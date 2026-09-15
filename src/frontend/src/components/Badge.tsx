type Severity = 'critical' | 'high' | 'moderate' | 'low' | 'info'

interface Props {
  severity?: Severity
  label: string
  size?: 'sm' | 'md'
}

const styles: Record<Severity, string> = {
  critical: 'bg-red-100 text-red-800 border border-red-200',
  high:     'bg-orange-100 text-orange-800 border border-orange-200',
  moderate: 'bg-yellow-100 text-yellow-800 border border-yellow-200',
  low:      'bg-green-100 text-green-700 border border-green-200',
  info:     'bg-blue-100 text-blue-700 border border-blue-200',
}

export function Badge({ severity = 'info', label, size = 'md' }: Props) {
  const sz = size === 'sm' ? 'text-xs px-1.5 py-0.5' : 'text-xs px-2 py-1'
  return (
    <span className={`inline-flex items-center rounded font-medium ${sz} ${styles[severity]}`}>
      {label}
    </span>
  )
}

export function PrrBadge({ prr }: { prr?: number }) {
  if (!prr) return <span className="text-gray-400 text-xs">—</span>
  const severity: Severity =
    prr >= 10 ? 'critical' :
    prr >= 5  ? 'high' :
    prr >= 2  ? 'moderate' : 'low'
  return <Badge severity={severity} label={`PRR ${prr.toFixed(2)}`} />
}

export function StatusBadge({ status }: { status: string }) {
  const map: Record<string, Severity> = {
    PRESENT:        'low',
    NEEDS_REVIEW:   'moderate',
    MISSING:        'critical',
    NOT_APPLICABLE: 'info',
  }
  return <Badge severity={map[status] ?? 'info'} label={status.replace('_', ' ')} size="sm" />
}

export function TrendBadge({ direction }: { direction?: string }) {
  if (!direction) return null
  const map: Record<string, { label: string; cls: string }> = {
    increasing:        { label: '↑ Increasing', cls: 'text-red-600 bg-red-50 border border-red-200' },
    decreasing:        { label: '↓ Decreasing', cls: 'text-green-600 bg-green-50 border border-green-200' },
    stable:            { label: '→ Stable',     cls: 'text-gray-600 bg-gray-50 border border-gray-200' },
    insufficient_data: { label: '? Insufficient', cls: 'text-gray-400 bg-gray-50 border border-gray-200' },
  }
  const { label, cls } = map[direction] ?? map['stable']
  return (
    <span className={`inline-flex items-center rounded text-xs px-2 py-0.5 font-medium ${cls}`}>
      {label}
    </span>
  )
}
