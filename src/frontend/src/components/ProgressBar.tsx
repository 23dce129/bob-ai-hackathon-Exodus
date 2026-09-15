interface Props {
  value: number   // 0–1
  label?: string
  colorClass?: string
  showPct?: boolean
}

export function ProgressBar({ value, label, colorClass = 'bg-pharma-600', showPct = true }: Props) {
  const pct = Math.round(value * 100)
  const color =
    pct >= 75 ? 'bg-green-500' :
    pct >= 50 ? 'bg-yellow-400' :
    pct >= 25 ? 'bg-orange-400' : 'bg-red-500'

  const barColor = colorClass !== 'bg-pharma-600' ? colorClass : color

  return (
    <div className="w-full">
      {(label || showPct) && (
        <div className="flex justify-between text-xs text-gray-600 mb-1">
          {label && <span>{label}</span>}
          {showPct && <span className="font-medium">{pct}%</span>}
        </div>
      )}
      <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${barColor}`}
          style={{ width: `${Math.min(100, pct)}%` }}
        />
      </div>
    </div>
  )
}
