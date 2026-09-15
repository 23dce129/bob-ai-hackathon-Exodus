export function Spinner({ size = 'md' }: { size?: 'sm' | 'md' | 'lg' }) {
  const sz = size === 'sm' ? 'h-4 w-4' : size === 'lg' ? 'h-10 w-10' : 'h-6 w-6'
  return (
    <span
      className={`inline-block ${sz} animate-spin rounded-full border-2 border-gray-200 border-t-pharma-600`}
      role="status"
      aria-label="Loading"
    />
  )
}

export function LoadingOverlay({ text = 'Loading…' }: { text?: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 gap-3 text-gray-500">
      <Spinner size="lg" />
      <span className="text-sm">{text}</span>
    </div>
  )
}
