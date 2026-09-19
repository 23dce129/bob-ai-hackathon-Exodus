import { useEffect, useState, useRef } from 'react'
import { checkSampleReadiness, checkDossierFile } from '../api/client'
import type { ReadinessResponse, ModuleScore, SectionResult } from '../api/types'
import { Card, CardBody, CardHeader } from '../components/Card'
import { LoadingOverlay, Spinner } from '../components/Spinner'
import { StatusBadge, Badge } from '../components/Badge'
import { ProgressBar } from '../components/ProgressBar'

function ModuleCard({ mid, m }: { mid: string; m: ModuleScore }) {
  const [open, setOpen] = useState(false)
  const [filter, setFilter] = useState<'all' | 'critical' | 'missing' | 'needs_review'>('all')
  const pct = Math.round(m.score * 100)

  const sections: SectionResult[] = (m.section_details ?? m.section_results ?? [])

  const filteredSections = sections.filter(s => {
    if (filter === 'critical') return s.is_critical || (s.safety_relevant && s.status === 'MISSING')
    if (filter === 'missing') return s.status === 'MISSING'
    if (filter === 'needs_review') return s.status === 'NEEDS_REVIEW'
    return true
  })

  return (
    <Card className="overflow-hidden">
      <button
        type="button"
        className="w-full text-left transition-colors hover:bg-gray-50/50"
        onClick={() => setOpen(o => !o)}
      >
        <CardHeader className="flex items-center gap-3">
          <span className="text-xs font-mono font-bold bg-pharma-100 text-pharma-800 px-2 py-0.5 rounded shrink-0">
            {mid}
          </span>
          <span className="text-sm font-semibold text-gray-800 flex-1 text-left">
            {m.module_title}
          </span>
          <div className="flex items-center gap-3 shrink-0">
            {m.critical_missing_count > 0 && (
              <Badge severity="critical" label={`${m.critical_missing_count} critical`} size="sm" />
            )}
            <span className={`text-sm font-bold ${
              pct >= 75 ? 'text-green-600' : pct >= 50 ? 'text-yellow-600' : pct >= 25 ? 'text-orange-600' : 'text-red-600'
            }`}>
              {pct}%
            </span>
            <span className="text-gray-400 text-xs w-4 text-center">{open ? '▲' : '▼'}</span>
          </div>
        </CardHeader>
        <CardBody className="py-3">
          <ProgressBar value={m.score} showPct={false} />
          <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2 text-xs text-gray-500">
            <span className="text-green-600 font-medium">✓ {m.present_count} present</span>
            <span className="text-yellow-600 font-medium">~ {m.needs_review_count} needs review</span>
            <span className="text-red-500 font-medium">✗ {m.missing_count} missing</span>
            <span className="text-gray-400">— {m.not_applicable_count} n/a</span>
          </div>
        </CardBody>
      </button>

      {open && (
        <div className="border-t border-gray-100 bg-gray-50/30">
          {/* Filter sub-bar */}
          <div className="px-4 py-2 bg-gray-50 border-b border-gray-100 flex items-center justify-between text-xs">
            <span className="text-gray-500 font-medium">
              Showing {filteredSections.length} of {sections.length} sections
            </span>
            <div className="flex gap-1">
              {(['all', 'critical', 'missing', 'needs_review'] as const).map(f => (
                <button
                  key={f}
                  onClick={(e) => { e.stopPropagation(); setFilter(f) }}
                  className={`px-2 py-0.5 rounded text-xs transition-colors ${
                    filter === f
                      ? 'bg-pharma-700 text-white font-medium shadow-xs'
                      : 'bg-white border border-gray-200 text-gray-600 hover:bg-gray-100'
                  }`}
                >
                  {f === 'all' ? 'All' : f === 'critical' ? 'Critical Gaps' : f === 'missing' ? 'Missing' : 'Needs Review'}
                </button>
              ))}
            </div>
          </div>

          {filteredSections.length === 0 ? (
            <div className="p-4 text-center text-xs text-gray-400">
              No sections match the selected filter.
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead className="bg-gray-100/70 text-gray-600 border-b border-gray-100">
                  <tr>
                    <th className="text-left font-semibold uppercase tracking-wider px-4 py-2 w-24">Section</th>
                    <th className="text-left font-semibold uppercase tracking-wider px-4 py-2">Title</th>
                    <th className="text-left font-semibold uppercase tracking-wider px-4 py-2 w-32">Status</th>
                    <th className="text-center font-semibold uppercase tracking-wider px-4 py-2 w-28">Requirement</th>
                    <th className="text-center font-semibold uppercase tracking-wider px-4 py-2 w-28">Safety Relevant</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100 bg-white">
                  {filteredSections.map((s: SectionResult, i: number) => {
                    const secId = s.section_id ?? s.ctd_section_id ?? '—'
                    const secTitle = s.section_title ?? s.ctd_section_title ?? '—'
                    const isReq = s.required ?? (s.requirement === 'required')
                    const reqLabel = s.requirement
                      ? (s.requirement.charAt(0).toUpperCase() + s.requirement.slice(1))
                      : (isReq ? 'Required' : 'Conditional')
                    const isCrit = s.is_critical || (s.safety_relevant && s.status === 'MISSING')

                    return (
                      <tr
                        key={secId || i}
                        className={isCrit ? 'bg-red-50/70 hover:bg-red-50' : 'hover:bg-gray-50/80'}
                      >
                        <td className="px-4 py-2 font-mono font-semibold text-gray-700">
                          {secId}
                        </td>
                        <td className="px-4 py-2 text-gray-800">
                          <div className="font-medium text-gray-900">{secTitle}</div>
                          {s.match_reason && (
                            <p className="text-[11px] text-gray-400 mt-0.5 line-clamp-1" title={s.match_reason}>
                              {s.match_reason}
                            </p>
                          )}
                        </td>
                        <td className="px-4 py-2">
                          <StatusBadge status={s.status} />
                        </td>
                        <td className="px-4 py-2 text-center">
                          <span className={`px-2 py-0.5 rounded text-[11px] ${
                            isReq ? 'bg-gray-100 text-gray-700 font-medium' : 'bg-gray-50 text-gray-400'
                          }`}>
                            {reqLabel}
                          </span>
                        </td>
                        <td className="px-4 py-2 text-center">
                          {s.safety_relevant ? (
                            <span className="inline-flex items-center gap-1 text-red-700 font-semibold bg-red-100/80 px-2 py-0.5 rounded text-[11px]">
                              <span>⚠</span> Safety
                            </span>
                          ) : (
                            <span className="text-gray-300">—</span>
                          )}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </Card>
  )
}

export function SubmissionReadiness() {
  const [data, setData] = useState<ReadinessResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [activeSource, setActiveSource] = useState<'sample' | 'custom'>('sample')
  const [fileName, setFileName] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const loadSample = () => {
    setLoading(true)
    setError(null)
    setActiveSource('sample')
    setFileName(null)
    checkSampleReadiness()
      .then(setData)
      .catch(e => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    loadSample()
  }, [])

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    setUploading(true)
    setError(null)
    try {
      const res = await checkDossierFile(file)
      setData(res)
      setActiveSource('custom')
      setFileName(file.name)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  if (loading) return <LoadingOverlay text="Loading CTD readiness…" />

  const pct = Math.round(data?.overall_score_pct ?? 0)
  const scoreColor =
    pct >= 75 ? 'text-green-600' :
    pct >= 50 ? 'text-yellow-600' :
    pct >= 25 ? 'text-orange-600' : 'text-red-600'

  return (
    <div className="space-y-6 max-w-5xl">
      {/* Header with Source Switching */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-gray-900">Submission Readiness</h1>
          <p className="text-sm text-gray-500 mt-0.5">
            ICH M4 CTD completeness check on {activeSource === 'sample' ? 'synthetic Synvastatin dossier fixture' : `custom outline (${fileName})`}.
          </p>
        </div>

        <div className="flex items-center gap-2">
          {activeSource === 'custom' ? (
            <button
              onClick={loadSample}
              className="px-3 py-1.5 text-xs font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors shadow-xs"
            >
              Reset to Sample
            </button>
          ) : null}

          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,.xlsx"
            className="hidden"
            onChange={handleFileUpload}
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
            className="px-4 py-1.5 text-xs font-semibold text-white bg-pharma-700 hover:bg-pharma-800 rounded-lg transition-colors flex items-center gap-2 shadow-xs disabled:opacity-50"
          >
            {uploading ? <Spinner size="sm" /> : <span>📁</span>}
            <span>{uploading ? 'Analyzing File…' : 'Upload Dossier (.csv/.xlsx)'}</span>
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-sm text-red-700 flex items-start justify-between gap-3">
          <div>
            <strong>Error:</strong> {error}
          </div>
          <button
            onClick={() => setError(null)}
            className="text-red-500 hover:text-red-700 font-bold"
          >
            ✕
          </button>
        </div>
      )}

      {data && (
        <>
          {/* Overall score card */}
          <Card>
            <CardBody className="flex flex-col md:flex-row md:items-center gap-6">
              <div className="flex items-baseline gap-3">
                <span className={`text-6xl font-bold tracking-tight ${scoreColor}`}>{pct}%</span>
                <div>
                  <p className="text-sm font-semibold text-gray-800">Overall Readiness</p>
                  <p className="text-xs text-gray-400">ICH M4 CTD weighted completeness</p>
                </div>
              </div>
              <div className="flex-1 max-w-sm">
                <ProgressBar value={data.overall_score} showPct={false} />
              </div>
              <div className="grid grid-cols-2 gap-x-8 gap-y-1.5 text-xs text-gray-600">
                <span className="text-green-600 font-medium">✓ {data.total_present} present</span>
                <span className="text-yellow-600 font-medium">~ {data.total_needs_review} needs review</span>
                <span className="text-red-500 font-medium">✗ {data.total_missing} missing</span>
                <span className="text-red-700 font-bold">⚠ {data.total_critical_missing} critical</span>
              </div>
            </CardBody>
          </Card>

          {/* Critical gaps callout */}
          {data.total_critical_missing > 0 && (
            <div className="bg-red-50 border border-red-200 rounded-xl px-5 py-4 flex items-start gap-3.5 shadow-xs">
              <span className="text-red-600 text-2xl leading-none mt-0.5">⚠</span>
              <div>
                <p className="text-sm font-bold text-red-800">
                  {data.total_critical_missing} Critical Gap{data.total_critical_missing > 1 ? 's' : ''} Detected
                </p>
                <p className="text-xs text-red-700 mt-1 leading-relaxed">
                  Required safety-relevant CTD sections are missing from this submission. These represent critical deficiencies that must be addressed prior to regulatory filing.
                </p>
              </div>
            </div>
          )}

          {/* Module breakdown */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-bold text-gray-800">
                Module Breakdown (Modules 1–5)
              </h2>
              <span className="text-xs text-gray-400">
                Click any module to inspect section matches & audit details
              </span>
            </div>

            <div className="space-y-3">
              {Object.entries(data.module_scores).map(([mid, m]) => (
                <ModuleCard key={mid} mid={mid} m={m} />
              ))}
            </div>
          </div>

          <div className="bg-gray-100/60 rounded-lg p-3 text-[11px] text-gray-500 leading-relaxed">
            <span className="font-semibold text-gray-600">Disclaimer: </span>
            {data.disclaimer}
          </div>
        </>
      )}
    </div>
  )
}
