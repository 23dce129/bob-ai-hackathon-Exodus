import { useEffect, useState } from 'react'
import { checkSampleReadiness } from '../api/client'
import type { ReadinessResponse, ModuleScore, SectionResult } from '../api/types'
import { Card, CardBody, CardHeader } from '../components/Card'
import { LoadingOverlay } from '../components/Spinner'
import { StatusBadge, Badge } from '../components/Badge'
import { ProgressBar } from '../components/ProgressBar'

function ModuleCard({ mid, m }: { mid: string; m: ModuleScore }) {
  const [open, setOpen] = useState(false)
  const pct = Math.round(m.score * 100)

  return (
    <Card>
      <button
        className="w-full text-left"
        onClick={() => setOpen(o => !o)}
      >
        <CardHeader className="flex items-center gap-3">
          <span className="text-xs font-mono font-bold text-pharma-700 w-8 shrink-0">{mid}</span>
          <span className="text-sm font-medium text-gray-800 flex-1 text-left">{m.module_title}</span>
          <div className="flex items-center gap-3 shrink-0">
            {m.critical_missing_count > 0 && (
              <Badge severity="critical" label={`${m.critical_missing_count} critical`} size="sm" />
            )}
            <span className={`text-sm font-bold ${
              pct >= 75 ? 'text-green-600' : pct >= 50 ? 'text-yellow-600' : pct >= 25 ? 'text-orange-600' : 'text-red-600'
            }`}>{pct}%</span>
            <span className="text-gray-400 text-xs">{open ? '▲' : '▼'}</span>
          </div>
        </CardHeader>
        <CardBody className="py-3">
          <ProgressBar value={m.score} showPct={false} />
          <div className="flex gap-4 mt-2 text-xs text-gray-500">
            <span className="text-green-600">✓ {m.present_count} present</span>
            <span className="text-yellow-600">~ {m.needs_review_count} needs review</span>
            <span className="text-red-500">✗ {m.missing_count} missing</span>
            <span className="text-gray-400">— {m.not_applicable_count} n/a</span>
          </div>
        </CardBody>
      </button>

      {open && m.section_results.length > 0 && (
        <div className="border-t border-gray-100">
          <table className="w-full text-xs">
            <thead className="bg-gray-50">
              <tr>
                {['Section', 'Title', 'Status', 'Required', 'Safety-Relevant'].map(h => (
                  <th key={h} className="text-left font-medium text-gray-500 uppercase tracking-wide px-4 py-2">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-50">
              {m.section_results.map((s: SectionResult, i: number) => (
                <tr
                  key={i}
                  className={s.status === 'MISSING' && s.safety_relevant ? 'bg-red-50' : 'hover:bg-gray-50'}
                >
                  <td className="px-4 py-1.5 font-mono text-gray-600">{s.ctd_section_id}</td>
                  <td className="px-4 py-1.5 text-gray-700 max-w-xs truncate">{s.ctd_section_title}</td>
                  <td className="px-4 py-1.5"><StatusBadge status={s.status} /></td>
                  <td className="px-4 py-1.5 text-center">{s.required ? '✓' : '—'}</td>
                  <td className="px-4 py-1.5 text-center">
                    {s.safety_relevant
                      ? <span className="text-red-600 font-medium">⚠ Yes</span>
                      : <span className="text-gray-400">—</span>
                    }
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  )
}

export function SubmissionReadiness() {
  const [data, setData] = useState<ReadinessResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    checkSampleReadiness()
      .then(setData)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <LoadingOverlay text="Loading CTD readiness…" />
  if (error)   return <p className="text-red-600 text-sm p-4">Backend error: {error}</p>
  if (!data)   return null

  const pct = Math.round(data.overall_score_pct)
  const scoreColor =
    pct >= 75 ? 'text-green-600' :
    pct >= 50 ? 'text-yellow-600' :
    pct >= 25 ? 'text-orange-600' : 'text-red-600'

  return (
    <div className="space-y-5 max-w-5xl">
      <div>
        <h1 className="text-xl font-semibold text-gray-900">Submission Readiness</h1>
        <p className="text-sm text-gray-500 mt-0.5">
          ICH M4 CTD completeness check on the synthetic sample dossier fixture.
        </p>
      </div>

      {/* Overall score */}
      <Card>
        <CardBody className="flex flex-col md:flex-row md:items-center gap-6">
          <div className="flex items-baseline gap-3">
            <span className={`text-6xl font-bold ${scoreColor}`}>{pct}%</span>
            <div>
              <p className="text-sm font-medium text-gray-700">Overall Readiness</p>
              <p className="text-xs text-gray-400">ICH M4 CTD weighted completeness</p>
            </div>
          </div>
          <div className="flex-1 max-w-sm">
            <ProgressBar value={data.overall_score} showPct={false} />
          </div>
          <div className="grid grid-cols-2 gap-x-8 gap-y-1 text-xs text-gray-600">
            <span className="text-green-600">✓ {data.total_present} present</span>
            <span className="text-yellow-600">~ {data.total_needs_review} needs review</span>
            <span className="text-red-500">✗ {data.total_missing} missing</span>
            <span className="text-red-700 font-medium">⚠ {data.total_critical_missing} critical</span>
          </div>
        </CardBody>
      </Card>

      {/* Critical gaps callout */}
      {data.total_critical_missing > 0 && (
        <div className="bg-red-50 border border-red-200 rounded-lg px-4 py-3 flex items-start gap-3">
          <span className="text-red-500 text-xl leading-none mt-0.5">⚠</span>
          <div>
            <p className="text-sm font-semibold text-red-700">
              {data.total_critical_missing} Critical Gap{data.total_critical_missing > 1 ? 's' : ''} Detected
            </p>
            <p className="text-xs text-red-600 mt-0.5">
              Safety-relevant required sections are missing. These must be addressed before submission.
              Review each module below for details.
            </p>
          </div>
        </div>
      )}

      {/* Module breakdown */}
      <div>
        <h2 className="text-sm font-semibold text-gray-700 mb-3">
          Module Breakdown
          <span className="ml-2 text-xs font-normal text-gray-400">Click a module to expand section list</span>
        </h2>
        <div className="space-y-3">
          {Object.entries(data.module_scores).map(([mid, m]) => (
            <ModuleCard key={mid} mid={mid} m={m} />
          ))}
        </div>
      </div>

      <p className="text-xs text-gray-400">{data.disclaimer}</p>
    </div>
  )
}
