import { useState, useCallback } from 'react'
import { getGapReport } from '../api/client'
import type { GapReportResponse, GapSection } from '../api/types'
import { Card, CardBody } from '../components/Card'
import { Spinner } from '../components/Spinner'

const SUB_TYPES = ['NDA', 'BLA', 'ANDA', 'MAA']

const subLabels: Record<string, string> = {
  NDA:  'New Drug Application (US FDA)',
  BLA:  'Biologics License Application (US FDA)',
  ANDA: 'Abbreviated New Drug Application (US FDA)',
  MAA:  'Marketing Authorisation Application (EMA)',
}

function SectionRow({ s, i }: { s: GapSection; i: number }) {
  return (
    <tr className={`${i % 2 === 0 ? 'bg-white' : 'bg-gray-50'} hover:bg-pharma-50 text-xs`}>
      <td className="px-4 py-2 font-mono text-gray-600 whitespace-nowrap">{s.id}</td>
      <td className="px-4 py-2 text-gray-700">{s.title}</td>
      <td className="px-4 py-2 text-center">
        {s.required
          ? <span className="text-green-600 font-medium">Required</span>
          : <span className="text-gray-400">Conditional</span>
        }
      </td>
      <td className="px-4 py-2 text-center">
        {s.safety_relevant
          ? <span className="text-red-600 font-medium">⚠ Yes</span>
          : <span className="text-gray-400">—</span>
        }
      </td>
      <td className="px-4 py-2">
        <div className="flex flex-wrap gap-1">
          {s.traceability_categories.map(c => (
            <span key={c} className="bg-pharma-100 text-pharma-700 px-1.5 py-0.5 rounded text-xs">
              {c}
            </span>
          ))}
        </div>
      </td>
    </tr>
  )
}

export function GapReport() {
  const [subType, setSubType] = useState('NDA')
  const [data, setData] = useState<GapReportResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [filter, setFilter] = useState<'all' | 'required' | 'safety'>('all')

  const run = useCallback(async () => {
    setLoading(true); setError(null)
    try {
      setData(await getGapReport(subType))
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [subType])

  const allSections: GapSection[] = [
    ...(data?.required_sections ?? []),
    ...(data?.conditional_sections ?? []),
  ]

  const visible = allSections.filter(s => {
    if (filter === 'required') return s.required
    if (filter === 'safety')   return s.safety_relevant
    return true
  })

  const safetyRequired = allSections.filter(s => s.required && s.safety_relevant)

  return (
    <div className="space-y-5 max-w-5xl">
      <div>
        <h1 className="text-xl font-semibold text-gray-900">Gap Report</h1>
        <p className="text-sm text-gray-500 mt-0.5">
          Required CTD sections per submission type, based on ICH M4 guidelines.
        </p>
      </div>

      {/* Controls */}
      <Card>
        <CardBody className="flex flex-wrap gap-3 items-end">
          <div className="flex flex-col gap-1">
            <label className="text-xs font-medium text-gray-600">Submission Type</label>
            <select
              value={subType}
              onChange={e => setSubType(e.target.value)}
              className="border border-gray-300 rounded px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-pharma-400"
            >
              {SUB_TYPES.map(t => (
                <option key={t} value={t}>{t} — {subLabels[t]}</option>
              ))}
            </select>
          </div>
          <button
            onClick={run}
            disabled={loading}
            className="flex items-center gap-2 bg-pharma-700 hover:bg-pharma-800 disabled:opacity-50 text-white text-sm font-medium px-4 py-1.5 rounded transition-colors"
          >
            {loading ? <Spinner size="sm" /> : null}
            Generate Report
          </button>
        </CardBody>
      </Card>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded p-3 text-sm text-red-700">{error}</div>
      )}

      {data && (
        <>
          {/* Summary cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Card><CardBody>
              <p className="text-xs text-gray-500 mb-1">Required</p>
              <p className="text-2xl font-bold text-gray-900">{data.required_count}</p>
            </CardBody></Card>
            <Card><CardBody>
              <p className="text-xs text-gray-500 mb-1">Conditional</p>
              <p className="text-2xl font-bold text-gray-900">{data.conditional_count}</p>
            </CardBody></Card>
            <Card><CardBody>
              <p className="text-xs text-gray-500 mb-1">Safety-Relevant Required</p>
              <p className="text-2xl font-bold text-red-600">{data.safety_relevant_required_count}</p>
            </CardBody></Card>
            <Card><CardBody>
              <p className="text-xs text-gray-500 mb-1">Submission Type</p>
              <p className="text-xl font-bold text-pharma-700">{data.submission_type}</p>
            </CardBody></Card>
          </div>

          {/* Safety-required callout */}
          {safetyRequired.length > 0 && (
            <div className="bg-red-50 border border-red-200 rounded-lg px-4 py-3">
              <p className="text-sm font-semibold text-red-700 mb-2">
                ⚠ Safety-Relevant Required Sections ({safetyRequired.length})
              </p>
              <div className="flex flex-wrap gap-2">
                {safetyRequired.slice(0, 12).map(s => (
                  <span key={s.id} className="bg-white border border-red-200 rounded px-2 py-0.5 text-xs text-red-700 font-mono">
                    {s.id} — {s.title}
                  </span>
                ))}
                {safetyRequired.length > 12 && (
                  <span className="text-xs text-red-500">+{safetyRequired.length - 12} more</span>
                )}
              </div>
            </div>
          )}

          {/* Filter tabs */}
          <div className="flex gap-2">
            {(['all', 'required', 'safety'] as const).map(f => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`px-3 py-1 text-xs rounded font-medium transition-colors ${
                  filter === f
                    ? 'bg-pharma-700 text-white'
                    : 'bg-white border border-gray-200 text-gray-600 hover:bg-gray-50'
                }`}
              >
                {f === 'all' ? `All (${allSections.length})` :
                 f === 'required' ? `Required only (${data.required_count})` :
                 `Safety-relevant (${data.safety_relevant_required_count})`}
              </button>
            ))}
          </div>

          {/* Full table */}
          <Card>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gray-50 border-b border-gray-100">
                  <tr>
                    {['Section ID', 'Title', 'Requirement', 'Safety-Relevant', 'Signal Categories'].map(h => (
                      <th key={h} className="text-left text-xs font-medium text-gray-500 uppercase tracking-wide px-4 py-2">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {visible.map((s, i) => <SectionRow key={s.id} s={s} i={i} />)}
                </tbody>
              </table>
            </div>
          </Card>

          <p className="text-xs text-gray-400">{data.disclaimer}</p>
        </>
      )}

      {!data && !loading && (
        <div className="text-center py-12 text-gray-400 text-sm">
          Select a submission type and click <strong>Generate Report</strong>.
        </div>
      )}
    </div>
  )
}
