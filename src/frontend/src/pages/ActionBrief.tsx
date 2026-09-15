import { useState, useCallback } from 'react'
import { analyzeSignals, traceSignals } from '../api/client'
import type { TraceResponse, TraceResult } from '../api/types'
import { Card, CardBody, CardHeader } from '../components/Card'
import { Spinner } from '../components/Spinner'
import { Badge, StatusBadge } from '../components/Badge'
import { ProgressBar } from '../components/ProgressBar'

function PriorityIcon({ priority }: { priority: string }) {
  if (priority === 'urgent') return <span className="text-red-600 font-bold text-sm">🔴</span>
  if (priority === 'high')   return <span className="text-orange-500 font-bold text-sm">🟠</span>
  return <span className="text-yellow-500 font-bold text-sm">🟡</span>
}

function TraceResultCard({ r }: { r: TraceResult }) {
  const [open, setOpen] = useState(false)
  const pct = Math.round(r.traceability_score * 100)

  return (
    <Card className={r.critical_gaps.length > 0 ? 'border-red-200' : ''}>
      <button className="w-full text-left" onClick={() => setOpen(o => !o)}>
        <CardHeader className="flex items-center gap-3">
          <span className="text-sm font-medium text-gray-800 flex-1">{r.adverse_event}</span>
          <Badge severity="info" label={r.signal_category} size="sm" />
          <span className={`text-sm font-bold ${
            pct >= 75 ? 'text-green-600' : pct >= 50 ? 'text-yellow-600' : 'text-red-600'
          }`}>{pct}% traced</span>
          {r.critical_gaps.length > 0 && (
            <Badge severity="critical" label={`${r.critical_gaps.length} gaps`} size="sm" />
          )}
          <span className="text-gray-400 text-xs">{open ? '▲' : '▼'}</span>
        </CardHeader>
        <CardBody className="py-2">
          <ProgressBar value={r.traceability_score} showPct={false} />
        </CardBody>
      </button>

      {open && (
        <div className="border-t border-gray-100">
          <table className="w-full text-xs">
            <thead className="bg-gray-50">
              <tr>
                {['Section', 'Title', 'Status', 'Safety', 'Rationale'].map(h => (
                  <th key={h} className="text-left font-medium text-gray-500 uppercase tracking-wide px-4 py-2">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-50">
              {r.entries.map((e, i) => (
                <tr key={i} className={e.status === 'MISSING' && e.safety_relevant ? 'bg-red-50' : 'hover:bg-gray-50'}>
                  <td className="px-4 py-1.5 font-mono text-gray-600">{e.ctd_section_id}</td>
                  <td className="px-4 py-1.5 text-gray-700 max-w-xs">{e.ctd_section_title}</td>
                  <td className="px-4 py-1.5"><StatusBadge status={e.status} /></td>
                  <td className="px-4 py-1.5 text-center">
                    {e.safety_relevant
                      ? <span className="text-red-600">⚠</span>
                      : <span className="text-gray-300">—</span>
                    }
                  </td>
                  <td className="px-4 py-1.5 text-gray-500 max-w-sm">{e.relevance_reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  )
}

export function ActionBrief() {
  const [drug, setDrug] = useState('')
  const [dossierInput, setDossierInput] = useState('')
  const [traceResult, setTraceResult] = useState<TraceResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [priorityActions, setPriorityActions] = useState<
    Array<{ priority: string; adverse_event: string; section_id: string; section_title: string; action: string; rationale: string }>
  >([])

  const run = useCallback(async () => {
    if (!drug.trim()) { setError('Drug name is required.'); return }
    setLoading(true); setError(null); setTraceResult(null); setPriorityActions([])

    try {
      // 1. Detect signals for this drug
      const signalData = await analyzeSignals({ drug_name: drug.trim(), include_trends: false })
      const flagged = signalData.flagged_signals

      if (flagged.length === 0) {
        setError(`No signals flagged for '${drug}' with default thresholds (PRR ≥ 2, n ≥ 3, p < 0.05).`)
        setLoading(false)
        return
      }

      // 2. Parse dossier sections from textarea (one section ID per line)
      const sectionNumbers = dossierInput
        .split(/[\n,]+/)
        .map(s => s.trim())
        .filter(s => s.length > 0)

      // 3. Trace signals to CTD sections
      const signals = flagged.map(s => ({
        drug_name: s.drug_name,
        adverse_event: s.adverse_event,
        signal_flag: true,
      }))

      const trace = await traceSignals(signals, sectionNumbers)
      setTraceResult(trace)

      // 4. Build priority actions from critical gaps
      const actions: typeof priorityActions = []
      for (const r of trace.results) {
        for (const gapId of r.critical_gaps) {
          const entry = r.entries.find(e => e.ctd_section_id === gapId)
          actions.push({
            priority: 'urgent',
            adverse_event: r.adverse_event,
            section_id: gapId,
            section_title: entry?.ctd_section_title ?? '',
            action: `Add required safety documentation for '${r.adverse_event}' to CTD Section ${gapId}.`,
            rationale: entry?.relevance_reason ?? '',
          })
        }
        if (r.traceability_score < 0.5) {
          actions.push({
            priority: 'high',
            adverse_event: r.adverse_event,
            section_id: '',
            section_title: '',
            action: `Review documentation coverage for '${r.adverse_event}' (current score: ${Math.round(r.traceability_score * 100)}%).`,
            rationale: 'Signal has low overall CTD section coverage.',
          })
        }
      }
      setPriorityActions(actions)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [drug, dossierInput])

  const overallPct = traceResult ? Math.round(traceResult.overall_score * 100) : null

  return (
    <div className="space-y-5 max-w-5xl">
      <div>
        <h1 className="text-xl font-semibold text-gray-900">Regulatory Action Brief</h1>
        <p className="text-sm text-gray-500 mt-0.5">
          Signal detection + traceability analysis combined into prioritised documentation actions.
        </p>
      </div>

      {/* Input form */}
      <Card>
        <CardBody className="space-y-4">
          <div className="flex flex-wrap gap-4">
            <div className="flex flex-col gap-1 flex-1 min-w-48">
              <label className="text-xs font-medium text-gray-600">Drug Name <span className="text-red-500">*</span></label>
              <input
                type="text"
                placeholder="e.g. warfarin, metformin"
                value={drug}
                onChange={e => setDrug(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && run()}
                className="border border-gray-300 rounded px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-pharma-400"
              />
            </div>
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs font-medium text-gray-600">
              Existing Dossier Sections
              <span className="ml-1 text-gray-400 font-normal">(optional — one section ID per line, e.g. 2.4, 2.7.4, 5.3.5)</span>
            </label>
            <textarea
              rows={3}
              placeholder={"2.4\n2.7.4\n5.3.5.1"}
              value={dossierInput}
              onChange={e => setDossierInput(e.target.value)}
              className="border border-gray-300 rounded px-3 py-1.5 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-pharma-400 resize-none"
            />
            <p className="text-xs text-gray-400">
              Leave blank for worst-case gap analysis (assumes empty dossier).
            </p>
          </div>
          <button
            onClick={run}
            disabled={loading}
            className="flex items-center gap-2 bg-pharma-700 hover:bg-pharma-800 disabled:opacity-50 text-white text-sm font-medium px-5 py-2 rounded transition-colors"
          >
            {loading ? <Spinner size="sm" /> : null}
            {loading ? 'Analyzing…' : 'Generate Action Brief'}
          </button>
        </CardBody>
      </Card>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded p-3 text-sm text-red-700">{error}</div>
      )}

      {traceResult && (
        <>
          {/* Overall summary */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Card><CardBody>
              <p className="text-xs text-gray-500 mb-1">Signals Traced</p>
              <p className="text-2xl font-bold text-gray-900">{traceResult.results.length}</p>
            </CardBody></Card>
            <Card><CardBody>
              <p className="text-xs text-gray-500 mb-1">Traceability Score</p>
              <p className={`text-2xl font-bold ${
                overallPct! >= 75 ? 'text-green-600' : overallPct! >= 50 ? 'text-yellow-600' : 'text-red-600'
              }`}>{overallPct}%</p>
            </CardBody></Card>
            <Card><CardBody>
              <p className="text-xs text-gray-500 mb-1">Critical Gaps</p>
              <p className={`text-2xl font-bold ${traceResult.critical_gap_count > 0 ? 'text-red-600' : 'text-green-600'}`}>
                {traceResult.critical_gap_count}
              </p>
            </CardBody></Card>
            <Card><CardBody>
              <p className="text-xs text-gray-500 mb-1">Fully Documented</p>
              <p className={`text-2xl font-bold ${traceResult.fully_documented ? 'text-green-600' : 'text-red-600'}`}>
                {traceResult.fully_documented ? 'Yes' : 'No'}
              </p>
            </CardBody></Card>
          </div>

          {/* Priority actions */}
          {priorityActions.length > 0 && (
            <Card>
              <CardHeader>
                <h2 className="text-sm font-semibold text-gray-700">
                  Priority Actions
                  <span className="ml-2 text-xs font-normal text-gray-400">
                    Ranked by urgency
                  </span>
                </h2>
              </CardHeader>
              <div className="divide-y divide-gray-50">
                {priorityActions.map((a, i) => (
                  <div key={i} className={`flex gap-3 px-5 py-3 ${a.priority === 'urgent' ? 'bg-red-50' : 'bg-orange-50'}`}>
                    <PriorityIcon priority={a.priority} />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap mb-1">
                        <Badge
                          severity={a.priority === 'urgent' ? 'critical' : 'high'}
                          label={a.priority.toUpperCase()}
                          size="sm"
                        />
                        <span className="text-xs text-gray-600 font-medium">{a.adverse_event}</span>
                        {a.section_id && (
                          <span className="text-xs font-mono text-pharma-700 bg-pharma-50 px-1.5 py-0.5 rounded">
                            §{a.section_id}
                          </span>
                        )}
                      </div>
                      <p className="text-sm text-gray-800">{a.action}</p>
                      {a.rationale && (
                        <p className="text-xs text-gray-500 mt-0.5">{a.rationale}</p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {priorityActions.length === 0 && traceResult.fully_documented && (
            <div className="bg-green-50 border border-green-200 rounded p-4 text-sm text-green-700">
              ✓ All traced signals are documented in the provided dossier sections. No urgent actions identified.
            </div>
          )}

          {/* Per-signal traceability */}
          <div>
            <h2 className="text-sm font-semibold text-gray-700 mb-3">
              Signal Traceability Detail
              <span className="ml-2 text-xs font-normal text-gray-400">Click to expand CTD section mapping</span>
            </h2>
            <div className="space-y-3">
              {traceResult.results.map((r, i) => (
                <TraceResultCard key={i} r={r} />
              ))}
            </div>
          </div>

          <p className="text-xs text-gray-400">{traceResult.disclaimer}</p>
        </>
      )}

      {!traceResult && !loading && (
        <div className="text-center py-12 text-gray-400 text-sm">
          Enter a drug name and click <strong>Generate Action Brief</strong>.
        </div>
      )}
    </div>
  )
}
