import { useState, useCallback, useEffect, useRef } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  Cell, ReferenceLine,
} from 'recharts'
import { analyzeSignals, getPriorityScores, getSignalExplanation } from '../api/client'
import type {
  AnalyzeResponse, PriorityScore, Signal, SignalExplanation,
} from '../api/types'
import { Card, CardBody, CardHeader } from '../components/Card'
import { Spinner } from '../components/Spinner'
import { PrrBadge, TrendBadge, Badge } from '../components/Badge'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const DRUGS_IN_DATASET = [
  '', 'warfarin', 'atorvastatin', 'metformin', 'amoxicillin', 'ibuprofen', 'lisinopril',
]

function fmt(n: number | null | undefined, dp = 2): string {
  if (n == null) return '—'
  return n.toFixed(dp)
}
function fmtP(p: number | null | undefined): string {
  if (p == null) return '—'
  return p < 0.001 ? '<0.001' : p.toFixed(4)
}
function fmtPct(r: number | null | undefined): string {
  if (r == null) return '—'
  return `${(r * 100).toFixed(0)}%`
}

function priorityColor(score: number): string {
  if (score >= 70) return 'bg-red-100 text-red-800 border-red-200'
  if (score >= 45) return 'bg-orange-100 text-orange-800 border-orange-200'
  if (score >= 25) return 'bg-yellow-100 text-yellow-800 border-yellow-200'
  return 'bg-green-100 text-green-700 border-green-200'
}

function PriorityBadge({ score, max }: { score: number; max: number }) {
  return (
    <span className={`inline-flex items-center gap-1 rounded border text-xs font-bold px-2 py-0.5 ${priorityColor(score)}`}>
      {score}<span className="font-normal opacity-60">/{max}</span>
    </span>
  )
}

// ---------------------------------------------------------------------------
// Evidence drawer — slides in from the right
// ---------------------------------------------------------------------------

function ParagraphText({ text }: { text: string }) {
  // Render **bold** markers
  const parts = text.split(/\*\*(.*?)\*\*/g)
  return (
    <p className="text-sm text-gray-700 leading-relaxed">
      {parts.map((p, i) =>
        i % 2 === 1 ? <strong key={i} className="font-semibold text-gray-900">{p}</strong> : p
      )}
    </p>
  )
}

function ComponentBar({ name, earned, maximum }: { name: string; earned: number; maximum: number }) {
  const pct = Math.round((earned / maximum) * 100)
  const color =
    pct >= 75 ? 'bg-green-500' : pct >= 50 ? 'bg-yellow-400' : pct >= 25 ? 'bg-orange-400' : 'bg-red-400'
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-32 text-gray-600 truncate shrink-0">{name}</span>
      <div className="flex-1 h-1.5 bg-gray-100 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-gray-500 w-14 text-right shrink-0">{earned.toFixed(1)}/{maximum}</span>
    </div>
  )
}

interface DrawerProps {
  signal: Signal
  priorityScore: PriorityScore | null
  onClose: () => void
}

function EvidenceDrawer({ signal, priorityScore, onClose }: DrawerProps) {
  const [explanation, setExplanation] = useState<SignalExplanation | null>(null)
  const [explaining, setExplaining] = useState(false)
  const [explainError, setExplainError] = useState<string | null>(null)
  const drawerRef = useRef<HTMLDivElement>(null)

  // close on Escape
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose])

  const handleExplain = useCallback(async () => {
    setExplaining(true); setExplainError(null)
    try {
      setExplanation(await getSignalExplanation(signal.drug_name, signal.adverse_event))
    } catch (e: unknown) {
      setExplainError(e instanceof Error ? e.message : String(e))
    } finally {
      setExplaining(false)
    }
  }, [signal])

  const ps = priorityScore
  const trend = signal.trend

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/30 z-30"
        onClick={onClose}
      />
      {/* Panel */}
      <div
        ref={drawerRef}
        className="fixed right-0 top-0 h-full w-full max-w-xl bg-white shadow-2xl z-40 flex flex-col"
      >
        {/* Header */}
        <div className="flex items-start justify-between px-5 py-4 border-b border-gray-200 bg-pharma-800 text-white">
          <div>
            <p className="text-xs font-semibold text-pharma-200 uppercase tracking-wide mb-0.5">Evidence View</p>
            <h2 className="text-base font-bold leading-snug">
              {signal.drug_name} <span className="text-pharma-300">·</span> {signal.adverse_event}
            </h2>
          </div>
          <button onClick={onClose} className="text-pharma-200 hover:text-white text-xl leading-none ml-4 mt-0.5">✕</button>
        </div>

        {/* Scrollable body */}
        <div className="flex-1 overflow-y-auto p-5 space-y-5">

          {/* Stats grid */}
          <div className="grid grid-cols-3 gap-3">
            {[
              { label: 'PRR', value: fmt(signal.prr), note: signal.ci_lower_95 != null ? `95% CI [${fmt(signal.ci_lower_95)}, ${fmt(signal.ci_upper_95)}]` : undefined },
              { label: 'p-value', value: fmtP(signal.p_value), note: signal.test_method?.toUpperCase() },
              { label: 'Reports (n)', value: fmt(signal.a, 0) },
            ].map(({ label, value, note }) => (
              <div key={label} className="bg-gray-50 rounded-lg p-3 border border-gray-100">
                <p className="text-xs text-gray-500 mb-1">{label}</p>
                <p className="text-lg font-bold text-gray-900">{value}</p>
                {note && <p className="text-xs text-gray-400 mt-0.5">{note}</p>}
              </div>
            ))}
          </div>

          {/* Trend badge + score */}
          <div className="flex items-center gap-3 flex-wrap">
            <TrendBadge direction={trend?.direction} />
            {trend?.trend_score != null && (
              <span className="text-xs text-gray-500">
                Trend score: <span className={`font-mono font-semibold ${trend.trend_score > 0 ? 'text-red-600' : 'text-green-600'}`}>
                  {trend.trend_score > 0 ? '+' : ''}{trend.trend_score.toFixed(4)}
                </span>
              </span>
            )}
            {trend?.pct_change != null && (
              <span className={`text-xs font-semibold ${trend.pct_change > 0 ? 'text-red-600' : 'text-green-600'}`}>
                {trend.pct_change > 0 ? '↑' : '↓'} {Math.abs(trend.pct_change).toFixed(1)}% period-over-period
              </span>
            )}
          </div>

          {/* Priority score breakdown */}
          {ps && (
            <div className="border border-gray-100 rounded-lg p-4 space-y-2">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-xs font-semibold text-gray-700 uppercase tracking-wide">Priority Score</h3>
                <PriorityBadge score={ps.final_score} max={ps.max_possible_score} />
              </div>
              {ps.components.map(c => (
                <ComponentBar key={c.name} name={c.name} earned={c.earned} maximum={c.maximum} />
              ))}
              <p className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded p-2 mt-2 leading-relaxed">
                {ps.disclaimer}
              </p>
            </div>
          )}

          {/* Explain Signal button + output */}
          <div className="border border-pharma-100 rounded-lg overflow-hidden">
            <div className="bg-pharma-50 px-4 py-3 flex items-center justify-between">
              <div>
                <h3 className="text-xs font-semibold text-pharma-800 uppercase tracking-wide">Signal Explanation</h3>
                <p className="text-xs text-pharma-600 mt-0.5">Evidence-grounded narrative from the analysis pipeline</p>
              </div>
              <button
                onClick={handleExplain}
                disabled={explaining}
                className="flex items-center gap-1.5 bg-pharma-700 hover:bg-pharma-800 disabled:opacity-50 text-white text-xs font-semibold px-3 py-1.5 rounded transition-colors shrink-0"
              >
                {explaining ? <Spinner size="sm" /> : '🔬'}
                {explaining ? 'Analyzing…' : 'Explain Signal'}
              </button>
            </div>

            {explainError && (
              <div className="px-4 py-3 text-xs text-red-700 bg-red-50">{explainError}</div>
            )}

            {explanation && (
              <div className="px-4 py-4 space-y-3 border-t border-pharma-100">
                {explanation.explanation_paragraphs.map((para, i) => (
                  <ParagraphText key={i} text={para} />
                ))}

                {/* Chi-squared + serious outcomes inline summary */}
                <div className="grid grid-cols-2 gap-3 mt-2 pt-3 border-t border-gray-100">
                  <div className="text-xs text-gray-600">
                    <span className="font-medium text-gray-800">Chi-squared stat:</span>{' '}
                    {fmt(explanation.chi2_statistic, 3)}
                  </div>
                  <div className="text-xs text-gray-600">
                    <span className="font-medium text-gray-800">Serious outcomes:</span>{' '}
                    {explanation.serious_count}/{explanation.total_reports} ({fmtPct(explanation.serious_rate)})
                  </div>
                </div>

                {/* Component score mini-bars from explanation */}
                <div className="space-y-1.5 pt-2 border-t border-gray-100">
                  {explanation.components.map(c => (
                    <ComponentBar key={c.name} name={c.name} earned={c.earned} maximum={c.maximum} />
                  ))}
                </div>

                <p className="text-xs text-gray-400 mt-1">{explanation.disclaimer}</p>
              </div>
            )}

            {!explanation && !explaining && (
              <div className="px-4 py-3 text-xs text-gray-400">
                Click <strong>Explain Signal</strong> to generate a structured evidence narrative from the analysis pipeline.
              </div>
            )}
          </div>

        </div>
      </div>
    </>
  )
}

// ---------------------------------------------------------------------------
// Filter controls
// ---------------------------------------------------------------------------

interface Filters {
  drug: string
  minPrr: number
  minCases: number
  maxPval: number
  dateFrom: string
  dateTo: string
}

function FilterPanel({
  filters, setFilters, onAnalyze, loading,
}: {
  filters: Filters
  setFilters: (f: Filters) => void
  onAnalyze: () => void
  loading: boolean
}) {
  const set = (k: keyof Filters) => (v: string | number) =>
    setFilters({ ...filters, [k]: v })

  return (
    <Card>
      <CardHeader>
        <h2 className="text-sm font-semibold text-gray-700">Analysis Parameters</h2>
      </CardHeader>
      <CardBody className="flex flex-wrap gap-4 items-end">
        {/* Drug selector */}
        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium text-gray-600">Drug</label>
          <select
            value={filters.drug}
            onChange={e => set('drug')(e.target.value)}
            className="border border-gray-300 rounded px-3 py-1.5 text-sm w-44 focus:outline-none focus:ring-2 focus:ring-pharma-400"
          >
            <option value="">All drugs</option>
            {DRUGS_IN_DATASET.filter(Boolean).map(d => (
              <option key={d} value={d}>{d}</option>
            ))}
          </select>
        </div>

        {/* Date range */}
        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium text-gray-600">Date From</label>
          <input
            type="date"
            value={filters.dateFrom}
            onChange={e => set('dateFrom')(e.target.value)}
            className="border border-gray-300 rounded px-3 py-1.5 text-sm w-36 focus:outline-none focus:ring-2 focus:ring-pharma-400"
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium text-gray-600">Date To</label>
          <input
            type="date"
            value={filters.dateTo}
            onChange={e => set('dateTo')(e.target.value)}
            className="border border-gray-300 rounded px-3 py-1.5 text-sm w-36 focus:outline-none focus:ring-2 focus:ring-pharma-400"
          />
        </div>

        {/* Min PRR */}
        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium text-gray-600">Min PRR</label>
          <select
            value={filters.minPrr}
            onChange={e => set('minPrr')(Number(e.target.value))}
            className="border border-gray-300 rounded px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-pharma-400"
          >
            <option value={2}>≥ 2.0 (FDA/EMA)</option>
            <option value={3}>≥ 3.0</option>
            <option value={5}>≥ 5.0</option>
          </select>
        </div>

        {/* Min reports */}
        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium text-gray-600">Min Reports (n)</label>
          <select
            value={filters.minCases}
            onChange={e => set('minCases')(Number(e.target.value))}
            className="border border-gray-300 rounded px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-pharma-400"
          >
            <option value={3}>≥ 3 (FDA/EMA)</option>
            <option value={5}>≥ 5</option>
            <option value={10}>≥ 10</option>
          </select>
        </div>

        {/* Max p-val */}
        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium text-gray-600">Max p-value</label>
          <select
            value={filters.maxPval}
            onChange={e => set('maxPval')(Number(e.target.value))}
            className="border border-gray-300 rounded px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-pharma-400"
          >
            <option value={0.05}>≤ 0.05 (standard)</option>
            <option value={0.01}>≤ 0.01</option>
            <option value={0.001}>≤ 0.001</option>
          </select>
        </div>

        <button
          onClick={onAnalyze}
          disabled={loading}
          className="flex items-center gap-2 bg-pharma-700 hover:bg-pharma-800 disabled:opacity-50 text-white text-sm font-semibold px-5 py-2 rounded transition-colors"
        >
          {loading ? <Spinner size="sm" /> : null}
          {loading ? 'Analyzing…' : 'Analyze Signals'}
        </button>

        <p className="text-xs text-gray-400 self-end">
          Note: date range filters are applied as reference context only;<br />
          the sample dataset spans Jan 2022 – Dec 2023.
        </p>
      </CardBody>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Signal row
// ---------------------------------------------------------------------------

interface RowProps {
  signal: Signal
  priority: PriorityScore | undefined
  onSelect: (s: Signal) => void
}

function SignalRow({ signal: s, priority, onSelect }: RowProps) {
  const seriousComp = priority?.components.find(c => c.name === 'Serious outcomes')
  const clusterComp = priority?.components.find(c => c.name === 'Cluster strength')
  const chi2 = s.chi2_statistic ?? null

  return (
    <tr
      className="hover:bg-pharma-50 cursor-pointer border-b border-gray-50 text-sm"
      onClick={() => onSelect(s)}
    >
      {/* Drug */}
      <td className="px-3 py-2.5 font-semibold text-gray-800 whitespace-nowrap">{s.drug_name}</td>

      {/* Adverse event */}
      <td className="px-3 py-2.5 text-gray-700 whitespace-nowrap">{s.adverse_event}</td>

      {/* Priority score */}
      <td className="px-3 py-2.5">
        {priority
          ? <PriorityBadge score={priority.final_score} max={priority.max_possible_score} />
          : <span className="text-gray-300 text-xs">—</span>
        }
      </td>

      {/* PRR */}
      <td className="px-3 py-2.5"><PrrBadge prr={s.prr} /></td>

      {/* 95 % CI */}
      <td className="px-3 py-2.5 text-xs text-gray-500 whitespace-nowrap font-mono">
        {s.ci_lower_95 != null
          ? `[${fmt(s.ci_lower_95)}, ${fmt(s.ci_upper_95)}]`
          : '—'
        }
      </td>

      {/* Chi-squared */}
      <td className="px-3 py-2.5 text-xs font-mono text-gray-600">
        {chi2 != null ? chi2.toFixed(2) : '—'}
        <span className="text-gray-400 ml-1 text-xs">{s.test_method?.toUpperCase()}</span>
      </td>

      {/* p-value */}
      <td className="px-3 py-2.5 text-xs font-mono text-gray-600">{fmtP(s.p_value)}</td>

      {/* Reports n */}
      <td className="px-3 py-2.5 text-xs text-gray-700">{s.a ?? '—'}</td>

      {/* Trend */}
      <td className="px-3 py-2.5"><TrendBadge direction={s.trend?.direction} /></td>

      {/* Serious outcome rate */}
      <td className="px-3 py-2.5 text-xs text-gray-600">
        {seriousComp?.raw_value != null
          ? <span className={seriousComp.raw_value > 0.3 ? 'text-red-600 font-semibold' : ''}>
              {fmtPct(seriousComp.raw_value)}
            </span>
          : '—'
        }
      </td>

      {/* Cluster */}
      <td className="px-3 py-2.5 text-xs text-gray-600">
        {clusterComp?.raw_value != null
          ? <span className={clusterComp.raw_value > 0.5 ? 'text-pharma-700 font-semibold' : 'text-gray-400'}>
              {fmtPct(clusterComp.raw_value)}
            </span>
          : '—'
        }
      </td>

      {/* Open drawer hint */}
      <td className="px-3 py-2.5 text-right">
        <span className="text-xs text-pharma-500">Evidence →</span>
      </td>
    </tr>
  )
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export function SignalDetection() {
  const [filters, setFilters] = useState<Filters>({
    drug: '',
    minPrr: 2,
    minCases: 3,
    maxPval: 0.05,
    dateFrom: '',
    dateTo: '',
  })

  const [signals, setSignals]   = useState<AnalyzeResponse | null>(null)
  const [priorities, setPriorities] = useState<PriorityScore[]>([])
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState<string | null>(null)
  const [selected, setSelected] = useState<Signal | null>(null)

  const runAnalysis = useCallback(async () => {
    setLoading(true); setError(null)
    try {
      const [sigData, priData] = await Promise.all([
        analyzeSignals({
          drug_name: filters.drug || undefined,
          min_prr:   filters.minPrr,
          min_cases: filters.minCases,
          max_pval:  filters.maxPval,
          include_trends: true,
        }),
        getPriorityScores(
          filters.drug || undefined,
          filters.minPrr,
          filters.minCases,
          filters.maxPval,
        ),
      ])
      setSignals(sigData)
      setPriorities(priData.priority_scores)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [filters])

  const flagged = signals?.flagged_signals ?? []

  // Merge priority into signals for display
  const priorityMap = new Map(
    priorities.map(p => [`${p.drug_name}|${p.adverse_event}`, p])
  )

  // PRR chart data
  const chartData = flagged.slice(0, 10).map(s => ({
    name: `${s.drug_name}/${s.adverse_event}`,
    prr:  s.prr ?? 0,
    pri:  priorityMap.get(`${s.drug_name}|${s.adverse_event}`)?.final_score ?? 0,
  }))

  return (
    <div className="space-y-5 max-w-7xl">
      {/* Page header */}
      <div>
        <h1 className="text-xl font-semibold text-gray-900">Signal Detection</h1>
        <p className="text-sm text-gray-500 mt-0.5">
          PRR disproportionality analysis with 6-dimension priority scoring · synthetic FAERS dataset
        </p>
      </div>

      {/* Filter panel */}
      <FilterPanel
        filters={filters}
        setFilters={setFilters}
        onAnalyze={runAnalysis}
        loading={loading}
      />

      {error && (
        <div className="bg-red-50 border border-red-200 rounded p-3 text-sm text-red-700">{error}</div>
      )}

      {signals && (
        <>
          {/* Summary chips */}
          <div className="flex gap-2 flex-wrap">
            <Badge severity="info"    label={`${signals.total_evaluated} pairs evaluated`} />
            <Badge severity={signals.total_flagged > 0 ? 'high' : 'low'}
                   label={`${signals.total_flagged} flagged`} />
            {filters.drug && <Badge severity="info" label={`Drug: ${filters.drug}`} />}
            <Badge severity="info"
                   label={`Thresholds: PRR ≥ ${filters.minPrr} · n ≥ ${filters.minCases} · p ≤ ${filters.maxPval}`} />
          </div>

          {/* PRR chart */}
          {chartData.length > 0 && (
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-sm font-semibold text-gray-700">PRR Values</h2>
                    <p className="text-xs text-gray-400">Click a bar to jump to that signal</p>
                  </div>
                  <div className="flex gap-3 text-xs text-gray-500">
                    <span><span className="inline-block w-2 h-2 rounded-sm bg-red-500 mr-1" />≥ 10</span>
                    <span><span className="inline-block w-2 h-2 rounded-sm bg-orange-400 mr-1" />5–10</span>
                    <span><span className="inline-block w-2 h-2 rounded-sm bg-yellow-400 mr-1" />2–5</span>
                  </div>
                </div>
              </CardHeader>
              <CardBody className="pt-0">
                <ResponsiveContainer width="100%" height={180}>
                  <BarChart
                    data={chartData}
                    margin={{ top: 4, right: 8, left: 0, bottom: 56 }}
                    onClick={(d: Record<string, unknown>) => {
                      const payload = d?.activePayload as Array<{payload: {name: string}}> | undefined
                      if (!payload?.[0]) return
                      const name = payload[0].payload.name
                      const [dn, ae] = name.split('/')
                      const match = flagged.find(s => s.drug_name === dn && s.adverse_event === ae)
                      if (match) setSelected(match)
                    }}
                  >
                    <XAxis dataKey="name" tick={{ fontSize: 10 }} interval={0} angle={-38} textAnchor="end" />
                    <YAxis tick={{ fontSize: 11 }} />
                    <Tooltip
                      formatter={(v: unknown, name: unknown) => [
                        name === 'prr' ? `PRR: ${(v as number).toFixed(2)}` : `Priority: ${v}`,
                        '',
                      ]}
                      contentStyle={{ fontSize: 11 }}
                    />
                    <ReferenceLine y={2} stroke="#9ca3af" strokeDasharray="4 2"
                      label={{ value: 'PRR=2', fontSize: 9, fill: '#9ca3af' }} />
                    <Bar dataKey="prr" radius={[3, 3, 0, 0]} cursor="pointer">
                      {chartData.map((d, i) => (
                        <Cell key={i} fill={d.prr >= 10 ? '#ef4444' : d.prr >= 5 ? '#f97316' : '#eab308'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </CardBody>
            </Card>
          )}

          {/* Signal table */}
          {flagged.length > 0 ? (
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-sm font-semibold text-gray-700">
                      Flagged Drug-Event Pairs
                    </h2>
                    <p className="text-xs text-gray-400 mt-0.5">
                      Click any row to open the full evidence view · Priority = 6-dimension heuristic score (max 100)
                    </p>
                  </div>
                  <span className="text-xs text-gray-400">{flagged.length} signals</span>
                </div>
              </CardHeader>
              <div className="overflow-x-auto">
                <table className="w-full text-sm min-w-[900px]">
                  <thead className="bg-gray-50 border-b border-gray-200">
                    <tr>
                      {[
                        'Drug', 'Adverse Event', 'Priority', 'PRR',
                        '95% CI', 'Chi² / Test', 'p-value',
                        'n', 'Trend', 'Serious Rate', 'Cluster', '',
                      ].map(h => (
                        <th
                          key={h}
                          className="text-left text-xs font-semibold text-gray-500 uppercase tracking-wide px-3 py-2 whitespace-nowrap"
                        >
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {flagged.map((s, i) => (
                      <SignalRow
                        key={i}
                        signal={s}
                        priority={priorityMap.get(`${s.drug_name}|${s.adverse_event}`)}
                        onSelect={setSelected}
                      />
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          ) : (
            <div className="text-center py-10 text-gray-400 text-sm">
              No signals meet the selected thresholds.
            </div>
          )}

          {signals.flagged_signals.length > 0 && (
            <p className="text-xs text-gray-400">
              ⚠️ HEURISTIC SCREENING ONLY. PRR is a statistical disproportionality measure.
              It does not establish causality. Priority scores are not FDA-approved risk scores.
              All results require review by a qualified pharmacovigilance professional.
            </p>
          )}
        </>
      )}

      {!signals && !loading && (
        <div className="text-center py-14 text-gray-400 text-sm space-y-2">
          <p className="text-3xl">🔍</p>
          <p>Select parameters and click <strong>Analyze Signals</strong> to scan the dataset.</p>
          <p className="text-xs text-gray-300">Default: all drugs · PRR ≥ 2 · n ≥ 3 · p ≤ 0.05</p>
        </div>
      )}

      {/* Evidence drawer */}
      {selected && (
        <EvidenceDrawer
          signal={selected}
          priorityScore={priorityMap.get(`${selected.drug_name}|${selected.adverse_event}`) ?? null}
          onClose={() => setSelected(null)}
        />
      )}
    </div>
  )
}
