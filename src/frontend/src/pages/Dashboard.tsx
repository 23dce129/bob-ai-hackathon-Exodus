import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { analyzeSignals, checkSampleReadiness } from '../api/client'
import type { AnalyzeResponse, ReadinessResponse } from '../api/types'
import { Card, CardBody, CardHeader } from '../components/Card'
import { LoadingOverlay } from '../components/Spinner'
import { PrrBadge, TrendBadge } from '../components/Badge'

// ---------------------------------------------------------------------------
// KPI card
// ---------------------------------------------------------------------------
function KpiCard({
  label, value, sub, accent, icon,
}: {
  label: string
  value: string | number
  sub?: string
  accent?: string
  icon: string
}) {
  return (
    <Card className="flex-1">
      <CardBody className="flex items-start gap-4">
        <span className="text-2xl leading-none mt-0.5">{icon}</span>
        <div className="min-w-0">
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide leading-none mb-1">
            {label}
          </p>
          <p className={`text-4xl font-bold leading-none ${accent ?? 'text-gray-900'}`}>{value}</p>
          {sub && <p className="text-xs text-gray-400 mt-1.5">{sub}</p>}
        </div>
      </CardBody>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Primary action button
// ---------------------------------------------------------------------------
function PrimaryBtn({ to, children }: { to: string; children: React.ReactNode }) {
  return (
    <Link
      to={to}
      className="flex-1 flex items-center justify-center gap-2 bg-pharma-700 hover:bg-pharma-800 text-white text-sm font-semibold px-6 py-3 rounded-lg shadow-sm transition-colors"
    >
      {children}
    </Link>
  )
}

// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------
export function Dashboard() {
  const [signals, setSignals]   = useState<AnalyzeResponse | null>(null)
  const [readiness, setReadiness] = useState<ReadinessResponse | null>(null)
  const [loading, setLoading]   = useState(true)
  const [error, setError]       = useState<string | null>(null)
  const navigate = useNavigate()

  useEffect(() => {
    Promise.all([
      analyzeSignals({ include_trends: true }),
      checkSampleReadiness(),
    ])
      .then(([s, r]) => { setSignals(s); setReadiness(r) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <LoadingOverlay text="Loading dashboard data…" />
  if (error)   return (
    <div className="bg-red-50 border border-red-200 rounded-lg p-5 text-sm text-red-700 max-w-lg">
      <strong>Backend error:</strong> {error}
      <p className="mt-1 text-xs text-red-500">
        Ensure the backend is running: <code>cd src/backend &amp;&amp; uvicorn main:app --port 8000</code>
      </p>
    </div>
  )

  // ── derived values from real API data ──────────────────────────────────
  const flagged        = signals?.flagged_signals ?? []
  const totalSignals   = signals?.total_flagged ?? 0
  const highPriority   = flagged.filter(s => (s.prr ?? 0) >= 5 || s.trend?.direction === 'increasing').length
  const readinessPct   = readiness?.overall_score_pct ?? 0
  const critGaps       = readiness?.total_critical_missing ?? 0

  const topSignals  = flagged.slice(0, 6)
  const chartData   = flagged.slice(0, 8).map(s => ({
    name: `${s.drug_name} / ${s.adverse_event}`,
    prr:  s.prr ?? 0,
    dir:  s.trend?.direction ?? 'stable',
  }))

  return (
    <div className="space-y-7 max-w-6xl">

      {/* ── Page header ─────────────────────────────────────────────────── */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900 tracking-tight">PharmaGuard AI</h1>
        <p className="text-sm text-gray-500 mt-1">
          Pharmacovigilance &amp; regulatory submission readiness · synthetic FAERS demo dataset
        </p>
      </div>

      {/* ── 4 KPI cards ─────────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <KpiCard
          icon="⚠️"
          label="Potential Safety Signals"
          value={totalSignals}
          sub={`of ${signals?.total_evaluated ?? 0} drug-event pairs evaluated`}
          accent={totalSignals > 5 ? 'text-red-600' : 'text-gray-900'}
        />
        <KpiCard
          icon="🔴"
          label="High Priority Signals"
          value={highPriority}
          sub="PRR ≥ 5 or increasing trend"
          accent={highPriority > 0 ? 'text-orange-600' : 'text-green-600'}
        />
        <KpiCard
          icon="📋"
          label="Overall CTD Readiness"
          value={`${readinessPct.toFixed(0)}%`}
          sub="ICH M4 CTD weighted completeness"
          accent={
            readinessPct >= 75 ? 'text-green-600' :
            readinessPct >= 50 ? 'text-yellow-600' : 'text-red-600'
          }
        />
        <KpiCard
          icon="🚨"
          label="Critical CTD Gaps"
          value={critGaps}
          sub="Safety-relevant sections missing"
          accent={critGaps > 0 ? 'text-red-600' : 'text-green-600'}
        />
      </div>

      {/* ── Primary CTAs ────────────────────────────────────────────────── */}
      <div className="flex flex-col sm:flex-row gap-3">
        <PrimaryBtn to="/signals">
          🔍 Analyze Safety Signals
        </PrimaryBtn>
        <PrimaryBtn to="/readiness">
          📋 Check Submission Readiness
        </PrimaryBtn>
      </div>

      {/* ── Secondary CTA ───────────────────────────────────────────────── */}
      <div>
        <Link
          to="/action-brief"
          className="inline-flex items-center gap-2 border-2 border-pharma-600 text-pharma-700 hover:bg-pharma-50 text-sm font-semibold px-6 py-2.5 rounded-lg transition-colors"
        >
          ⚡ Generate Regulatory Action Brief
        </Link>
        <p className="text-xs text-gray-400 mt-1.5 ml-1">
          Combines signal detection + CTD traceability into a prioritised action list.
        </p>
      </div>

      {/* ── PRR chart ───────────────────────────────────────────────────── */}
      {chartData.length > 0 && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-sm font-semibold text-gray-800">
                  Flagged Signal PRR Values
                </h2>
                <p className="text-xs text-gray-400 mt-0.5">
                  Proportional Reporting Ratio — higher = stronger disproportionality vs. background
                </p>
              </div>
              <div className="flex gap-3 text-xs text-gray-500 shrink-0">
                <span><span className="inline-block w-2 h-2 rounded-sm bg-red-500 mr-1" />≥ 10 Critical</span>
                <span><span className="inline-block w-2 h-2 rounded-sm bg-orange-400 mr-1" />5–10 High</span>
                <span><span className="inline-block w-2 h-2 rounded-sm bg-yellow-400 mr-1" />2–5 Moderate</span>
              </div>
            </div>
          </CardHeader>
          <CardBody>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={chartData} margin={{ top: 4, right: 8, left: 0, bottom: 64 }}>
                <XAxis
                  dataKey="name"
                  tick={{ fontSize: 10 }}
                  interval={0}
                  angle={-38}
                  textAnchor="end"
                />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip
                  formatter={(val: unknown) => [`PRR: ${(val as number).toFixed(2)}`, '']}
                  contentStyle={{ fontSize: 12 }}
                />
                <Bar dataKey="prr" radius={[3, 3, 0, 0]}>
                  {chartData.map((d, i) => (
                    <Cell
                      key={i}
                      fill={
                        d.prr >= 10 ? '#ef4444' :
                        d.prr >= 5  ? '#f97316' :
                                      '#eab308'
                      }
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </CardBody>
        </Card>
      )}

      {/* ── Two-column bottom row ────────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">

        {/* Top signals table */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold text-gray-800">Top Flagged Signals</h2>
              <Link to="/signals" className="text-xs text-pharma-600 hover:underline font-medium">
                Full analysis →
              </Link>
            </div>
          </CardHeader>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 border-b border-gray-100">
                <tr>
                  {['Drug', 'Event', 'PRR', 'Trend'].map(h => (
                    <th key={h} className="text-left text-xs font-medium text-gray-500 uppercase tracking-wide px-4 py-2">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-50">
                {topSignals.map((s, i) => (
                  <tr
                    key={i}
                    className="hover:bg-pharma-50 cursor-pointer"
                    onClick={() => navigate(`/signals/${encodeURIComponent(s.drug_name)}/${encodeURIComponent(s.adverse_event)}`)}
                  >
                    <td className="px-4 py-2 font-medium text-gray-800">{s.drug_name}</td>
                    <td className="px-4 py-2 text-gray-600 text-xs">{s.adverse_event}</td>
                    <td className="px-4 py-2"><PrrBadge prr={s.prr} /></td>
                    <td className="px-4 py-2"><TrendBadge direction={s.trend?.direction} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        {/* CTD module readiness */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold text-gray-800">CTD Readiness by Module</h2>
              <Link to="/readiness" className="text-xs text-pharma-600 hover:underline font-medium">
                Full report →
              </Link>
            </div>
          </CardHeader>
          <CardBody className="space-y-3">
            {readiness && Object.entries(readiness.module_scores).map(([mid, m]) => {
              const pct = Math.round(m.score * 100)
              const bar =
                pct >= 75 ? 'bg-green-500' :
                pct >= 50 ? 'bg-yellow-400' :
                pct >= 25 ? 'bg-orange-400' : 'bg-red-500'
              const txt =
                pct >= 75 ? 'text-green-700' :
                pct >= 50 ? 'text-yellow-700' :
                pct >= 25 ? 'text-orange-600' : 'text-red-600'
              return (
                <div key={mid} className="flex items-center gap-3">
                  <span className="text-xs font-mono font-bold text-pharma-700 w-6 shrink-0">{mid}</span>
                  <span className="text-xs text-gray-600 w-36 truncate shrink-0">{m.module_title}</span>
                  <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
                    <div className={`h-full rounded-full transition-all ${bar}`} style={{ width: `${pct}%` }} />
                  </div>
                  <span className={`text-xs font-bold w-10 text-right shrink-0 ${txt}`}>{pct}%</span>
                  {m.critical_missing_count > 0 && (
                    <span className="text-xs text-red-600 font-semibold shrink-0">
                      ⚠ {m.critical_missing_count}
                    </span>
                  )}
                </div>
              )
            })}
            <div className="pt-2 border-t border-gray-100 flex gap-4 text-xs text-gray-500">
              <span className="text-green-600">✓ {readiness?.total_present ?? 0} present</span>
              <span className="text-yellow-600">~ {readiness?.total_needs_review ?? 0} review</span>
              <span className="text-red-500">✗ {readiness?.total_missing ?? 0} missing</span>
            </div>
          </CardBody>
        </Card>

      </div>
    </div>
  )
}
