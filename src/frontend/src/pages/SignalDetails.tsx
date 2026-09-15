import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import {
  XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, ReferenceLine, Area, AreaChart,
} from 'recharts'
import { getTrend, analyzeSignals } from '../api/client'
import type { TrendResponse, Signal } from '../api/types'
import { Card, CardBody, CardHeader } from '../components/Card'
import { LoadingOverlay } from '../components/Spinner'
import { TrendBadge, PrrBadge } from '../components/Badge'

function ConfidenceGauge({ prr, ci_lower, ci_upper }: { prr?: number; ci_lower?: number; ci_upper?: number }) {
  if (!prr) return null
  // Narrow CI = high confidence
  const ciWidth = ci_upper != null && ci_lower != null ? ci_upper - ci_lower : null
  const confidence = ciWidth != null ? Math.max(0, Math.min(100, 100 - (ciWidth / prr) * 30)) : null
  return (
    <div className="space-y-1">
      <div className="flex items-baseline gap-2">
        <span className="text-4xl font-bold text-gray-900">{prr.toFixed(2)}</span>
        <span className="text-sm text-gray-500">PRR</span>
      </div>
      {ci_lower != null && ci_upper != null && (
        <p className="text-xs text-gray-500">95% CI: {ci_lower.toFixed(2)} – {ci_upper.toFixed(2)}</p>
      )}
      {confidence != null && (
        <div className="mt-2">
          <div className="flex justify-between text-xs text-gray-500 mb-1">
            <span>Estimate precision</span>
            <span>{confidence.toFixed(0)}%</span>
          </div>
          <div className="h-1.5 bg-gray-100 rounded-full overflow-hidden">
            <div
              className="h-full bg-pharma-500 rounded-full"
              style={{ width: `${confidence}%` }}
            />
          </div>
        </div>
      )}
    </div>
  )
}

export function SignalDetails() {
  const { drug, event } = useParams<{ drug: string; event: string }>()
  const [trend, setTrend] = useState<TrendResponse | null>(null)
  const [signal, setSignal] = useState<Signal | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!drug || !event) return
    setLoading(true)
    Promise.all([
      getTrend(drug, event),
      analyzeSignals({ drug_name: drug, include_trends: true }),
    ])
      .then(([t, s]) => {
        setTrend(t)
        const match = s.flagged_signals.find(
          x => x.drug_name.toLowerCase() === drug.toLowerCase()
          && x.adverse_event.toLowerCase() === event.toLowerCase()
        ) ?? s.all_signals?.find(
          x => x.drug_name.toLowerCase() === drug.toLowerCase()
          && x.adverse_event.toLowerCase() === event.toLowerCase()
        ) ?? null
        setSignal(match)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [drug, event])

  if (loading) return <LoadingOverlay text="Loading signal details…" />
  if (error)   return (
    <div>
      <p className="text-red-600 text-sm p-4">Error: {error}</p>
      <Link to="/signals" className="text-sm text-pharma-600 hover:underline ml-4">← Back to Signal Detection</Link>
    </div>
  )
  if (!trend)  return <p className="text-gray-500 text-sm p-4">No data found for this pair.</p>

  const quarterData = trend.quarters.map(q => ({ quarter: q.quarter, count: q.count }))
  const midIdx      = Math.floor(quarterData.length / 2)
  const prevAvg     = quarterData.slice(0, midIdx).reduce((s, q) => s + q.count, 0) / Math.max(1, midIdx)

  return (
    <div className="space-y-5 max-w-4xl">
      <div className="flex items-center gap-2">
        <Link to="/signals" className="text-xs text-pharma-600 hover:underline">
          ← Signal Detection
        </Link>
        <span className="text-gray-300">/</span>
        <span className="text-sm text-gray-700 font-medium">{drug} / {event}</span>
      </div>

      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-xl font-semibold text-gray-900">
            {drug} <span className="text-gray-400">·</span> {event}
          </h1>
          <p className="text-sm text-gray-500 mt-0.5">
            Signal trend analysis · {trend.date_range_start} to {trend.date_range_end}
          </p>
        </div>
        <TrendBadge direction={trend.direction} />
      </div>

      {/* PRR + stats row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardHeader><h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide">PRR Statistics</h3></CardHeader>
          <CardBody>
            <ConfidenceGauge
              prr={signal?.prr}
              ci_lower={signal?.ci_lower_95}
              ci_upper={signal?.ci_upper_95}
            />
            {signal?.p_value != null && (
              <p className="text-xs text-gray-500 mt-2">
                p-value: <span className="font-mono">{signal.p_value < 0.001 ? '<0.001' : signal.p_value.toFixed(4)}</span>
                {' · '}{signal.test_method?.toUpperCase()}
              </p>
            )}
            {!signal?.prr && (
              <p className="text-xs text-gray-400">PRR data not available for this pair at current thresholds.</p>
            )}
          </CardBody>
        </Card>

        <Card>
          <CardHeader><h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Reporting Volume</h3></CardHeader>
          <CardBody className="space-y-3">
            <div className="flex justify-between">
              <span className="text-xs text-gray-500">Total Reports</span>
              <span className="text-sm font-semibold text-gray-800">{trend.total_reports}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-xs text-gray-500">Recent Period</span>
              <span className="text-sm font-semibold text-gray-800">{trend.recent_count}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-xs text-gray-500">Previous Period</span>
              <span className="text-sm font-semibold text-gray-800">{trend.previous_count}</span>
            </div>
            <div className="flex justify-between border-t border-gray-100 pt-2">
              <span className="text-xs text-gray-500">Change</span>
              <span className={`text-sm font-semibold ${trend.pct_change > 0 ? 'text-red-600' : 'text-green-600'}`}>
                {trend.pct_change > 0 ? '+' : ''}{trend.pct_change.toFixed(1)}%
              </span>
            </div>
          </CardBody>
        </Card>

        <Card>
          <CardHeader><h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Trend Assessment</h3></CardHeader>
          <CardBody className="space-y-2">
            <div className="flex items-center gap-2">
              <TrendBadge direction={trend.direction} />
            </div>
            <div className="flex justify-between">
              <span className="text-xs text-gray-500">Trend Score</span>
              <span className={`text-sm font-mono font-semibold ${
                trend.trend_score > 0 ? 'text-red-600' : trend.trend_score < 0 ? 'text-green-600' : 'text-gray-600'
              }`}>
                {trend.trend_score > 0 ? '+' : ''}{trend.trend_score.toFixed(4)}
              </span>
            </div>
            <p className="text-xs text-gray-400 mt-1">
              Score range: −1.0 (declining) to +1.0 (emerging).
            </p>
            {signal && (
              <div className="mt-2 pt-2 border-t border-gray-100">
                <PrrBadge prr={signal.prr} />
              </div>
            )}
          </CardBody>
        </Card>
      </div>

      {/* Quarterly trend chart */}
      <Card>
        <CardHeader>
          <h2 className="text-sm font-semibold text-gray-700">Quarterly Report Count</h2>
        </CardHeader>
        <CardBody>
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={quarterData} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#6170f6" stopOpacity={0.2} />
                  <stop offset="95%" stopColor="#6170f6" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="quarter" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} allowDecimals={false} />
              <Tooltip contentStyle={{ fontSize: 12 }} />
              <ReferenceLine
                y={prevAvg}
                stroke="#9ca3af"
                strokeDasharray="4 2"
                label={{ value: 'Prev avg', fontSize: 10, fill: '#9ca3af' }}
              />
              <Area
                type="monotone"
                dataKey="count"
                stroke="#6170f6"
                strokeWidth={2}
                fill="url(#areaGrad)"
                dot={{ r: 3, fill: '#6170f6' }}
              />
            </AreaChart>
          </ResponsiveContainer>
          <p className="text-xs text-gray-400 mt-2 text-center">
            Dashed line = average of first half of dataset period.
            Upward deviation indicates emerging signal activity.
          </p>
        </CardBody>
      </Card>

      {/* Evidence panel */}
      <Card>
        <CardHeader>
          <h2 className="text-sm font-semibold text-gray-700">Evidence Summary</h2>
        </CardHeader>
        <CardBody className="space-y-2 text-sm text-gray-600">
          <div className="bg-amber-50 border border-amber-200 rounded p-3 text-xs text-amber-800">
            <strong>Statistical association only.</strong> These results are derived from a synthetic
            demonstration dataset and use disproportionality analysis (PRR). This does not establish
            a causal relationship between <em>{drug}</em> and <em>{event}</em>.
          </div>
          <div className="grid grid-cols-2 gap-4 mt-2 text-xs">
            <div>
              <p className="font-medium text-gray-700 mb-1">Interpretation Guidance</p>
              <ul className="space-y-0.5 text-gray-500 list-disc list-inside">
                <li>PRR ≥ 2 = disproportionate reporting vs. background</li>
                <li>Increasing trend = more recent reports relative to earlier</li>
                <li>Wide CI = fewer cases, less stable estimate</li>
              </ul>
            </div>
            <div>
              <p className="font-medium text-gray-700 mb-1">Next Steps</p>
              <ul className="space-y-0.5 text-gray-500 list-disc list-inside">
                <li>Review clinical literature for mechanism</li>
                <li>Check CTD documentation coverage</li>
                <li>
                  <Link to="/action-brief" className="text-pharma-600 hover:underline">
                    Generate Regulatory Action Brief →
                  </Link>
                </li>
              </ul>
            </div>
          </div>
        </CardBody>
      </Card>

      <p className="text-xs text-gray-400">{trend.disclaimer}</p>
    </div>
  )
}
