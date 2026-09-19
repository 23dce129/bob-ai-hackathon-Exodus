import axios from 'axios'
import type {
  AnalyzeRequest,
  AnalyzeResponse,
  TrendResponse,
  ReadinessResponse,
  GapReportResponse,
  TraceResponse,
} from './types'

const api = axios.create({ baseURL: '/api' })

// ---------------------------------------------------------------------------
// Signals
// ---------------------------------------------------------------------------

export async function analyzeSignals(req: AnalyzeRequest): Promise<AnalyzeResponse> {
  const { data } = await api.post<AnalyzeResponse>('/signals/analyze', {
    drug_name: req.drug_name ?? null,
    min_prr: req.min_prr ?? 2.0,
    min_cases: req.min_cases ?? 3,
    max_pval: req.max_pval ?? 0.05,
    include_trends: req.include_trends ?? true,
  })
  return data
}

export async function getTrend(drug: string, event: string): Promise<TrendResponse> {
  const { data } = await api.get<TrendResponse>(
    `/signals/trend/${encodeURIComponent(drug)}/${encodeURIComponent(event)}`
  )
  return data
}

export async function getSampleSignals(): Promise<AnalyzeResponse> {
  const { data } = await api.get<AnalyzeResponse>('/signals/sample')
  return data
}

// ---------------------------------------------------------------------------
// Dossier / CTD
// ---------------------------------------------------------------------------

export async function checkSampleReadiness(): Promise<ReadinessResponse> {
  const { data } = await api.get<ReadinessResponse>('/dossier/sample-check')
  return data
}

export async function checkDossierFile(
  file: File,
  drug_name?: string
): Promise<ReadinessResponse> {
  const formData = new FormData()
  formData.append('file', file)
  if (drug_name) {
    formData.append('drug_name', drug_name)
  }
  const { data } = await api.post<ReadinessResponse>('/dossier/check', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export async function checkDossierText(
  drug_name: string,
  outline_text: string
): Promise<ReadinessResponse> {
  const { data } = await api.post<ReadinessResponse>('/dossier/check', {
    drug_name,
    outline_text,
    source: 'text',
  })
  return data
}

export async function getGapReport(submissionType: string): Promise<GapReportResponse> {
  const { data } = await api.get<GapReportResponse>(
    `/dossier/gap-report/${submissionType.toUpperCase()}`
  )
  return data
}

export async function getPriorityScores(
  drug_name?: string,
  min_prr = 2.0,
  min_cases = 3,
  max_pval = 0.05,
): Promise<import('./types').PriorityResponse> {
  const { data } = await api.get('/signals/priority', {
    params: { drug_name: drug_name ?? undefined, min_prr, min_cases, max_pval },
  })
  return data
}

export async function getSignalExplanation(
  drug: string,
  event: string,
): Promise<import('./types').SignalExplanation> {
  const { data } = await api.get(
    `/signals/explain/${encodeURIComponent(drug)}/${encodeURIComponent(event)}`
  )
  return data
}

export async function getModules() {
  const { data } = await api.get('/dossier/modules')
  return data
}

// ---------------------------------------------------------------------------
// Traceability
// ---------------------------------------------------------------------------

export async function traceSignals(
  signals: Array<{ drug_name: string; adverse_event: string; signal_flag: boolean }>,
  section_numbers: string[]
): Promise<TraceResponse> {
  const { data } = await api.post<TraceResponse>('/traceability/trace-text', {
    signals,
    section_numbers,
  })
  return data
}

export async function getTraceCategories() {
  const { data } = await api.get('/traceability/categories')
  return data
}
