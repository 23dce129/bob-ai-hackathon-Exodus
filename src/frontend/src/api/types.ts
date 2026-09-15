// ---------------------------------------------------------------------------
// Shared API types — mirror the FastAPI backend response shapes
// ---------------------------------------------------------------------------

export interface Signal {
  drug_name: string
  adverse_event: string
  a?: number
  b?: number
  c?: number
  d?: number
  prr?: number
  ci_lower_95?: number
  ci_upper_95?: number
  chi2_statistic?: number | null
  p_value?: number
  signal_flag: boolean
  test_method?: string
  trend?: {
    direction: 'increasing' | 'decreasing' | 'stable' | 'insufficient_data'
    trend_score?: number
    pct_change?: number
  }
}

export interface AnalyzeRequest {
  drug_name?: string | null
  min_prr?: number
  min_cases?: number
  max_pval?: number
  include_trends?: boolean
}

export interface AnalyzeResponse {
  drug_filter: string | null
  total_evaluated: number
  total_flagged: number
  flagged_signals: Signal[]
  all_signals: Signal[]
  thresholds: { min_prr: number; min_cases: number; max_pval: number }
  disclaimer: string
}

export interface QuarterCount {
  quarter: string
  count: number
}

export interface TrendResponse {
  drug_name: string
  adverse_event: string
  direction: 'increasing' | 'decreasing' | 'stable' | 'insufficient_data'
  trend_score: number
  recent_count: number
  previous_count: number
  pct_change: number
  total_reports: number
  date_range_start: string
  date_range_end: string
  quarters: QuarterCount[]
  disclaimer: string
}

// Dossier / CTD types
export interface SectionResult {
  ctd_section_id: string
  ctd_section_title: string
  status: 'PRESENT' | 'NEEDS_REVIEW' | 'MISSING' | 'NOT_APPLICABLE'
  required: boolean
  safety_relevant: boolean
  match_score?: number
  relevance_reason?: string
}

export interface ModuleScore {
  module_id: string
  module_title: string
  score: number
  present_count: number
  needs_review_count: number
  missing_count: number
  not_applicable_count: number
  critical_missing_count: number
  section_results: SectionResult[]
}

export interface ReadinessResponse {
  overall_score: number
  overall_score_pct: number
  total_present: number
  total_needs_review: number
  total_missing: number
  total_critical_missing: number
  module_scores: Record<string, ModuleScore>
  parse_metadata: Record<string, unknown>
  disclaimer: string
}

export interface GapSection {
  id: string
  title: string
  required: boolean
  conditional: boolean
  safety_relevant: boolean
  traceability_categories: string[]
  submission_types: string[]
}

export interface GapReportResponse {
  submission_type: string
  required_count: number
  conditional_count: number
  safety_relevant_required_count: number
  required_sections: GapSection[]
  conditional_sections: GapSection[]
  disclaimer: string
}

// Traceability types
export interface TraceEntry {
  ctd_section_id: string
  ctd_section_title: string
  status: 'PRESENT' | 'NEEDS_REVIEW' | 'MISSING' | 'NOT_APPLICABLE'
  required: boolean
  safety_relevant: boolean
  relevance_reason: string
}

export interface TraceResult {
  adverse_event: string
  signal_category: string
  traceability_score: number
  entries: TraceEntry[]
  critical_gaps: string[]
  covered_sections: string[]
  missing_sections: string[]
  disclaimer: string
}

// Priority score types
export interface ComponentScore {
  name: string
  earned: number
  maximum: number
  raw_value: number | null
  formula: string
}

export interface PriorityScore {
  drug_name: string
  adverse_event: string
  final_score: number
  max_possible_score: number
  components: ComponentScore[]
  disclaimer: string
}

export interface PriorityResponse {
  total_flagged: number
  priority_scores: PriorityScore[]
  disclaimer: string
}

// Signal explanation
export interface SignalExplanation {
  drug_name: string
  adverse_event: string
  prr: number | null
  ci_lower_95: number | null
  ci_upper_95: number | null
  p_value: number | null
  chi2_statistic: number | null
  test_method: string
  total_reports: number
  serious_count: number
  serious_rate: number
  trend_direction: string
  trend_score: number
  pct_change: number
  priority_score: number
  max_score: number
  components: ComponentScore[]
  explanation_paragraphs: string[]
  disclaimer: string
}

export interface TraceResponse {
  drug_name: string
  overall_score: number
  critical_gap_count: number
  fully_documented: boolean
  results: TraceResult[]
  disclaimer: string
}
