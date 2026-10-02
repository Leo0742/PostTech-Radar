export type NavPage = 'processing' | 'history' | 'analytics' | 'overview' | 'new' | 'quality' | 'data'

export interface DatasetOptions {
  services: string[]
  components: string[]
  request_types: string[]
  criticalities: string[]
  urgencies: string[]
  priorities: string[]
  service_classes: string[]
  timezones: string[]
  categories: string[]
  support_lines: string[]
}

export interface DatasetSummary {
  records: number
  distinct_categories: number
  top15_records: number
  top15: { name: string; count: number }[]
  configured_top_k?: number
  dataset_sha256?: string
  support_lines?: Record<string, number>
}

export interface PredictionOption {
  label: string
  confidence: number
}

export interface SimilarTicket {
  request_id: string
  similarity: number
  description: string
  category: string
  final_line: string
  result: string
  overdue: boolean
  duration: string | null
  clarifications: number
}

export interface AnalysisResult {
  category: {
    label: string
    confidence: number
    accepted: boolean
    review_required: boolean
    threshold: number
    margin_threshold: number
    review_reason?: string
    alternatives: PredictionOption[]
    signals: string[]
    explanation: string
  }
  routing: {
    label: string
    confidence: number
    accepted: boolean
    review_required: boolean
    threshold: number
    review_reason?: string
    mode?: string
    category_prior_used?: boolean
    alternatives: PredictionOption[]
    signals: string[]
    explanation: string
  }
  sla_risk: { risk: number; sample_size: number; level: string; explanation: string; basis?: string; support_n?: number; reliability?: string; interval_95?: number[] }
  retrieval?: { rejected: boolean; reason: string; threshold: number; best_relevance: number }
  similar: SimilarTicket[]
  model_provenance?: {
    candidate_id?: string
    family?: string
    model_family?: string
    model_id?: string
    model_revision?: string
    profile?: string
    trained_at?: string
    policy?: string
    fallback?: boolean
    fallback_reason?: string
  }
}

export interface QwenRecheckResult {
  category: {
    label: string
    confidence: number
    alternatives: PredictionOption[]
  }
  model_provenance?: {
    candidate_id?: string
    model_id?: string
    model_revision?: string
    profile?: string
    embedding_dim?: number
    runtime?: string
    ephemeral?: boolean
  }
  latency_seconds: number
}

export interface IncomingBatch {
  id: string
  filename: string
  status: 'uploaded' | 'analyzing' | 'waiting_review' | 'saved' | string
  total_rows: number
  valid_rows: number
  invalid_rows: number
  recognized_columns: string[]
  missing_optional: string[]
  description_column?: string | null
  labeled_historical?: boolean
  created_at: string
  analyzed_at?: string | null
  state_counts: Record<string, number>
  duplicate_ids?: string[]
  ticket_id?: number
}

export interface IncomingTicket {
  id: number
  batch_id: string
  row_number: number
  request_id?: string | null
  registration_date?: string | null
  user_name?: string | null
  service?: string | null
  component?: string | null
  request_type?: string | null
  description: string
  criticality?: string | null
  urgency?: string | null
  priority?: string | null
  service_class?: string | null
  timezone?: string | null
  state: 'uploaded' | 'waiting_review' | 'saved' | 'failed' | string
  analysis: AnalysisResult | null
  model_category?: string | null
  model_category_confidence?: number | null
  model_route?: string | null
  model_route_confidence?: number | null
  operator_category?: string | null
  operator_route?: string | null
  category_confirmed: number
  route_confirmed: number
  reviewed_at?: string | null
  saved_request_id?: string | null
  error_text?: string | null
  low_information?: boolean
  prediction_timestamp?: string | null
  learning_priority?: number
  needs_training_review?: boolean
  attention: boolean
  raw: Record<string, unknown>
}

export interface BreakdownRow {
  name: string
  count: number
  overdue: number
  sample_n: number
  raw_overdue_rate: number
  overdue_share: number
  smoothed_risk: number
  median_duration_seconds: number
  p90_duration_seconds: number
  avg_duration_seconds: number
  avg_clarifications: number
}

export interface AnalyticsData {
  kpis: {
    tickets: number
    overdue: number
    overdue_share: number
    median_duration_seconds: number
    median_duration: string
    p90_duration_seconds: number
    p90_duration: string
    multi_line: number
    high_clarifications: number
    clarification_threshold: number
  }
  breakdowns: {
    categories: BreakdownRow[]
    services: BreakdownRow[]
    priorities: BreakdownRow[]
    support_lines: BreakdownRow[]
  }
  time: { month: string; count: number; overdue: number; overdue_share: number; smoothed_risk: number }[]
  methodology: Record<string, string>
}

export interface ProcessData {
  status_history: { available: boolean; message: string; required_columns: string[]; event_count?: number; request_count?: number; transitions: { from: string; to: string; count: number; share: number }[]; dwell: { status: string; median_seconds: number; sample_n: number }[] }
  participation: { edge_semantics: string; combinations: { combination: string; count: number; overdue_rate: number; median_duration_seconds: number }[]; resolver_edges: { participant: string; resolver: string; count: number }[]; co_participation: { line_a: string; line_b: string; count: number }[]; duration_groups: { single_line: { sample_n: number; median_seconds: number; p90_seconds: number }; multi_line: { sample_n: number; median_seconds: number; p90_seconds: number } }; multi_line_count: number; ticket_count: number }
}
