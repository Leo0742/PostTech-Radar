export type NavPage = 'processing' | 'history' | 'analytics' | 'process' | 'system' | 'overview' | 'new' | 'quality' | 'data'

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
  model_provenance?: { candidate_id?: string; family?: string; trained_at?: string; policy?: string }
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
