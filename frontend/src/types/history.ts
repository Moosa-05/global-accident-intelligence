export interface PredictionHistoryItem {
  id: string;
  type: 'injury_severity' | 'exposure_risk' | 'corridor_risk' | 'traffic_severity';
  model_id: string;
  created_at: string | null;
  latitude: number | null;
  longitude: number | null;
  summary: string;
  result: Record<string, any>;
  latency_ms: number;
  model_version: string;
}

export interface PredictionHistoryResponse {
  total: number;
  records: PredictionHistoryItem[];
}

export interface AuditLogItem {
  id: string;
  endpoint: string;
  method: string;
  model_id: string | null;
  status_code: number;
  latency_ms: number;
  ip_address: string | null;
  created_at: string | null;
}
