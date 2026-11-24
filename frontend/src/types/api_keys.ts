export interface ApiKeyItem {
  id: string;
  name: string;
  key_prefix: string;
  scopes: string[];
  is_active: boolean;
  total_requests: number;
  last_used_at: string | null;
  created_at: string;
  expires_at: string | null;
  revoked_at: string | null;
}

export interface ApiKeyCreateRequest {
  name: string;
  scopes?: string[];
  is_live?: boolean;
  expires_in_days?: number | null;
}

export interface ApiKeyCreateResponse {
  id: string;
  name: string;
  key_prefix: string;
  api_key: string;
  scopes: string[];
  created_at: string;
  expires_at: string | null;
}

export interface ApiKeyRotateResponse {
  message: string;
  old_key_id: string;
  new_key: ApiKeyCreateResponse;
}
