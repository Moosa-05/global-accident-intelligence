import type {
  User,
  Organization,
  AuthTokens,
  LoginResponse,
  RegisterResponse,
} from '../types/auth';
import type { ModelsListResponse } from '../types/models';
import type {
  InjurySeverityRequest,
  InjurySeverityResponse,
  ExposureRiskRequest,
  ExposureRiskResponse,
  CorridorRiskRequest,
  CorridorRiskResponse,
  TrafficSeverityRequest,
  TrafficSeverityResponse,
} from '../types/predictions';
import type {
  PlansListResponse,
  SubscriptionInfo,
  SubscriptionChangeResponse,
} from '../types/subscriptions';
import type {
  ApiKeyItem,
  ApiKeyCreateRequest,
  ApiKeyCreateResponse,
  ApiKeyRotateResponse,
} from '../types/api_keys';
import type {
  OrganizationUsageResponse,
  OrganizationMember,
} from '../types/organizations';
import type {
  PredictionHistoryResponse,
  AuditLogItem,
} from '../types/history';

const API_BASE = '/api/v1';

class ApiClient {
  private getAccessToken(): string | null {
    return localStorage.getItem('agy_access_token');
  }

  private getRefreshToken(): string | null {
    return localStorage.getItem('agy_refresh_token');
  }

  public setTokens(tokens: AuthTokens | { access_token: string; refresh_token?: string }) {
    localStorage.setItem('agy_access_token', tokens.access_token);
    if ('refresh_token' in tokens && tokens.refresh_token) {
      localStorage.setItem('agy_refresh_token', tokens.refresh_token);
    }
  }

  private getSessionToken(): string | null {
    return sessionStorage.getItem('gai_session_token');
  }

  private setSessionToken(token: string) {
    if (token) {
      sessionStorage.setItem('gai_session_token', token);
    }
  }

  public clearTokens() {
    localStorage.removeItem('agy_access_token');
    localStorage.removeItem('agy_refresh_token');
    localStorage.removeItem('agy_user');
    localStorage.removeItem('agy_org');
    sessionStorage.removeItem('gai_session_token');
  }

  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${API_BASE}${endpoint}`;
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...(options.headers as Record<string, string>),
    };

    const token = this.getAccessToken();
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    const sessionToken = this.getSessionToken();
    if (sessionToken) {
      headers['X-Session-ID'] = sessionToken;
    }

    let response = await fetch(url, { ...options, headers });

    // Store server-issued cryptographically verified session token
    const returnedSessionId = response.headers?.get
      ? (response.headers.get('x-session-id') || response.headers.get('X-Session-ID'))
      : null;
    if (returnedSessionId) {
      this.setSessionToken(returnedSessionId);
    }

    // Handle 401 token refresh
    if (response.status === 401 && this.getRefreshToken() && !endpoint.includes('/auth/refresh')) {
      const refreshed = await this.refreshToken();
      if (refreshed) {
        headers['Authorization'] = `Bearer ${this.getAccessToken()}`;
        response = await fetch(url, { ...options, headers });
      }
    }

    if (!response.ok) {
      let errorData: any = {};
      try {
        errorData = await response.json();
      } catch {
        errorData = { detail: response.statusText || 'Unknown network error' };
      }
      const message = errorData.detail || errorData.message || `Request failed with status ${response.status}`;
      const error: any = new Error(typeof message === 'string' ? message : JSON.stringify(message));
      error.status = response.status;
      error.data = errorData;
      throw error;
    }

    return response.json();
  }

  private async refreshToken(): Promise<boolean> {
    const refreshToken = this.getRefreshToken();
    if (!refreshToken) return false;

    try {
      const res = await fetch(`${API_BASE}/auth/refresh`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: refreshToken }),
      });
      if (!res.ok) {
        this.clearTokens();
        return false;
      }
      const data: AuthTokens = await res.json();
      this.setTokens(data);
      return true;
    } catch {
      this.clearTokens();
      return false;
    }
  }

  // Auth Endpoints
  async register(payload: {
    email: string;
    password: string;
    full_name?: string;
    organization_name: string;
    subscription_tier?: string;
    billing_email?: string;
  }): Promise<RegisterResponse> {
    const res = await this.request<RegisterResponse>('/auth/register', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    this.setTokens(res.tokens);
    return res;
  }

  async login(formData: { username: string; password: string }): Promise<LoginResponse> {
    const res = await this.request<LoginResponse>('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email: formData.username, password: formData.password }),
    });
    this.setTokens(res);
    return res;
  }

  async getMe(): Promise<User> {
    const res = await this.request<{ user?: User; organization?: Organization } | User>('/auth/me');
    if ('user' in res && res.user) {
      return res.user;
    }
    return res as User;
  }

  // Organizations
  async getCurrentOrg(): Promise<Organization> {
    return this.request<Organization>('/organizations/current');
  }

  async getOrgUsage(): Promise<OrganizationUsageResponse> {
    return this.request<OrganizationUsageResponse>('/organizations/usage');
  }

  async getOrgMembers(): Promise<OrganizationMember[]> {
    return this.request<OrganizationMember[]>('/organizations/members');
  }

  // Subscriptions & Plans
  async getPlans(): Promise<PlansListResponse> {
    return this.request<PlansListResponse>('/subscriptions/plans');
  }

  async getCurrentSubscription(): Promise<SubscriptionInfo> {
    return this.request<SubscriptionInfo>('/subscriptions/current');
  }

  async changeSubscription(newTier: string): Promise<SubscriptionChangeResponse> {
    return this.request<SubscriptionChangeResponse>('/subscriptions/change-plan', {
      method: 'POST',
      body: JSON.stringify({ new_tier: newTier }),
    });
  }

  async cancelSubscription(): Promise<{ message: string }> {
    return this.request<{ message: string }>('/subscriptions/cancel', {
      method: 'POST',
    });
  }

  // API Keys
  async getApiKeys(): Promise<ApiKeyItem[]> {
    return this.request<ApiKeyItem[]>('/api-keys');
  }

  async createApiKey(payload: ApiKeyCreateRequest): Promise<ApiKeyCreateResponse> {
    return this.request<ApiKeyCreateResponse>('/api-keys', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  async rotateApiKey(keyId: string): Promise<ApiKeyRotateResponse> {
    return this.request<ApiKeyRotateResponse>(`/api-keys/${keyId}/rotate`, {
      method: 'POST',
    });
  }

  async revokeApiKey(keyId: string): Promise<{ message: string; key_id: string }> {
    return this.request<{ message: string; key_id: string }>(`/api-keys/${keyId}`, {
      method: 'DELETE',
    });
  }

  // Models
  async getModels(commercialOnly?: boolean): Promise<ModelsListResponse> {
    const q = commercialOnly !== undefined ? `?commercial_only=${commercialOnly}` : '';
    return this.request<ModelsListResponse>(`/models${q}`);
  }

  // Predictions
  async predictInjury(payload: InjurySeverityRequest): Promise<InjurySeverityResponse> {
    return this.request<InjurySeverityResponse>('/predictions/injury-severity', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  async predictExposure(payload: ExposureRiskRequest): Promise<ExposureRiskResponse> {
    return this.request<ExposureRiskResponse>('/predictions/exposure-risk', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  async predictRisk(payload: CorridorRiskRequest): Promise<CorridorRiskResponse> {
    return this.request<CorridorRiskResponse>('/predictions/risk', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  async predictSeverity(payload: TrafficSeverityRequest): Promise<TrafficSeverityResponse> {
    return this.request<TrafficSeverityResponse>('/predictions/severity', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  // History & Audit
  async getPredictionHistory(limit = 50, type?: string): Promise<PredictionHistoryResponse> {
    const query = new URLSearchParams();
    query.append('limit', limit.toString());
    if (type) query.append('prediction_type', type);
    return this.request<PredictionHistoryResponse>(`/history/predictions?${query.toString()}`);
  }

  async getAuditLogs(limit = 50): Promise<AuditLogItem[]> {
    return this.request<AuditLogItem[]>(`/history/audit-logs?limit=${limit}`);
  }
}

export const api = new ApiClient();
