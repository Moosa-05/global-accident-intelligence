export type UserRole = 'OWNER' | 'ADMIN' | 'MEMBER' | 'VIEWER'

export type SubscriptionTier = 'FREE' | 'DEVELOPER' | 'PROFESSIONAL' | 'BUSINESS' | 'ENTERPRISE'

export interface User {
  id: string
  email: string
  full_name: string | null
  role: UserRole
  is_active: boolean
  organization_id: string
  created_at?: string
}

export interface Organization {
  id: string
  name: string
  slug: string
  subscription_tier: SubscriptionTier
  subscription_status: string
  is_commercial_cleared: boolean
  research_access_permitted: boolean
  rate_limit_per_minute: number
  monthly_request_quota: number
  current_month_requests: number
  quota_reset_date: string
  billing_email?: string
  created_at?: string
}

export interface AuthTokens {
  access_token: string
  refresh_token: string
  token_type: string
  expires_in_seconds: number
}

export interface LoginResponse extends AuthTokens {}

export interface RegisterResponse {
  message: string
  user: User
  organization: Organization
  tokens: AuthTokens
}
