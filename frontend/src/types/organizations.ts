import type { SubscriptionTier, UserRole } from './auth';

export interface OrganizationUsageResponse {
  organization_id: string;
  organization_name: string;
  subscription_tier: SubscriptionTier;
  monthly_request_quota: number;
  current_month_requests: number;
  remaining_quota: number;
  quota_utilization_pct: number;
  rate_limit_per_minute: number;
  is_commercial_cleared: boolean;
  research_access_permitted: boolean;
  quota_reset_date: string;
}

export interface OrganizationMember {
  id: string;
  email: string;
  full_name: string | null;
  role: UserRole;
  is_active: boolean;
  created_at: string;
}
