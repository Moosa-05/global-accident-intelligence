export interface PlanDetails {
  tier: string;
  name: string;
  price_monthly_usd: number;
  monthly_request_quota: number;
  rate_limit_per_minute: number;
  is_commercial_cleared: boolean;
  allowed_models: string[];
  features: string[];
}

export interface PlansListResponse {
  plans: PlanDetails[];
}

export interface SubscriptionInfo {
  organization_id: string;
  subscription_tier: string;
  subscription_status: string;
  is_commercial_cleared: boolean;
  research_access_permitted: boolean;
  monthly_request_quota: number;
  current_month_requests: number;
  rate_limit_per_minute: number;
  quota_reset_date: string | null;
  billing_email: string | null;
}

export interface SubscriptionChangeRequest {
  new_tier: 'FREE' | 'DEVELOPER' | 'PROFESSIONAL' | 'BUSINESS' | 'ENTERPRISE' | string;
}

export interface SubscriptionChangeResponse {
  organization_id: string;
  previous_tier: string;
  new_tier: string;
  monthly_request_quota: number;
  rate_limit_per_minute: number;
  is_commercial_cleared: boolean;
  status: string;
  message: string;
}
