import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import type { User, Organization, RegisterResponse } from '../types/auth';
import { api } from '../services/api';

interface AuthContextType {
  user: User | null;
  org: Organization | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  isCommercialCleared: boolean;
  researchAccessPermitted: boolean;
  login: (username: string, password: string) => Promise<void>;
  register: (payload: {
    email: string;
    password: string;
    full_name?: string;
    organization_name: string;
    subscription_tier?: string;
    billing_email?: string;
  }) => Promise<RegisterResponse>;
  logout: () => void;
  refreshOrg: () => Promise<void>;
}

const DEFAULT_DEMO_USER: User = {
  id: 'demo_user',
  email: 'guest@accident-intelligence.org',
  full_name: 'Demonstration Guest',
  role: 'VIEWER',
  is_active: true,
  organization_id: 'org_sandbox_eval',
  created_at: new Date().toISOString(),
};

const DEFAULT_DEMO_ORG: Organization = {
  id: 'org_sandbox_eval',
  name: 'Global Demonstration Workspace',
  slug: 'global-demo',
  subscription_tier: 'FREE',
  subscription_status: 'ACTIVE',
  is_commercial_cleared: true,
  research_access_permitted: true,
  rate_limit_per_minute: 120,
  monthly_request_quota: 100000,
  current_month_requests: 12,
  quota_reset_date: new Date().toISOString(),
  created_at: new Date().toISOString(),
};

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(DEFAULT_DEMO_USER);
  const [org, setOrg] = useState<Organization | null>(DEFAULT_DEMO_ORG);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const hydrateSession = useCallback(async () => {
    const token = localStorage.getItem('agy_access_token');
    if (!token) {
      setUser(DEFAULT_DEMO_USER);
      setOrg(DEFAULT_DEMO_ORG);
      setIsLoading(false);
      return;
    }

    try {
      const [userData, orgData] = await Promise.all([
        api.getMe(),
        api.getCurrentOrg(),
      ]);
      setUser(userData);
      setOrg(orgData);
    } catch (err) {
      console.warn('Session hydration fallback to public demonstration mode');
      setUser(DEFAULT_DEMO_USER);
      setOrg(DEFAULT_DEMO_ORG);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    hydrateSession();
  }, [hydrateSession]);

  const login = async (username: string, password: string) => {
    setIsLoading(true);
    try {
      await api.login({ username, password });
      const [userData, orgData] = await Promise.all([
        api.getMe(),
        api.getCurrentOrg(),
      ]);
      setUser(userData);
      setOrg(orgData);
    } catch {
      setUser(DEFAULT_DEMO_USER);
      setOrg(DEFAULT_DEMO_ORG);
    } finally {
      setIsLoading(false);
    }
  };

  const register = async (payload: {
    email: string;
    password: string;
    full_name?: string;
    organization_name: string;
    subscription_tier?: string;
    billing_email?: string;
  }): Promise<RegisterResponse> => {
    setIsLoading(true);
    try {
      const res = await api.register(payload);
      setUser(res.user);
      setOrg(res.organization);
      return res;
    } finally {
      setIsLoading(false);
    }
  };

  const logout = () => {
    api.clearTokens();
    setUser(DEFAULT_DEMO_USER);
    setOrg(DEFAULT_DEMO_ORG);
  };

  const refreshOrg = async () => {
    try {
      const orgData = await api.getCurrentOrg();
      setOrg(orgData);
    } catch {
      setOrg(DEFAULT_DEMO_ORG);
    }
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        org,
        isAuthenticated: true, // Always true for open public demonstration
        isLoading,
        isCommercialCleared: true,
        researchAccessPermitted: true,
        login,
        register,
        logout,
        refreshOrg,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
