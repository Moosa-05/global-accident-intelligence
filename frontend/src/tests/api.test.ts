import { describe, it, expect, vi, beforeEach } from 'vitest';
import { api } from '../services/api';

describe('Centralized ApiClient', () => {
  beforeEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
  });

  it('stores and retrieves access and refresh tokens', () => {
    api.setTokens({
      access_token: 'test_access_token',
      refresh_token: 'test_refresh_token',
      token_type: 'bearer',
      expires_in_seconds: 3600,
    });
    expect(localStorage.getItem('agy_access_token')).toBe('test_access_token');
    expect(localStorage.getItem('agy_refresh_token')).toBe('test_refresh_token');

    api.clearTokens();
    expect(localStorage.getItem('agy_access_token')).toBeNull();
    expect(localStorage.getItem('agy_refresh_token')).toBeNull();
  });

  it('attaches Authorization bearer token to HTTP requests', async () => {
    api.setTokens({
      access_token: 'valid_bearer_token',
      refresh_token: 'valid_refresh_token',
      token_type: 'bearer',
      expires_in_seconds: 3600,
    });

    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        user: {
          id: 'usr_123',
          email: 'test@accident-intel.io',
          role: 'ADMIN',
          organization_id: 'org_123',
        },
      }),
    });
    globalThis.fetch = mockFetch as any;

    const user = await api.getMe();
    expect(user.id).toBe('usr_123');
    expect(mockFetch).toHaveBeenCalledWith(
      '/api/v1/auth/me',
      expect.objectContaining({
        headers: expect.objectContaining({
          Authorization: 'Bearer valid_bearer_token',
        }),
      })
    );
  });

  it('captures server-issued X-Session-ID and attaches it to subsequent requests', async () => {
    sessionStorage.clear();
    const mockHeaders = new Map();
    mockHeaders.set('x-session-id', 'sess_abc123.hmac_signed_sig');

    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      headers: {
        get: (h: string) => mockHeaders.get(h.toLowerCase()),
      },
      json: async () => ({ models: [] }),
    });
    globalThis.fetch = mockFetch as any;

    await api.getModels();
    expect(sessionStorage.getItem('gai_session_token')).toBe('sess_abc123.hmac_signed_sig');

    // Next request should attach X-Session-ID header
    await api.getPredictionHistory(10);
    expect(mockFetch).toHaveBeenLastCalledWith(
      '/api/v1/history/predictions?limit=10',
      expect.objectContaining({
        headers: expect.objectContaining({
          'X-Session-ID': 'sess_abc123.hmac_signed_sig',
        }),
      })
    );
  });

  it('handles 403 Forbidden licensing violations with descriptive error', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 403,
      json: async () => ({
        detail: 'Commercial accounts barred from CC BY-NC-SA 4.0 research endpoints.',
      }),
    });
    globalThis.fetch = mockFetch as any;

    await expect(
      api.predictSeverity({
        latitude: 35.0,
        longitude: -85.0,
      })
    ).rejects.toThrow(/Commercial accounts barred/);
  });
});
