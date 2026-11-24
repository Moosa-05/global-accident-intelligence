import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';
import { Badge } from '../../components/ui/Badge';
import {
  FileCode2,
  Copy,
  Check,
  Terminal,
  ShieldCheck,
  Lock,
  Layers,
  ArrowRight,
} from 'lucide-react';

export const DeveloperDocsPage: React.FC = () => {
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const copyCode = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const curlInjury = `curl -X POST "https://api.accident-intel.io/api/v1/predictions/injury-severity" \\
  -H "Content-Type: application/json" \\
  -H "X-API-Key: agy_live_99f381c828e84e55a722e0e09d57a916" \\
  -d '{
    "latitude": 51.5074,
    "longitude": -0.1278,
    "road_class": "MOTORWAY",
    "speed_limit_kmh": 110,
    "vehicle_type": "CAR",
    "number_of_vehicles": 2,
    "number_of_casualties": 1,
    "urban_or_rural": "RURAL",
    "light_conditions": "DAYLIGHT",
    "weather_conditions": "FINE_NO_HIGH_WINDS",
    "surface_conditions": "DRY"
  }'`;

  const pythonInjury = `import requests

url = "https://api.accident-intel.io/api/v1/predictions/injury-severity"
headers = {
    "X-API-Key": "agy_live_YOUR_SECRET_KEY",
    "Content-Type": "application/json"
}
payload = {
    "latitude": 51.5074,
    "longitude": -0.1278,
    "road_class": "MOTORWAY",
    "speed_limit_kmh": 110,
    "vehicle_type": "CAR",
    "number_of_vehicles": 2,
    "number_of_casualties": 1,
    "urban_or_rural": "RURAL"
}

response = requests.post(url, json=payload, headers=headers)
print(response.json())
# Output: {"predicted_severity": "SLIGHT", "probabilities": {...}, "fatality_risk_alert": false}`;

  const curlExposure = `curl -X POST "https://api.accident-intel.io/api/v1/predictions/exposure-risk" \\
  -H "Content-Type: application/json" \\
  -H "X-API-Key: agy_live_99f381c828e84e55a722e0e09d57a916" \\
  -d '{
    "latitude": 53.4808,
    "longitude": -2.2426,
    "road_class": "MOTORWAY",
    "road_name": "M62 Trans-Pennine",
    "annual_average_daily_flow": 48500,
    "segment_length_km": 2.5
  }'`;

  return (
    <div className="space-y-6 max-w-5xl">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
          <FileCode2 className="h-6 w-6 text-amber-400" />
          Developer Specifications & API Reference
        </h1>
        <p className="text-xs text-slate-400 mt-1">
          Complete REST API documentation for programmatic integration into autonomous fleets and telematics
        </p>
      </div>

      {/* Auth & Rate Limit Info */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-semibold flex items-center gap-2">
              <ShieldCheck className="h-4 w-4 text-emerald-400" />
              Authentication Headers
            </CardTitle>
          </CardHeader>
          <CardContent className="text-xs text-slate-300 space-y-2">
            <p>
              In Public Demonstration Mode, endpoints accept direct unauthenticated requests without credentials:
            </p>
            <pre className="p-2.5 rounded bg-slate-950 border border-slate-800 font-mono text-[11px] text-emerald-400">
              # Direct public requests allowed without headers
            </pre>
            <p className="text-slate-400 text-[11px]">
              Programmatic clients may also pass an optional <code className="text-slate-300 font-mono">X-API-Key: live_demo_key</code> or standard JWT Bearer token.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-semibold flex items-center gap-2">
              <Layers className="h-4 w-4 text-sky-400" />
              Rate Limits & Headers
            </CardTitle>
          </CardHeader>
          <CardContent className="text-xs text-slate-300 space-y-2">
            <p>Every response returns standard rate-limiting headers:</p>
            <ul className="space-y-1 font-mono text-[11px] text-slate-400">
              <li><span className="text-slate-200">X-RateLimit-Limit:</span> Max requests permitted per minute</li>
              <li><span className="text-slate-200">X-RateLimit-Remaining:</span> Quota left in current 60s sliding window</li>
              <li><span className="text-slate-200">X-RateLimit-Reset:</span> Epoch seconds until window resets</li>
            </ul>
          </CardContent>
        </Card>
      </div>

      {/* Endpoint 1: Human Injury Severity */}
      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <span className="px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 font-mono text-xs font-bold border border-emerald-800">
                POST
              </span>
              <span className="font-mono text-sm font-semibold text-slate-100">
                /api/v1/predictions/injury-severity
              </span>
            </div>
            <Badge variant="success" size="sm">COMMERCIAL CLEARED</Badge>
          </div>
          <CardDescription>
            Predicts UNECE standard three-class physiological trauma severity (SLIGHT, SERIOUS, FATAL).
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <div className="flex justify-between items-center mb-1 text-xs text-slate-400">
              <span>cURL Snippet:</span>
              <button
                onClick={() => copyCode('curl1', curlInjury)}
                className="flex items-center gap-1 hover:text-white cursor-pointer font-mono text-[11px]"
              >
                {copiedId === 'curl1' ? <Check className="h-3 w-3 text-emerald-400" /> : <Copy className="h-3 w-3" />}
                {copiedId === 'curl1' ? 'Copied' : 'Copy'}
              </button>
            </div>
            <pre className="p-3 rounded-lg bg-slate-950 border border-slate-800 font-mono text-xs text-amber-300 overflow-x-auto">
              {curlInjury}
            </pre>
          </div>

          <div>
            <div className="flex justify-between items-center mb-1 text-xs text-slate-400">
              <span>Python Integration:</span>
              <button
                onClick={() => copyCode('py1', pythonInjury)}
                className="flex items-center gap-1 hover:text-white cursor-pointer font-mono text-[11px]"
              >
                {copiedId === 'py1' ? <Check className="h-3 w-3 text-emerald-400" /> : <Copy className="h-3 w-3" />}
                {copiedId === 'py1' ? 'Copied' : 'Copy'}
              </button>
            </div>
            <pre className="p-3 rounded-lg bg-slate-950 border border-slate-800 font-mono text-xs text-sky-300 overflow-x-auto">
              {pythonInjury}
            </pre>
          </div>
        </CardContent>
      </Card>

      {/* Endpoint 2: Annual Exposure Risk */}
      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <span className="px-2 py-0.5 rounded bg-sky-950 text-sky-400 font-mono text-xs font-bold border border-sky-800">
                POST
              </span>
              <span className="font-mono text-sm font-semibold text-slate-100">
                /api/v1/predictions/exposure-risk
              </span>
            </div>
            <Badge variant="success" size="sm">COMMERCIAL CLEARED</Badge>
          </div>
          <CardDescription>
            Predicts expected annual collisions per road segment using traffic exposure (AADF/TMJA) and Poisson regression.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div>
            <div className="flex justify-between items-center mb-1 text-xs text-slate-400">
              <span>cURL Snippet:</span>
              <button
                onClick={() => copyCode('curl2', curlExposure)}
                className="flex items-center gap-1 hover:text-white cursor-pointer font-mono text-[11px]"
              >
                {copiedId === 'curl2' ? <Check className="h-3 w-3 text-emerald-400" /> : <Copy className="h-3 w-3" />}
                {copiedId === 'curl2' ? 'Copied' : 'Copy'}
              </button>
            </div>
            <pre className="p-3 rounded-lg bg-slate-950 border border-slate-800 font-mono text-xs text-amber-300 overflow-x-auto">
              {curlExposure}
            </pre>
          </div>
        </CardContent>
      </Card>

      {/* Status Codes & Licensing Errors */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-semibold">HTTP Status Code Reference</CardTitle>
          <CardDescription>Standardized JSON error schemas across all API endpoints</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="text-[10px] uppercase font-mono text-slate-500 border-b border-slate-800">
                <tr>
                  <th className="pb-2">Code</th>
                  <th className="pb-2">Meaning</th>
                  <th className="pb-2">Scenario</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono text-[11px] text-slate-300">
                <tr>
                  <td className="py-2.5 text-emerald-400 font-bold">200 OK</td>
                  <td className="py-2.5 font-sans">Success</td>
                  <td className="py-2.5 font-sans text-slate-400">Model inference completed and logged to audit table.</td>
                </tr>
                <tr>
                  <td className="py-2.5 text-rose-400 font-bold">400 Bad Request</td>
                  <td className="py-2.5 font-sans">Validation Error</td>
                  <td className="py-2.5 font-sans text-slate-400">Invalid WGS84 coordinate bounds or missing feature payload.</td>
                </tr>
                <tr>
                  <td className="py-2.5 text-rose-400 font-bold">401 Unauthorized</td>
                  <td className="py-2.5 font-sans">Authentication Failed</td>
                  <td className="py-2.5 font-sans text-slate-400">Invalid or revoked X-API-Key header.</td>
                </tr>
                <tr>
                  <td className="py-2.5 text-amber-400 font-bold">403 Forbidden</td>
                  <td className="py-2.5 font-sans">License Enforcement</td>
                  <td className="py-2.5 font-sans text-slate-400">
                    Commercial account attempted to invoke a CC BY-NC-SA 4.0 research endpoint.
                  </td>
                </tr>
                <tr>
                  <td className="py-2.5 text-rose-400 font-bold">429 Rate Limit</td>
                  <td className="py-2.5 font-sans">Quota Exceeded</td>
                  <td className="py-2.5 font-sans text-slate-400">
                    Requests per minute exceeded tenant plan entitlement.
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
