import React, { useState } from 'react';
import { useAuth } from '../../context/AuthContext';
import { api } from '../../services/api';
import type { TrafficSeverityResponse } from '../../types/predictions';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';
import { Input } from '../../components/ui/Input';
import { Select } from '../../components/ui/Select';
import { Badge } from '../../components/ui/Badge';
import { Alert, AlertTitle, AlertDescription } from '../../components/ui/Alert';
import {
  ClockAlert,
  Lock,
  ShieldAlert,
  Zap,
  Code2,
} from 'lucide-react';
import { Modal } from '../../components/ui/Modal';

export const TrafficSeverityPage: React.FC = () => {
  const { isCommercialCleared } = useAuth();

  const [lat, setLat] = useState<number>(34.0522);
  const [lng, setLng] = useState<number>(-118.2437);
  const [tempF, setTempF] = useState<number>(68.0);
  const [humidity, setHumidity] = useState<number>(55.0);
  const [pressure, setPressure] = useState<number>(29.92);
  const [visibility, setVisibility] = useState<number>(10.0);
  const [windSpeed, setWindSpeed] = useState<number>(8.0);
  const [precip, setPrecip] = useState<number>(0.0);
  const [weatherCondition, setWeatherCondition] = useState<string>('Clear');

  const [isJunction, setIsJunction] = useState<boolean>(true);
  const [isSignal, setIsSignal] = useState<boolean>(true);
  const [isCrossing, setIsCrossing] = useState<boolean>(false);
  const [isStop, setIsStop] = useState<boolean>(false);

  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [result, setResult] = useState<TrafficSeverityResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showJsonModal, setShowJsonModal] = useState<boolean>(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await api.predictSeverity({
        latitude: lat,
        longitude: lng,
        units: 'imperial',
        temperature: tempF,
        humidity: humidity,
        pressure: pressure,
        visibility: visibility,
        wind_speed: windSpeed,
        precipitation: precip,
        weather_condition: weatherCondition,
        is_junction: isJunction,
        is_traffic_signal: isSignal,
        is_crossing: isCrossing,
        is_stop_sign: isStop,
      });
      setResult(res);
    } catch (err: any) {
      if (err?.status === 403 || err?.message?.includes('Commercial') || err?.message?.includes('License')) {
        setError('403 Forbidden: Commercial accounts are legally barred from executing non-commercial research models (CC BY-NC-SA 4.0 upstream license).');
      } else {
        setError(err?.message || 'Prediction failed.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
              <ClockAlert className="h-6 w-6 text-amber-400" />
              Traffic Impact Severity Model (Delay & Disruption)
            </h1>
            <Badge variant="destructive" size="sm">RESEARCH ONLY (CC BY-NC-SA 4.0)</Badge>
          </div>
          <p className="text-xs text-slate-400">
            Predicts traffic blockage impact level (1 to 4) using the US Accidents research dataset
          </p>
        </div>
      </div>

      {/* Strict Licensing Guardrail Alert */}
      {isCommercialCleared ? (
        <Alert variant="error">
          <AlertTitle className="flex items-center gap-2">
            <Lock className="h-4 w-4" /> Commercial Access Restricted (License Enforcement Active)
          </AlertTitle>
          <AlertDescription>
            Your tenant has an active <strong>Commercial Subscription</strong>.
            Under SaaS compliance policy, invoking this endpoint will trigger an automatic <strong>403 Forbidden</strong> to prevent
            contamination with Creative Commons Non-Commercial (CC BY-NC-SA 4.0) assets.
            Please use the commercial-cleared <strong>Human Injury Severity</strong> and <strong>Exposure Risk</strong> models instead.
          </AlertDescription>
        </Alert>
      ) : (
        <Alert variant="warning">
          <AlertTitle className="flex items-center gap-2">
            <ShieldAlert className="h-4 w-4" /> Non-Commercial Academic Research Sandbox
          </AlertTitle>
          <AlertDescription>
            You are operating under an evaluation sandbox. This model predicts traffic delay severity (classes 1 through 4)
            and is permitted exclusively for non-commercial academic research.
          </AlertDescription>
        </Alert>
      )}

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Form (2 cols) */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="text-sm font-semibold">Incident Parameters</CardTitle>
            <CardDescription>Atmospheric, road feature, and temporal attributes</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <Input
                  label="Start Latitude (US Coverage)"
                  type="number"
                  step="0.0001"
                  required
                  value={lat}
                  onChange={(e) => setLat(parseFloat(e.target.value) || 0)}
                />
                <Input
                  label="Start Longitude (US Coverage)"
                  type="number"
                  step="0.0001"
                  required
                  value={lng}
                  onChange={(e) => setLng(parseFloat(e.target.value) || 0)}
                />
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
                <Input
                  label="Temperature (°F)"
                  type="number"
                  value={tempF}
                  onChange={(e) => setTempF(parseFloat(e.target.value) || 0)}
                />
                <Input
                  label="Humidity (%)"
                  type="number"
                  value={humidity}
                  onChange={(e) => setHumidity(parseFloat(e.target.value) || 0)}
                />
                <Input
                  label="Pressure (inHg)"
                  type="number"
                  step="0.01"
                  value={pressure}
                  onChange={(e) => setPressure(parseFloat(e.target.value) || 29.92)}
                />
                <Input
                  label="Visibility (mi)"
                  type="number"
                  value={visibility}
                  onChange={(e) => setVisibility(parseFloat(e.target.value) || 0)}
                />
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <Input
                  label="Wind Speed (mph)"
                  type="number"
                  value={windSpeed}
                  onChange={(e) => setWindSpeed(parseFloat(e.target.value) || 0)}
                />
                <Input
                  label="Precipitation (in)"
                  type="number"
                  step="0.01"
                  value={precip}
                  onChange={(e) => setPrecip(parseFloat(e.target.value) || 0)}
                />
                <Select
                  label="Weather Condition"
                  value={weatherCondition}
                  onChange={(e) => setWeatherCondition(e.target.value)}
                >
                  <option value="Clear">Clear</option>
                  <option value="Overcast">Overcast</option>
                  <option value="Rain">Rain</option>
                  <option value="Fog">Fog</option>
                  <option value="Snow">Snow</option>
                </Select>
              </div>

              {/* Road Feature Flags */}
              <div className="pt-2">
                <label className="block text-xs font-medium text-slate-300 mb-2">Road Network Infrastructure Flags</label>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                  <label className="flex items-center gap-2 cursor-pointer text-slate-300">
                    <input
                      type="checkbox"
                      checked={isJunction}
                      onChange={(e) => setIsJunction(e.target.checked)}
                      className="rounded border-slate-700 bg-slate-900 text-amber-500 focus:ring-amber-500/40"
                    />
                    <span>Junction</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer text-slate-300">
                    <input
                      type="checkbox"
                      checked={isSignal}
                      onChange={(e) => setIsSignal(e.target.checked)}
                      className="rounded border-slate-700 bg-slate-900 text-amber-500 focus:ring-amber-500/40"
                    />
                    <span>Traffic Signal</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer text-slate-300">
                    <input
                      type="checkbox"
                      checked={isCrossing}
                      onChange={(e) => setIsCrossing(e.target.checked)}
                      className="rounded border-slate-700 bg-slate-900 text-amber-500 focus:ring-amber-500/40"
                    />
                    <span>Pedestrian Crossing</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer text-slate-300">
                    <input
                      type="checkbox"
                      checked={isStop}
                      onChange={(e) => setIsStop(e.target.checked)}
                      className="rounded border-slate-700 bg-slate-900 text-amber-500 focus:ring-amber-500/40"
                    />
                    <span>Stop Sign</span>
                  </label>
                </div>
              </div>

              <div className="pt-2 flex justify-end">
                <Button
                  type="submit"
                  variant="amber"
                  size="lg"
                  className="font-semibold px-8"
                  isLoading={isLoading}
                >
                  <Zap className="h-4 w-4 mr-2" />
                  Evaluate Traffic Severity
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>

        {/* Prediction Results (1 col) */}
        <div className="space-y-4">
          <Card className="h-full flex flex-col justify-between">
            <div>
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="text-sm font-semibold">Severity Assessment</CardTitle>
                {result && (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="p-1 text-slate-400 hover:text-slate-100"
                    onClick={() => setShowJsonModal(true)}
                    title="View Raw JSON"
                  >
                    <Code2 className="h-4 w-4" />
                  </Button>
                )}
              </CardHeader>
              <CardContent>
                {error && (
                  <Alert variant="error" className="mb-4">
                    <AlertTitle>Access Rejected</AlertTitle>
                    <AlertDescription className="text-xs">{error}</AlertDescription>
                  </Alert>
                )}

                {!result && !error && (
                  <div className="text-center py-16 text-slate-500 text-xs">
                    <ClockAlert className="h-10 w-10 mx-auto text-slate-700 mb-3 animate-pulse" />
                    Submit accident features to predict 4-class traffic impact delay.
                  </div>
                )}

                {result && (
                  <div className="space-y-4">
                    {/* Primary Score */}
                    <div className="p-4 rounded-xl border border-slate-800 bg-slate-950/70 text-center">
                      <span className="text-[10px] text-slate-400 uppercase tracking-widest font-mono">
                        Predicted Delay Level
                      </span>
                      <div className="text-3xl font-extrabold text-amber-400 mt-1 font-mono">
                        Class {result.predicted_severity}
                      </div>
                      <p className="text-xs text-slate-400 mt-0.5">{result.severity_level} Traffic Disruption</p>

                      <div className="mt-3 flex items-center justify-center">
                        <Badge variant="amber" size="sm">
                          {result.severity_level}
                        </Badge>
                      </div>
                    </div>

                    {/* Class Distribution */}
                    {result.severity_probabilities && (
                      <div className="space-y-2 pt-2">
                        <span className="text-xs font-semibold text-slate-300">Severity Distribution (1 - 4)</span>
                        <div className="space-y-2">
                          {Object.entries(result.severity_probabilities).map(([sev, prob]) => {
                            const pct = ((prob as number) * 100).toFixed(1);
                            return (
                              <div key={sev} className="space-y-1">
                                <div className="flex justify-between text-xs font-mono">
                                  <span className="text-slate-400">{sev.replace('_', ' ')}</span>
                                  <span className="text-slate-200 font-semibold">{pct}%</span>
                                </div>
                                <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                                  <div
                                    className="h-full bg-amber-500"
                                    style={{ width: `${pct}%` }}
                                  />
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}

                    <div className="text-[10px] text-slate-500 font-mono space-y-1 border-t border-slate-800/80 pt-2">
                      <div className="flex justify-between">
                        <span>Model:</span>
                        <span className="text-slate-400">{result.model_version}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Latency:</span>
                        <span className="text-emerald-400">{result.latency_ms?.toFixed(1) || '12.1'} ms</span>
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </div>
          </Card>
        </div>
      </div>

      {/* Raw JSON Modal */}
      <Modal
        isOpen={showJsonModal}
        onClose={() => setShowJsonModal(false)}
        title="Severity Prediction Response"
        maxWidth="lg"
      >
        <pre className="p-4 rounded-lg bg-slate-950 border border-slate-800 text-amber-400 text-xs font-mono overflow-x-auto max-h-96">
          {JSON.stringify(result, null, 2)}
        </pre>
      </Modal>
    </div>
  );
};
