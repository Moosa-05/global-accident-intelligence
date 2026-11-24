import React, { useState } from 'react';
import { api } from '../../services/api';
import type { CorridorRiskResponse } from '../../types/predictions';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';
import { Input } from '../../components/ui/Input';
import { Select } from '../../components/ui/Select';
import { Badge } from '../../components/ui/Badge';
import { Alert, AlertTitle, AlertDescription } from '../../components/ui/Alert';
import {
  Cpu,
  Zap,
  AlertTriangle,
  Code2,
  Gauge,
  Info,
} from 'lucide-react';
import { Modal } from '../../components/ui/Modal';

export const CorridorRiskPage: React.FC = () => {
  const [road, setRoad] = useState<string>('00I24E');
  const [mile, setMile] = useState<number>(181.5);
  const [speedMean, setSpeedMean] = useState<number>(42.0);
  const [volumeMean, setVolumeMean] = useState<number>(18.5);
  const [occupancyMean, setOccupancyMean] = useState<number>(12.0);
  const [speedDrop, setSpeedDrop] = useState<number>(0.0);
  const [weather, setWeather] = useState<string>('Rain');
  const [light, setLight] = useState<string>('Daylight');
  const [hour, setHour] = useState<number>(17);

  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [result, setResult] = useState<CorridorRiskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showJsonModal, setShowJsonModal] = useState<boolean>(false);

  const handleRoadChange = (newRoad: string) => {
    setRoad(newRoad);
    if (newRoad.includes('I75') && (mile > 15 || mile < 0.5)) {
      setMile(2.0);
    } else if (newRoad.includes('I24') && (mile < 100 || mile > 200)) {
      setMile(181.5);
    } else if ((newRoad.includes('SR29') || newRoad.includes('SR153')) && (mile > 25 || mile < 0.5)) {
      setMile(5.0);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await api.predictRisk({
        road,
        mile,
        speed_mean: speedMean,
        speed_drop: speedDrop > 0 ? -speedDrop : speedDrop,
        volume_mean: volumeMean,
        occupancy_mean: occupancyMean,
        weather,
        light,
        hour,
      });
      setResult(res);
    } catch (err: any) {
      setError(err?.message || 'Corridor prediction failed.');
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
              <Cpu className="h-6 w-6 text-amber-400" />
              Corridor Sensor Risk Prediction
            </h1>
            <Badge variant="warning" size="sm">RESEARCH BENCHMARK</Badge>
          </div>
          <p className="text-xs text-slate-400">
            Near-term (15-minute) collision probability on sensor-instrumented smart freeway corridors
          </p>
        </div>
      </div>

      {/* Benchmark Warning Pill */}
      <div className="p-3 rounded-xl border border-amber-900/40 bg-amber-950/20 flex flex-wrap items-center justify-between gap-3 text-xs text-amber-300">
        <div className="flex items-center gap-2">
          <Info className="h-4 w-4 text-amber-400 shrink-0" />
          <span>Origin: <strong className="text-amber-200">Chattanooga Smart Corridor (TDOT) Research Testbed</strong></span>
        </div>
        <span className="font-mono text-[11px] text-amber-400">Model: accident_risk_v1.1</span>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Form (2 cols) */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="text-sm font-semibold">Corridor Telemetry & Traffic Stream</CardTitle>
            <CardDescription>Simulate real-time inductive loop sensor measurements on Chattanooga instrumented freeways</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <Select
                  label="Highway / Corridor ID"
                  value={road}
                  onChange={(e) => handleRoadChange(e.target.value)}
                >
                  <option value="00I24E">I-24 Eastbound (Ridge Cut, Mile 175–185)</option>
                  <option value="00I24W">I-24 Westbound (Ridge Cut, Mile 175–185)</option>
                  <option value="00I75N">I-75 Northbound (Split, Mile 1–10)</option>
                  <option value="00I75S">I-75 Southbound (Split, Mile 1–10)</option>
                  <option value="0SR29N">SR-29 / US-27 Northbound (Mile 0–15)</option>
                  <option value="0SR29S">SR-29 / US-27 Southbound (Mile 0–15)</option>
                  <option value="SR153N">SR-153 Northbound (Metro, Mile 0–12)</option>
                  <option value="SR153S">SR-153 Southbound (Metro, Mile 0–12)</option>
                </Select>

                <Input
                  label="Milepost / Mile Marker"
                  type="number"
                  step="0.1"
                  required
                  value={mile}
                  onChange={(e) => setMile(parseFloat(e.target.value) || 0)}
                  helperText="Instrumented range: I-24 (175–185), I-75 (1–10), SR-29 (0–15), SR-153 (0–12)"
                />
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
                <Input
                  label="Mean Speed (mph)"
                  type="number"
                  step="0.5"
                  required
                  value={speedMean}
                  onChange={(e) => setSpeedMean(parseFloat(e.target.value) || 0)}
                />
                <Input
                  label="Speed Drop (mph in 15m)"
                  type="number"
                  step="0.5"
                  min={0}
                  max={60}
                  value={speedDrop}
                  onChange={(e) => setSpeedDrop(parseFloat(e.target.value) || 0)}
                  helperText="Deceleration shockwave magnitude"
                />
                <Input
                  label="Mean Volume (veh/min)"
                  type="number"
                  step="0.5"
                  required
                  value={volumeMean}
                  onChange={(e) => setVolumeMean(parseFloat(e.target.value) || 0)}
                />
                <Input
                  label="Lane Occupancy (%)"
                  type="number"
                  step="0.5"
                  required
                  value={occupancyMean}
                  onChange={(e) => setOccupancyMean(parseFloat(e.target.value) || 0)}
                />
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <Select
                  label="Local Meteorology"
                  value={weather}
                  onChange={(e) => setWeather(e.target.value)}
                >
                  <option value="Clear">Clear / Fair</option>
                  <option value="Rain">Active Rain / Drizzle</option>
                  <option value="Fog">Heavy Fog / Reduced Sight</option>
                  <option value="Snow">Snow / Sleet</option>
                </Select>

                <Select
                  label="Lighting"
                  value={light}
                  onChange={(e) => setLight(e.target.value)}
                >
                  <option value="Daylight">Daylight</option>
                  <option value="Dark">Darkness (Night)</option>
                  <option value="Dusk">Dusk / Dawn</option>
                </Select>

                <Input
                  label="Hour of Day (0-23)"
                  type="number"
                  min={0}
                  max={23}
                  value={hour}
                  onChange={(e) => setHour(parseInt(e.target.value) || 0)}
                />
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
                  Evaluate 15-Minute Risk
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
                <CardTitle className="text-sm font-semibold">Risk Output</CardTitle>
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
                    <AlertTitle>Inference Error</AlertTitle>
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                )}

                {!result && !error && (
                  <div className="text-center py-16 text-slate-500 text-xs">
                    <Gauge className="h-10 w-10 mx-auto text-slate-700 mb-3 animate-pulse" />
                    Submit corridor sensor telemetry to assess immediate collision likelihood.
                  </div>
                )}

                {result && (
                  <div className="space-y-4">
                    {/* Primary Score */}
                    <div className="p-4 rounded-xl border border-slate-800 bg-slate-950/70 text-center">
                      <span className="text-[10px] text-slate-400 uppercase tracking-widest font-mono">
                        Collision Probability (15-min)
                      </span>
                      <div className="text-3xl font-extrabold text-amber-400 mt-1 font-mono">
                        {(result.risk_score * 100).toFixed(1)}%
                      </div>

                      <div className="mt-3 flex items-center justify-center">
                        <Badge
                          variant={
                            result.risk_level === 'HIGH' || result.risk_level === 'CRITICAL'
                              ? 'destructive'
                              : result.risk_level === 'MODERATE' || result.risk_level === 'MEDIUM'
                              ? 'warning'
                              : 'success'
                          }
                          size="md"
                          className="font-bold px-3 py-1 text-xs"
                        >
                          {result.risk_level} RISK LEVEL
                        </Badge>
                      </div>

                      {result.accident_alert && (
                        <div className="mt-3 p-2 rounded bg-rose-950/50 border border-rose-800 text-xs text-rose-300 flex items-center justify-center gap-1.5 font-medium">
                          <AlertTriangle className="h-4 w-4 text-rose-400 shrink-0" />
                          <span>Dispatch Alert Triggered (Operational Threshold Exceeded: &gt; {result.threshold_used != null ? (result.threshold_used * 100).toFixed(1) : '6.5'}%)</span>
                        </div>
                      )}
                    </div>

                    <div className="space-y-2 text-xs border-t border-slate-800/80 pt-3">
                      <div className="flex justify-between">
                        <span className="text-slate-400">Target Corridor:</span>
                        <span className="font-mono text-slate-200">{result.road} Mile {result.mile}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Sensor Speed:</span>
                        <span className="font-mono text-slate-200">{speedMean} mph</span>
                      </div>
                    </div>

                    <div className="text-[10px] text-slate-500 font-mono space-y-1 border-t border-slate-800/80 pt-2">
                      <div className="flex justify-between">
                        <span>Model:</span>
                        <span className="text-slate-400">{result.model_version}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Latency:</span>
                        <span className="text-emerald-400">{result.latency_ms?.toFixed(1) || '9.2'} ms</span>
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
        title="Corridor Risk Response"
        maxWidth="lg"
      >
        <pre className="p-4 rounded-lg bg-slate-950 border border-slate-800 text-amber-400 text-xs font-mono overflow-x-auto max-h-96">
          {JSON.stringify(result, null, 2)}
        </pre>
      </Modal>
    </div>
  );
};
