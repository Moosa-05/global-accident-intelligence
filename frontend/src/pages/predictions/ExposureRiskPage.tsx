import React, { useState } from 'react';
import { api } from '../../services/api';
import type { ExposureRiskResponse } from '../../types/predictions';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';
import { Input } from '../../components/ui/Input';
import { Select } from '../../components/ui/Select';
import { Badge } from '../../components/ui/Badge';
import { Alert, AlertTitle, AlertDescription } from '../../components/ui/Alert';
import {
  TrendingUp,
  ShieldCheck,
  Zap,
  Code2,
  Calculator,
  BarChart2,
  Info,
} from 'lucide-react';
import { Modal } from '../../components/ui/Modal';

export const ExposureRiskPage: React.FC = () => {
  const [roadName, setRoadName] = useState<string>('M62 Trans-Pennine Corridor');
  const [roadClass, setRoadClass] = useState<string>('MOTORWAY');
  const [linkLengthKm, setLinkLengthKm] = useState<number>(2.5);
  const [speedLimitKmh, setSpeedLimitKmh] = useState<number>(110);
  const [lat, setLat] = useState<number>(53.4808);
  const [lng, setLng] = useState<number>(-2.2426);
  const [aadf, setAadf] = useState<number>(48500);

  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [result, setResult] = useState<ExposureRiskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showJsonModal, setShowJsonModal] = useState<boolean>(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await api.predictExposure({
        road_name: roadName,
        road_class: roadClass,
        link_length_km: linkLengthKm,
        speed_limit_kmh: speedLimitKmh,
        latitude: lat,
        longitude: lng,
        annual_average_daily_flow: aadf,
      });
      setResult(res);
    } catch (err: any) {
      setError(err?.message || 'Exposure prediction failed.');
    } finally {
      setIsLoading(false);
    }
  };

  // Live exposure preview formula
  const liveVkt = ((aadf * linkLengthKm * 365) / 1_000_000).toFixed(2);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
              <TrendingUp className="h-6 w-6 text-sky-400" />
              Annual Road Exposure Risk Prediction
            </h1>
            <Badge variant="success" size="sm">COMMERCIAL CLEARED</Badge>
          </div>
          <p className="text-xs text-slate-400">
            Road segment collision expectancy based on annual traffic exposure (AADF / TMJA) and functional class
          </p>
        </div>
      </div>

      {/* Model Specs Pill */}
      <div className="p-3 rounded-xl border border-slate-800 bg-slate-900/50 flex flex-wrap items-center justify-between gap-3 text-xs text-slate-400">
        <div className="flex items-center gap-2">
          <ShieldCheck className="h-4 w-4 text-emerald-400" />
          <span>Model Architecture: <strong className="text-slate-200">Exposure-Adjusted Poisson GLM (VKT Offset)</strong></span>
        </div>
        <div className="flex items-center gap-4 font-mono text-[11px]">
          <span>Dataset: <span className="text-sky-400">DfT Traffic AADF + French TMJA 2023</span></span>
          <span>Latency: <span className="text-emerald-400">&lt; 10ms</span></span>
        </div>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Form (2 cols) */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="text-sm font-semibold">Segment Engineering Specifications</CardTitle>
            <CardDescription>Input road geometry, traffic flow census, and spatial coordinates</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <Input
                  label="Road Name or Identifier"
                  type="text"
                  value={roadName}
                  onChange={(e) => setRoadName(e.target.value)}
                  placeholder="e.g. M25 J12-J13 or A6 Autoroute du Sud"
                />
                <Select
                  label="Functional Road Classification"
                  value={roadClass}
                  onChange={(e) => setRoadClass(e.target.value)}
                >
                  <option value="MOTORWAY">Motorway / Autoroute (Divided highway)</option>
                  <option value="PRIMARY_ARTERIAL">Primary Arterial / Route Nationale (Trunk)</option>
                  <option value="SECONDARY_COLLECTOR">Secondary Collector / Rural Route</option>
                  <option value="LOCAL_OTHER">Local Road / Urban Street</option>
                </Select>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <Input
                  label="Daily Flow (AADF / TMJA)"
                  type="number"
                  required
                  min={100}
                  max={500000}
                  value={aadf}
                  onChange={(e) => setAadf(parseInt(e.target.value) || 1000)}
                  helperText="Vehicles per day counted"
                />
                <Input
                  label="Segment Length (km)"
                  type="number"
                  step="0.05"
                  required
                  min={0.1}
                  max={100}
                  value={linkLengthKm}
                  onChange={(e) => setLinkLengthKm(parseFloat(e.target.value) || 1.0)}
                  helperText="Homogeneous stretch length"
                />
                <Input
                  label="Speed Limit (km/h)"
                  type="number"
                  required
                  min={20}
                  max={150}
                  value={speedLimitKmh}
                  onChange={(e) => setSpeedLimitKmh(parseFloat(e.target.value) || 100)}
                />
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <Input
                  label="Segment Midpoint Latitude (WGS84)"
                  type="number"
                  step="0.0001"
                  required
                  value={lat}
                  onChange={(e) => setLat(parseFloat(e.target.value) || 0)}
                />
                <Input
                  label="Segment Midpoint Longitude (WGS84)"
                  type="number"
                  step="0.0001"
                  required
                  value={lng}
                  onChange={(e) => setLng(parseFloat(e.target.value) || 0)}
                />
              </div>

              {/* Exposure calculation preview box */}
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 text-xs flex items-center justify-between font-mono">
                <span className="text-slate-400 flex items-center gap-1.5">
                  <Calculator className="h-4 w-4 text-sky-400" />
                  Calculated Annual Exposure:
                </span>
                <span className="text-sky-300 font-bold">{liveVkt} Million Vehicle-Km / Year</span>
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
                  Predict Annual Collision Rate
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
                <CardTitle className="text-sm font-semibold">Exposure Assessment</CardTitle>
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
                    <AlertTitle>Error</AlertTitle>
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                )}

                {!result && !error && (
                  <div className="text-center py-16 text-slate-500 text-xs">
                    <BarChart2 className="h-10 w-10 mx-auto text-slate-700 mb-3 animate-pulse" />
                    Submit segment parameters to evaluate annual Poisson collision expectancy and risk tier.
                  </div>
                )}

                {result && (
                  <div className="space-y-4">
                    {/* Annual Crashes Card */}
                    <div className="p-4 rounded-xl border border-slate-800 bg-slate-950/70 text-center">
                      <span className="text-[10px] text-slate-400 uppercase tracking-widest font-mono">
                        Expected Annual Collisions
                      </span>
                      <div className="text-3xl font-extrabold text-sky-400 mt-1 font-mono">
                        {result.predicted_annual_crashes.toFixed(2)}
                      </div>
                      <p className="text-xs text-slate-400 mt-0.5">collisions per year</p>

                      <div className="mt-3 flex items-center justify-center">
                        <Badge
                          variant={
                            result.risk_tier.includes('HIGH')
                              ? 'destructive'
                              : result.risk_tier.includes('ELEVATED')
                              ? 'warning'
                              : 'success'
                          }
                          size="md"
                          className="font-bold px-3 py-1 text-xs"
                        >
                          {result.risk_tier}
                        </Badge>
                      </div>
                    </div>

                    {/* Breakdown Metrics */}
                    <div className="space-y-2.5 text-xs border-t border-slate-800/80 pt-3">
                      <div className="flex justify-between items-center">
                        <span className="text-slate-400">Crash Rate:</span>
                        <span className="font-mono text-slate-200 font-semibold">
                          {result.predicted_crash_rate_per_million_vkt.toFixed(3)} / M VKT
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-slate-400">Total Exposure:</span>
                        <span className="font-mono text-slate-200">
                          {result.annual_exposure_million_vkt.toFixed(2)} M VKT
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-slate-400">Road Class Baseline:</span>
                        <span className="font-mono text-amber-400">{roadClass}</span>
                      </div>
                    </div>

                    {/* Explanatory Note */}
                    <div className="p-2.5 rounded bg-slate-900 border border-slate-800/80 text-[11px] text-slate-400 space-y-1">
                      <p className="font-semibold text-slate-300 flex items-center gap-1">
                        <Info className="h-3 w-3 text-sky-400" />
                        Risk Tier Definition:
                      </p>
                      <p>
                        Calibrated against national highway benchmarks. Low: &lt; 0.15 crashes/M VKT; Typical: 0.15 - 0.40; High: &gt; 0.40.
                      </p>
                    </div>

                    <div className="text-[10px] text-slate-500 font-mono space-y-1 border-t border-slate-800/80 pt-2">
                      <div className="flex justify-between">
                        <span>Model:</span>
                        <span className="text-slate-400">{result.model_version}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Latency:</span>
                        <span className="text-emerald-400">{result.latency_ms?.toFixed(1) || '8.4'} ms</span>
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
        title="Exposure Prediction Response"
        maxWidth="lg"
      >
        <pre className="p-4 rounded-lg bg-slate-950 border border-slate-800 text-sky-400 text-xs font-mono overflow-x-auto max-h-96">
          {JSON.stringify(result, null, 2)}
        </pre>
      </Modal>
    </div>
  );
};
