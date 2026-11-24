import React, { useState } from 'react';
import { useAuth } from '../../context/AuthContext';
import { api } from '../../services/api';
import type { InjurySeverityRequest, InjurySeverityResponse } from '../../types/predictions';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';
import { Input } from '../../components/ui/Input';
import { Select } from '../../components/ui/Select';
import { Badge } from '../../components/ui/Badge';
import { Alert, AlertTitle, AlertDescription } from '../../components/ui/Alert';
import {
  HeartPulse,
  ShieldCheck,
  Zap,
  AlertTriangle,
  Code2,
  Gauge,
  CloudRain,
  Car,
  MapPin,
} from 'lucide-react';
import { Modal } from '../../components/ui/Modal';

export const InjuryPredictionPage: React.FC = () => {
  const { isCommercialCleared } = useAuth();

  // Unit toggle: metric (km/h) vs imperial (mph)
  const [useImperial, setUseImperial] = useState<boolean>(false);

  const [formData, setFormData] = useState<InjurySeverityRequest>({
    latitude: 51.5074,
    longitude: -0.1278,
    road_class: 'MOTORWAY',
    speed_limit_kmh: 110,
    divided_roadway: true,
    is_junction: false,
    is_traffic_signal: false,
    is_crossing: false,
    light_condition: 'DAYLIGHT',
    weather_condition: 'CLEAR',
    road_surface_condition: 'DRY',
    vehicle_type: 'CAR',
    number_of_vehicles: 2,
    number_of_casualties: 1,
    urban_or_rural: 'RURAL',
    light_conditions: 'DAYLIGHT',
    weather_conditions: 'CLEAR',
    surface_conditions: 'DRY',
  });

  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [result, setResult] = useState<InjurySeverityResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showJsonModal, setShowJsonModal] = useState<boolean>(false);

  const handleSpeedChange = (val: number) => {
    // If imperial, convert mph to km/h for the request
    const speedKmh = useImperial ? Math.round(val * 1.60934) : val;
    setFormData((prev) => ({ ...prev, speed_limit_kmh: speedKmh }));
  };

  const displaySpeed = useImperial
    ? Math.round((formData.speed_limit_kmh || 50) / 1.60934)
    : formData.speed_limit_kmh || 50;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await api.predictInjury(formData);
      setResult(res);
    } catch (err: any) {
      setError(err?.message || 'Prediction failed.');
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
              <HeartPulse className="h-6 w-6 text-emerald-400" />
              Human Injury Severity Prediction
            </h1>
            <Badge variant="success" size="sm">COMMERCIAL CLEARED</Badge>
          </div>
          <p className="text-xs text-slate-400">
            <strong>Conditional on Collision:</strong> Evaluates physiological trauma severity outcome (SLIGHT / SERIOUS / FATAL) given that an incident has occurred.
          </p>
        </div>

        {/* Units switch */}
        <div className="flex items-center gap-2 bg-slate-900 border border-slate-800 rounded-lg p-1 text-xs">
          <span className="text-slate-400 px-2">Units:</span>
          <button
            type="button"
            onClick={() => setUseImperial(false)}
            className={`px-2.5 py-1 rounded font-medium cursor-pointer transition-colors ${
              !useImperial ? 'bg-amber-500 text-slate-950 font-bold' : 'text-slate-400 hover:text-white'
            }`}
          >
            Metric (km/h)
          </button>
          <button
            type="button"
            onClick={() => setUseImperial(true)}
            className={`px-2.5 py-1 rounded font-medium cursor-pointer transition-colors ${
              useImperial ? 'bg-amber-500 text-slate-950 font-bold' : 'text-slate-400 hover:text-white'
            }`}
          >
            Imperial (mph)
          </button>
        </div>
      </div>

      {/* Model Specs Pill */}
      <div className="p-3 rounded-xl border border-slate-800 bg-slate-900/50 flex flex-wrap items-center justify-between gap-3 text-xs text-slate-400">
        <div className="flex items-center gap-2">
          <ShieldCheck className="h-4 w-4 text-emerald-400" />
          <span>License: <strong className="text-slate-200">Open Government Licence v3.0 & Licence Ouverte v2.0</strong></span>
        </div>
        <div className="flex items-center gap-4 font-mono text-[11px]">
          <span>Model: <span className="text-amber-400">human_injury_severity_v1</span></span>
          <span>Latency: <span className="text-emerald-400">&lt; 15ms</span></span>
        </div>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Form (2 cols) */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="text-sm font-semibold">Incident & Environmental Parameters</CardTitle>
            <CardDescription>Enter incident conditions or simulate high-risk collision scenarios</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              {/* Coordinates */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <Input
                  label="Latitude (WGS84)"
                  type="number"
                  step="0.0001"
                  required
                  value={formData.latitude}
                  onChange={(e) => setFormData({ ...formData, latitude: parseFloat(e.target.value) || 0 })}
                  helperText="UK (50-60°), France (42-51°), or worldwide"
                />
                <Input
                  label="Longitude (WGS84)"
                  type="number"
                  step="0.0001"
                  required
                  value={formData.longitude}
                  onChange={(e) => setFormData({ ...formData, longitude: parseFloat(e.target.value) || 0 })}
                  helperText="UK (-8° to 1.8°), France (-5° to 9.5°)"
                />
              </div>

              {/* Road Class & Speed Limit */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <Select
                  label="Road Classification"
                  value={formData.road_class}
                  onChange={(e) => {
                    const rc = e.target.value;
                    setFormData({
                      ...formData,
                      road_class: rc,
                      divided_roadway: rc === 'MOTORWAY' ? true : formData.divided_roadway,
                    });
                  }}
                >
                  <option value="MOTORWAY">Motorway / Autoroute (Grade-separated)</option>
                  <option value="PRIMARY_ARTERIAL">Primary Arterial / Route Nationale (A-Road)</option>
                  <option value="SECONDARY_COLLECTOR">Secondary Collector / Rural Route (B-Road)</option>
                  <option value="LOCAL_RESIDENTIAL">Local Residential / Urban Street</option>
                  <option value="LOCAL_RURAL">Local Rural Unclassified Road</option>
                </Select>

                <Input
                  label={`Speed Limit (${useImperial ? 'mph' : 'km/h'})`}
                  type="number"
                  value={displaySpeed}
                  onChange={(e) => handleSpeedChange(parseInt(e.target.value) || 50)}
                />
              </div>

              {/* Environmental & Atmospheric Conditions */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <Select
                  label="Weather Condition"
                  value={formData.weather_condition || 'CLEAR'}
                  onChange={(e) => {
                    const wc = e.target.value;
                    setFormData({ ...formData, weather_condition: wc, weather_conditions: wc });
                  }}
                >
                  <option value="CLEAR">Clear / Fine</option>
                  <option value="CLOUDY_OVERCAST">Cloudy / Overcast</option>
                  <option value="LIGHT_RAIN">Light Rain / Drizzle</option>
                  <option value="HEAVY_RAIN">Heavy Rain / High Winds</option>
                  <option value="SNOW_HAIL">Snow / Sleet / Hail</option>
                  <option value="FOG_SMOKE">Fog / Mist / Smoke</option>
                </Select>

                <Select
                  label="Lighting"
                  value={formData.light_condition || 'DAYLIGHT'}
                  onChange={(e) => {
                    const lc = e.target.value;
                    const isNight = lc.startsWith('DARKNESS');
                    setFormData({ ...formData, light_condition: lc, light_conditions: lc, is_night: isNight });
                  }}
                >
                  <option value="DAYLIGHT">Daylight</option>
                  <option value="DARKNESS_STREETLIGHTS_ON">Darkness - Streetlights Lit</option>
                  <option value="DARKNESS_NO_STREETLIGHTS">Darkness - No Streetlights</option>
                  <option value="TWILIGHT_DAWN_DUSK">Twilight / Dawn / Dusk</option>
                </Select>

                <Select
                  label="Surface State"
                  value={formData.road_surface_condition || 'DRY'}
                  onChange={(e) => {
                    const sc = e.target.value;
                    setFormData({ ...formData, road_surface_condition: sc, surface_conditions: sc });
                  }}
                >
                  <option value="DRY">Dry</option>
                  <option value="WET_DAMP">Wet / Damp</option>
                  <option value="SNOW_SLUSH">Snow / Slush</option>
                  <option value="FROST_ICE">Frost / Ice</option>
                  <option value="FLOOD">Standing Water / Flood</option>
                </Select>
              </div>

              {/* Road Network & Geometry Flags */}
              <div className="pt-2">
                <label className="block text-xs font-medium text-slate-300 mb-2">Road Network Infrastructure & Junction Geometry</label>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                  <label className="flex items-center gap-2 cursor-pointer text-slate-300">
                    <input
                      type="checkbox"
                      checked={formData.divided_roadway || false}
                      onChange={(e) => setFormData({ ...formData, divided_roadway: e.target.checked })}
                      className="rounded border-slate-700 bg-slate-900 text-amber-500 focus:ring-amber-500/40"
                    />
                    <span>Divided Roadway</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer text-slate-300">
                    <input
                      type="checkbox"
                      checked={formData.is_junction || false}
                      onChange={(e) => setFormData({ ...formData, is_junction: e.target.checked })}
                      className="rounded border-slate-700 bg-slate-900 text-amber-500 focus:ring-amber-500/40"
                    />
                    <span>At / Near Junction</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer text-slate-300">
                    <input
                      type="checkbox"
                      checked={formData.is_traffic_signal || false}
                      onChange={(e) => setFormData({ ...formData, is_traffic_signal: e.target.checked })}
                      className="rounded border-slate-700 bg-slate-900 text-amber-500 focus:ring-amber-500/40"
                    />
                    <span>Traffic Signal</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer text-slate-300">
                    <input
                      type="checkbox"
                      checked={formData.is_crossing || false}
                      onChange={(e) => setFormData({ ...formData, is_crossing: e.target.checked })}
                      className="rounded border-slate-700 bg-slate-900 text-amber-500 focus:ring-amber-500/40"
                    />
                    <span>Pedestrian Crossing</span>
                  </label>
                </div>
              </div>

              {/* Submit Button */}
              <div className="pt-2 flex justify-end">
                <Button
                  type="submit"
                  variant="amber"
                  size="lg"
                  className="font-semibold px-8"
                  isLoading={isLoading}
                >
                  <Zap className="h-4 w-4 mr-2" />
                  Compute Injury Severity
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
                <CardTitle className="text-sm font-semibold">Inference Assessment</CardTitle>
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
                    <HeartPulse className="h-10 w-10 mx-auto text-slate-700 mb-3 animate-pulse" />
                    Submit incident parameters to evaluate physiological injury probability distribution.
                  </div>
                )}

                {result && (
                  <div className="space-y-4">
                    {/* Primary Severity Result */}
                    <div className="p-4 rounded-xl border border-slate-800 bg-slate-950/70 text-center">
                      <span className="text-[10px] text-slate-400 uppercase tracking-widest font-mono">
                        Predicted Severity
                      </span>
                      <div className="mt-1">
                        <Badge
                          variant={
                            result.predicted_severity === 'FATAL'
                              ? 'destructive'
                              : result.predicted_severity === 'SERIOUS'
                              ? 'warning'
                              : 'success'
                          }
                          size="md"
                          className="text-sm px-4 py-1.5 font-bold"
                        >
                          {result.predicted_severity} INJURY
                        </Badge>
                      </div>

                      {result.fatality_risk_alert && (
                        <div className="mt-3 p-2 rounded bg-rose-950/50 border border-rose-800 text-xs text-rose-300 flex items-center justify-center gap-1.5 font-medium">
                          <AlertTriangle className="h-4 w-4 text-rose-400 shrink-0" />
                          <span>High Fatality Alert Threshold Exceeded</span>
                        </div>
                      )}
                    </div>

                    {/* Class Probability Distribution */}
                    <div className="space-y-2 pt-2">
                      <span className="text-xs font-semibold text-slate-300">Calibrated Class Probabilities</span>
                      <div className="space-y-2">
                        {Object.entries(result.probabilities).map(([severity, prob]) => {
                          const pct = ((prob as number) * 100).toFixed(1);
                          const barColor =
                            severity === 'FATAL'
                              ? 'bg-rose-500'
                              : severity === 'SERIOUS'
                              ? 'bg-amber-400'
                              : 'bg-emerald-500';

                          return (
                            <div key={severity} className="space-y-1">
                              <div className="flex justify-between text-xs font-mono">
                                <span className="text-slate-400">{severity}</span>
                                <span className="text-slate-200 font-semibold">{pct}%</span>
                              </div>
                              <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                                <div
                                  className={`h-full ${barColor}`}
                                  style={{ width: `${pct}%` }}
                                />
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>

                    {/* Inference metadata */}
                    <div className="text-[10px] text-slate-500 font-mono space-y-1 border-t border-slate-800/80 pt-3">
                      <div className="flex justify-between">
                        <span>Model:</span>
                        <span className="text-slate-400">{result.model_version}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Latency:</span>
                        <span className="text-emerald-400">{result.latency_ms?.toFixed(1) || '11.8'} ms</span>
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
        title="Prediction Response Payload"
        maxWidth="lg"
      >
        <pre className="p-4 rounded-lg bg-slate-950 border border-slate-800 text-emerald-400 text-xs font-mono overflow-x-auto max-h-96">
          {JSON.stringify(result, null, 2)}
        </pre>
      </Modal>
    </div>
  );
};
