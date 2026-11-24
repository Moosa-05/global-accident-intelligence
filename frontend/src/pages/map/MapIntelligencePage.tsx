import React, { useEffect, useRef, useState } from 'react';
import * as maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { api } from '../../services/api';
import { Card, CardHeader, CardTitle, CardContent } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';
import { Input } from '../../components/ui/Input';
import { Select } from '../../components/ui/Select';
import { Badge } from '../../components/ui/Badge';
import { Alert, AlertTitle, AlertDescription } from '../../components/ui/Alert';
import {
  Globe,
  Crosshair,
  Zap,
  AlertTriangle,
  HeartPulse,
  TrendingUp,
  Sparkles,
} from 'lucide-react';
import type { InjurySeverityResponse, ExposureRiskResponse } from '../../types/predictions';

interface LocationPreset {
  name: string;
  lat: number;
  lng: number;
  country: string;
  isOod: boolean;
}

const PRESETS: LocationPreset[] = [
  { name: 'London (UK STATS19 Coverage)', lat: 51.5074, lng: -0.1278, country: 'GB', isOod: false },
  { name: 'Paris (France BAAC Coverage)', lat: 48.8566, lng: 2.3522, country: 'FR', isOod: false },
  { name: 'Chattanooga (US Smart Corridor)', lat: 35.0456, lng: -85.3097, country: 'US', isOod: false },
  { name: 'Manchester (M62 Motorway, UK)', lat: 53.4808, lng: -2.2426, country: 'GB', isOod: false },
  { name: 'Lyon (A6 Autoroute, France)', lat: 45.7640, lng: 4.8357, country: 'FR', isOod: false },
  { name: 'Tokyo (Out-of-Distribution Prior)', lat: 35.6762, lng: 139.6503, country: 'JP', isOod: true },
  { name: 'Berlin (Out-of-Distribution Prior)', lat: 52.5200, lng: 13.4050, country: 'DE', isOod: true },
];

export const MapIntelligencePage: React.FC = () => {
  const mapContainer = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const markerRef = useRef<maplibregl.Marker | null>(null);

  const [lat, setLat] = useState<number>(51.5074);
  const [lng, setLng] = useState<number>(-0.1278);
  const [inferenceMode, setInferenceMode] = useState<'injury' | 'exposure'>('injury');
  const [roadClass, setRoadClass] = useState<string>('MOTORWAY');
  const [speedLimitKmh, setSpeedLimitKmh] = useState<number>(100);
  const [aadf, setAadf] = useState<number>(35000);
  const [segmentLengthKm, setSegmentLengthKm] = useState<number>(1.2);

  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [injuryResult, setInjuryResult] = useState<InjurySeverityResponse | null>(null);
  const [exposureResult, setExposureResult] = useState<ExposureRiskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Check Out-of-Distribution (OOD): UK (lat 49.8-60.9, lng -8.5 to 1.8), France (lat 41.3-51.1, lng -5.2 to 9.6), US (lat 24.3-49.4, lng -125 to -66.9)
  const isOod = !(
    (lat >= 49.8 && lat <= 60.9 && lng >= -8.5 && lng <= 1.8) || // UK
    (lat >= 41.3 && lat <= 51.1 && lng >= -5.2 && lng <= 9.6) || // France
    (lat >= 24.3 && lat <= 49.4 && lng >= -125.0 && lng <= -66.9) // US
  );

  // Initialize MapLibre
  useEffect(() => {
    if (!mapContainer.current || mapRef.current) return;

    // Use free CARTO Dark Matter raster tiles (completely open, no token required)
    const map = new maplibregl.Map({
      container: mapContainer.current,
      style: {
        version: 8,
        sources: {
          'carto-dark': {
            type: 'raster',
            tiles: [
              'https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png',
              'https://b.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png',
              'https://c.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png',
            ],
            tileSize: 256,
            attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
          },
        },
        layers: [
          {
            id: 'carto-dark-layer',
            type: 'raster',
            source: 'carto-dark',
            minzoom: 0,
            maxzoom: 19,
          },
        ],
      },
      center: [lng, lat],
      zoom: 6,
    });

    // Add navigation controls
    map.addControl(new maplibregl.NavigationControl(), 'top-right');

    // Add draggable marker
    const marker = new maplibregl.Marker({ color: '#f59e0b', draggable: true })
      .setLngLat([lng, lat])
      .addTo(map);

    marker.on('dragend', () => {
      const pos = marker.getLngLat();
      setLat(parseFloat(pos.lat.toFixed(5)));
      setLng(parseFloat(pos.lng.toFixed(5)));
    });

    map.on('click', (e: any) => {
      marker.setLngLat(e.lngLat);
      setLat(parseFloat(e.lngLat.lat.toFixed(5)));
      setLng(parseFloat(e.lngLat.lng.toFixed(5)));
    });

    mapRef.current = map;
    markerRef.current = marker;

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  const handlePresetSelect = (preset: LocationPreset) => {
    setLat(preset.lat);
    setLng(preset.lng);
    if (mapRef.current && markerRef.current) {
      markerRef.current.setLngLat([preset.lng, preset.lat]);
      mapRef.current.flyTo({ center: [preset.lng, preset.lat], zoom: 10 });
    }
  };

  const handleRunInference = async () => {
    setIsLoading(true);
    setError(null);
    setInjuryResult(null);
    setExposureResult(null);

    try {
      if (inferenceMode === 'injury') {
        const res = await api.predictInjury({
          latitude: lat,
          longitude: lng,
          road_class: roadClass,
          speed_limit_kmh: speedLimitKmh,
          vehicle_type: 'CAR',
          number_of_vehicles: 2,
          number_of_casualties: 1,
          urban_or_rural: 'RURAL',
          light_conditions: 'DAYLIGHT',
          weather_conditions: 'FINE_NO_HIGH_WINDS',
          surface_conditions: 'DRY',
        });
        setInjuryResult(res);
      } else {
        const res = await api.predictExposure({
          latitude: lat,
          longitude: lng,
          road_class: roadClass,
          speed_limit_kmh: speedLimitKmh,
          link_length_km: segmentLengthKm,
          annual_average_daily_flow: aadf,
        });
        setExposureResult(res);
      }
    } catch (err: any) {
      setError(err?.message || 'Prediction failed. Check backend connection.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-2">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
            <Globe className="h-5 w-5 text-amber-400" />
            Global Geographic Intelligence Map
          </h1>
          <p className="text-xs text-slate-400">
            Interactive multi-country accident prediction across worldwide WGS84 coordinates
          </p>
        </div>

        {/* Preset quick jump */}
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-xs text-slate-400 mr-1">Presets:</span>
          {PRESETS.slice(0, 4).map((p) => (
            <button
              key={p.name}
              onClick={() => handlePresetSelect(p)}
              className="text-[11px] px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 cursor-pointer transition-colors"
            >
              {p.name.split(' ')[0]} ({p.country})
            </button>
          ))}
        </div>
      </div>

      {/* Main Grid: Map + Controls */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Map Canvas (2 cols) */}
        <div className="lg:col-span-2 relative rounded-xl border border-slate-800 overflow-hidden shadow-2xl h-[560px]">
          <div ref={mapContainer} className="h-full w-full" />

          {/* Floating Coordinate HUD */}
          <div className="absolute top-3 left-3 z-10 bg-slate-900/90 backdrop-blur border border-slate-700/80 rounded-lg p-2.5 shadow-lg text-xs space-y-1">
            <div className="flex items-center gap-1.5 font-mono text-amber-400 font-semibold">
              <Crosshair className="h-3.5 w-3.5" />
              <span>{lat.toFixed(5)}°, {lng.toFixed(5)}°</span>
            </div>
            <div className="text-[10px] text-slate-400 flex items-center gap-1">
              <span>Coverage:</span>
              {isOod ? (
                <Badge variant="warning" size="sm">OOD (Global Prior)</Badge>
              ) : (
                <Badge variant="success" size="sm">Primary Grounding</Badge>
              )}
            </div>
          </div>

          {/* Map instructions overlay */}
          <div className="absolute bottom-3 left-3 z-10 bg-slate-950/80 backdrop-blur border border-slate-800 rounded px-2.5 py-1 text-[10px] text-slate-400">
            Click anywhere or drag marker to set coordinates
          </div>
        </div>

        {/* Prediction Panel (1 col) */}
        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-semibold flex items-center justify-between">
                <span>Location & Inference Parameters</span>
                <Badge variant="amber" size="sm">WGS84</Badge>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3.5">
              {/* OOD Warning */}
              {isOod && (
                <Alert variant="warning">
                  <AlertTitle className="text-xs">Out-of-Distribution Warning</AlertTitle>
                  <AlertDescription className="text-[11px]">
                    Coordinates lie outside primary UK STATS19 and French BAAC road training domains.
                    Predictions utilize uncalibrated global bayesian priors.
                  </AlertDescription>
                </Alert>
              )}

              {/* Mode switch */}
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1.5">
                  Timescale & Target Model
                </label>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    onClick={() => setInferenceMode('injury')}
                    className={`py-2 px-2.5 rounded-lg text-xs font-medium border flex items-center justify-center gap-1.5 transition-colors cursor-pointer ${
                      inferenceMode === 'injury'
                        ? 'bg-emerald-950 text-emerald-300 border-emerald-700'
                        : 'bg-slate-900 text-slate-400 border-slate-800 hover:bg-slate-800'
                    }`}
                  >
                    <HeartPulse className="h-3.5 w-3.5" />
                    <span>Injury (Immediate)</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => setInferenceMode('exposure')}
                    className={`py-2 px-2.5 rounded-lg text-xs font-medium border flex items-center justify-center gap-1.5 transition-colors cursor-pointer ${
                      inferenceMode === 'exposure'
                        ? 'bg-sky-950 text-sky-300 border-sky-700'
                        : 'bg-slate-900 text-slate-400 border-slate-800 hover:bg-slate-800'
                    }`}
                  >
                    <TrendingUp className="h-3.5 w-3.5" />
                    <span>Exposure (Annual)</span>
                  </button>
                </div>
              </div>

              {/* Coordinates Inputs */}
              <div className="grid grid-cols-2 gap-2">
                <Input
                  label="Latitude"
                  type="number"
                  step="0.0001"
                  value={lat}
                  onChange={(e) => {
                    const v = parseFloat(e.target.value) || 0;
                    setLat(v);
                    if (markerRef.current) markerRef.current.setLngLat([lng, v]);
                  }}
                />
                <Input
                  label="Longitude"
                  type="number"
                  step="0.0001"
                  value={lng}
                  onChange={(e) => {
                    const v = parseFloat(e.target.value) || 0;
                    setLng(v);
                    if (markerRef.current) markerRef.current.setLngLat([v, lat]);
                  }}
                />
              </div>

              {/* Road Type */}
              <Select
                label="Road Functional Class"
                value={roadClass}
                onChange={(e) => setRoadClass(e.target.value)}
              >
                <option value="MOTORWAY">Motorway / Autoroute / Interstate</option>
                <option value="PRIMARY_ARTERIAL">Primary Arterial / National Trunk</option>
                <option value="SECONDARY_COLLECTOR">Secondary Collector / Rural</option>
                <option value="LOCAL_OTHER">Local Road / Urban Street</option>
              </Select>

              {/* Dynamic inputs based on mode */}
              {inferenceMode === 'injury' ? (
                <div className="grid grid-cols-2 gap-2">
                  <Input
                    label="Speed Limit (km/h)"
                    type="number"
                    value={speedLimitKmh}
                    onChange={(e) => setSpeedLimitKmh(parseInt(e.target.value) || 50)}
                  />
                  <div className="space-y-1.5">
                    <label className="block text-xs font-medium text-slate-300">Weather Grounding</label>
                    <div className="h-10 rounded-lg border border-slate-800 bg-slate-950 px-3 py-2 text-xs text-sky-400 flex items-center gap-1 font-mono">
                      <Sparkles className="h-3 w-3" />
                      <span>ERA5 Hourly Auto</span>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="grid grid-cols-2 gap-2">
                  <Input
                    label="Daily Flow (AADF)"
                    type="number"
                    value={aadf}
                    onChange={(e) => setAadf(parseInt(e.target.value) || 1000)}
                  />
                  <Input
                    label="Segment Length (km)"
                    type="number"
                    step="0.1"
                    value={segmentLengthKm}
                    onChange={(e) => setSegmentLengthKm(parseFloat(e.target.value) || 1.0)}
                  />
                </div>
              )}

              {/* Action Button */}
              <Button
                variant="amber"
                className="w-full font-semibold"
                isLoading={isLoading}
                onClick={handleRunInference}
              >
                <Zap className="h-4 w-4 mr-1.5" />
                Run Live Inference
              </Button>

              {error && (
                <Alert variant="error">
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>

          {/* Results Display */}
          {(injuryResult || exposureResult) && (
            <Card className="border-emerald-800/60 bg-slate-900/90 shadow-xl">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-semibold text-emerald-400 flex items-center justify-between">
                  <span>Inference Output</span>
                  <Badge variant="success" size="sm">200 OK</Badge>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {injuryResult && (
                  <>
                    <div className="p-3 rounded-lg bg-slate-950/80 border border-slate-800">
                      <span className="text-[10px] text-slate-400 uppercase tracking-wide">Predicted Human Injury</span>
                      <div className="text-xl font-bold text-white mt-0.5 flex items-center gap-2">
                        <Badge
                          variant={
                            injuryResult.predicted_severity === 'FATAL'
                              ? 'destructive'
                              : injuryResult.predicted_severity === 'SERIOUS'
                              ? 'warning'
                              : 'success'
                          }
                          size="md"
                        >
                          {injuryResult.predicted_severity}
                        </Badge>
                        {injuryResult.fatality_risk_alert && (
                          <span className="text-xs text-rose-400 font-semibold flex items-center gap-1">
                            <AlertTriangle className="h-3.5 w-3.5" /> Fatality Alert
                          </span>
                        )}
                      </div>
                    </div>

                    <div className="space-y-1.5 text-xs">
                      <span className="text-slate-400">Class Probabilities:</span>
                      <div className="space-y-1">
                        {Object.entries(injuryResult.probabilities).map(([k, v]) => (
                          <div key={k} className="flex items-center justify-between font-mono text-[11px]">
                            <span className="text-slate-400">{k}:</span>
                            <div className="flex items-center gap-2">
                              <div className="w-20 bg-slate-800 h-1.5 rounded-full overflow-hidden">
                                <div
                                  className="h-full bg-emerald-500"
                                  style={{ width: `${(v as number) * 100}%` }}
                                />
                              </div>
                              <span className="text-slate-200">{((v as number) * 100).toFixed(1)}%</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </>
                )}

                {exposureResult && (
                  <>
                    <div className="p-3 rounded-lg bg-slate-950/80 border border-slate-800">
                      <span className="text-[10px] text-slate-400 uppercase tracking-wide">Expected Annual Crashes</span>
                      <div className="text-2xl font-bold text-sky-400 mt-0.5 font-mono">
                        {exposureResult.predicted_annual_crashes.toFixed(2)}
                        <span className="text-xs text-slate-400 font-normal"> crashes/yr</span>
                      </div>
                      <div className="mt-2 flex items-center justify-between text-xs">
                        <span className="text-slate-400">Assigned Risk Tier:</span>
                        <Badge
                          variant={
                            exposureResult.risk_tier.includes('HIGH')
                              ? 'destructive'
                              : exposureResult.risk_tier.includes('ELEVATED')
                              ? 'warning'
                              : 'success'
                          }
                          size="sm"
                        >
                          {exposureResult.risk_tier}
                        </Badge>
                      </div>
                    </div>

                    <div className="text-xs space-y-1 text-slate-400 border-t border-slate-800 pt-2 font-mono text-[11px]">
                      <div className="flex justify-between">
                        <span>Exposure:</span>
                        <span className="text-slate-200">{exposureResult.annual_exposure_million_vkt.toFixed(2)} M VKT</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Rate:</span>
                        <span className="text-slate-200">{exposureResult.predicted_crash_rate_per_million_vkt.toFixed(2)} / M VKT</span>
                      </div>
                    </div>
                  </>
                )}
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};
