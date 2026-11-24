import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../../services/api';
import type { ModelsListResponse, ModelMetadataItem } from '../../types/models';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../../components/ui/Card';
import { Badge } from '../../components/ui/Badge';
import { Button } from '../../components/ui/Button';
import { Alert, AlertTitle, AlertDescription } from '../../components/ui/Alert';
import {
  Cpu,
  ShieldCheck,
  ShieldAlert,
  Globe,
  Database,
  BarChart3,
  HeartPulse,
  TrendingUp,
  ClockAlert,
  ArrowRight,
  Layers,
  Sparkles,
  CheckCircle2,
  ExternalLink,
} from 'lucide-react';

export const ModelRegistryPage: React.FC = () => {
  const [data, setData] = useState<ModelsListResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [filterCommercial, setFilterCommercial] = useState<boolean>(false);

  useEffect(() => {
    const fetchModels = async () => {
      setLoading(true);
      try {
        const res = await api.getModels(filterCommercial ? true : undefined);
        setData(res);
      } catch (err) {
        console.error('Failed to load model registry', err);
      } finally {
        setLoading(false);
      }
    };
    fetchModels();
  }, [filterCommercial]);

  const getModelRoute = (modelId: string) => {
    switch (modelId) {
      case 'human_injury_severity_v1':
        return '/predictions/injury-severity';
      case 'exposure_risk_v1':
        return '/predictions/exposure-risk';
      case 'accident_risk_v1':
        return '/predictions/corridor-risk';
      case 'accident_severity_v2':
        return '/predictions/traffic-severity';
      default:
        return '/dashboard';
    }
  };

  const getModelIcon = (modelId: string) => {
    switch (modelId) {
      case 'human_injury_severity_v1':
        return <HeartPulse className="h-5 w-5 text-emerald-400" />;
      case 'exposure_risk_v1':
        return <TrendingUp className="h-5 w-5 text-emerald-400" />;
      case 'accident_risk_v1':
        return <Cpu className="h-5 w-5 text-amber-400" />;
      case 'accident_severity_v2':
        return <ClockAlert className="h-5 w-5 text-amber-400" />;
      default:
        return <Layers className="h-5 w-5 text-sky-400" />;
    }
  };

  return (
    <div className="space-y-6 max-w-6xl">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
            <Cpu className="h-6 w-6 text-amber-400" />
            AI Model Registry & Architecture
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Verified machine learning pipelines, multi-country training corpora, evaluation metrics, and licensing governance
          </p>
        </div>

        {/* Filter Toggle */}
        <div className="flex items-center gap-2 bg-slate-900 border border-slate-800 p-1 rounded-lg">
          <Button
            variant={!filterCommercial ? 'amber' : 'ghost'}
            size="sm"
            onClick={() => setFilterCommercial(false)}
            className="text-xs"
          >
            All Models (4)
          </Button>
          <Button
            variant={filterCommercial ? 'amber' : 'ghost'}
            size="sm"
            onClick={() => setFilterCommercial(true)}
            className="text-xs"
          >
            Commercial Cleared (2)
          </Button>
        </div>
      </div>

      {/* Top Architecture Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 backdrop-blur">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-mono text-slate-400 uppercase">Models Registered</span>
            <Layers className="h-4 w-4 text-amber-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-white mt-1">
            {data?.total_registered_models || 4}
          </div>
          <span className="text-[11px] text-slate-400">4 distinct operational tasks</span>
        </div>

        <div className="p-4 rounded-xl border border-emerald-900/40 bg-emerald-950/20 backdrop-blur">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-mono text-emerald-400 uppercase">Commercial Cleared</span>
            <ShieldCheck className="h-4 w-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-emerald-400 mt-1">
            {data?.commercial_cleared_count || 2}
          </div>
          <span className="text-[11px] text-emerald-500/80">OGL v3.0 & Licence Ouverte</span>
        </div>

        <div className="p-4 rounded-xl border border-amber-900/40 bg-amber-950/20 backdrop-blur">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-mono text-amber-400 uppercase">Research Benchmarks</span>
            <ShieldAlert className="h-4 w-4 text-amber-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-amber-400 mt-1">
            {data?.restricted_research_count || 2}
          </div>
          <span className="text-[11px] text-amber-500/80">CC BY-NC-SA 4.0 / TDOT</span>
        </div>

        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 backdrop-blur">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-mono text-slate-400 uppercase">Global Grounding</span>
            <Globe className="h-4 w-4 text-sky-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-sky-400 mt-1">
            WGS84
          </div>
          <span className="text-[11px] text-slate-400">ECMWF ERA5 Weather-Linked</span>
        </div>
      </div>

      {/* Multi-Scale Modeling Framework Explanation */}
      <Card className="border-slate-800 bg-slate-900/50">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-semibold flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-amber-400" />
            Multi-Scale Predictive Modeling Architecture
          </CardTitle>
          <CardDescription>
            Rigorous separation of spatial and temporal scales prevents methodological errors
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-3 gap-4 text-xs">
            <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800">
              <span className="font-semibold text-emerald-400 block mb-1">Meso-Scale: Injury Severity</span>
              <p className="text-slate-300 leading-relaxed">
                Predicts trauma severity (Slight, Serious, Fatal) <em>conditional on a collision having occurred</em>.
                Trained on 160k+ verified UK STATS19 and French BAAC crash events.
              </p>
            </div>
            <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800">
              <span className="font-semibold text-emerald-400 block mb-1">Macro-Scale: Road Exposure</span>
              <p className="text-slate-300 leading-relaxed">
                Estimates expected annual crash frequency per road link and accident rates per Million Vehicle-Kilometres Travelled (MVKT).
                Poisson deviance calibrated on 18,933 physical count links.
              </p>
            </div>
            <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800">
              <span className="font-semibold text-amber-400 block mb-1">Micro-Scale: Corridor Delay & Risk</span>
              <p className="text-slate-300 leading-relaxed">
                Real-time 15-minute sensor telemetry risk along smart highway corridors and nationwide US traffic congestion delay.
                Strictly designated as academic research benchmarks.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Models List */}
      <div className="space-y-4">
        {loading ? (
          <div className="p-12 text-center text-xs text-slate-400">Loading model specifications...</div>
        ) : (
          data?.models.map((m) => (
            <Card
              key={m.model_id}
              className={`border transition-all ${
                m.commercial_use_permitted
                  ? 'border-slate-800 hover:border-emerald-500/40 bg-slate-900/70'
                  : 'border-slate-800 hover:border-amber-500/40 bg-slate-900/70'
              }`}
            >
              <CardHeader className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3">
                <div className="flex items-start gap-3">
                  <div className="h-10 w-10 rounded-xl bg-slate-950 border border-slate-800 flex items-center justify-center shrink-0 mt-0.5">
                    {getModelIcon(m.model_id)}
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <CardTitle className="text-base font-bold text-white">
                        {m.model_name}
                      </CardTitle>
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                        v{m.model_version}
                      </span>
                    </div>
                    <span className="text-[11px] font-mono text-slate-400 block mt-0.5">
                      ID: {m.model_id} • Target: {m.target_type}
                    </span>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <Badge variant={m.commercial_use_permitted ? 'success' : 'amber'} size="md">
                    {m.commercial_use_permitted ? 'COMMERCIAL CLEARED' : 'RESEARCH BENCHMARK'}
                  </Badge>
                  <Link to={getModelRoute(m.model_id)}>
                    <Button variant="outline" size="sm" className="text-xs">
                      Try Model <ArrowRight className="h-3.5 w-3.5 ml-1" />
                    </Button>
                  </Link>
                </div>
              </CardHeader>

              <CardContent className="space-y-4 pt-1">
                <p className="text-xs text-slate-300 leading-relaxed">
                  {m.task_description}
                </p>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                  <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80">
                    <span className="text-slate-400 font-medium block mb-1 flex items-center gap-1.5">
                      <Database className="h-3.5 w-3.5 text-sky-400" /> Training Data & Territory
                    </span>
                    <p className="text-slate-200 font-medium">{m.training_data_source}</p>
                    <p className="text-slate-400 text-[11px] mt-0.5">{m.training_geography}</p>
                  </div>

                  <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80">
                    <span className="text-slate-400 font-medium block mb-1 flex items-center gap-1.5">
                      <BarChart3 className="h-3.5 w-3.5 text-emerald-400" /> Performance & Validation
                    </span>
                    <div className="flex flex-wrap gap-2 text-[11px] font-mono text-slate-300">
                      {m.metrics && Object.keys(m.metrics).length > 0 ? (
                        Object.entries(m.metrics).map(([k, v]) => (
                          <span key={k} className="px-1.5 py-0.5 bg-slate-900 rounded border border-slate-800">
                            {k}: <strong className="text-amber-400">{typeof v === 'number' ? v.toFixed(4) : String(v)}</strong>
                          </span>
                        ))
                      ) : (
                        <span className="text-slate-400">Verified LightGBM Gradient Boosting</span>
                      )}
                    </div>
                  </div>
                </div>

                <div className="p-3 rounded-lg bg-slate-950/40 border border-slate-800/60 text-[11px] text-slate-400 flex items-start gap-2">
                  <Globe className="h-4 w-4 text-slate-500 shrink-0 mt-0.5" />
                  <span>
                    <strong className="text-slate-300">Scope & Extensibility:</strong> {m.coverage_and_scope_notes}
                  </span>
                </div>
              </CardContent>
            </Card>
          ))
        )}
      </div>

      {/* Extensibility & Standards */}
      {data?.global_extensibility && (
        <Card className="border-slate-800 bg-slate-900/60">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-semibold flex items-center gap-2">
              <Globe className="h-4 w-4 text-sky-400" />
              Global Engineering Standards & Adapter Architecture
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3 text-xs">
              <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800">
                <span className="text-slate-400">Coordinates</span>
                <p className="font-mono text-slate-200 mt-1 font-semibold">{data.global_extensibility.coordinate_system}</p>
              </div>
              <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800">
                <span className="text-slate-400">Units Engine</span>
                <p className="font-mono text-slate-200 mt-1 font-semibold">Metric & Imperial Autonomous</p>
              </div>
              <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800">
                <span className="text-slate-400">Timezone Engine</span>
                <p className="font-mono text-slate-200 mt-1 font-semibold">{data.global_extensibility.timezone_standard}</p>
              </div>
              <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800">
                <span className="text-slate-400">Weather Grounding</span>
                <p className="font-mono text-slate-200 mt-1 font-semibold">Copernicus ERA5 Reanalysis</p>
              </div>
            </div>

            <div className="mt-3 pt-3 border-t border-slate-800 flex flex-wrap gap-1.5 items-center text-[11px]">
              <span className="text-slate-400 mr-2">Loaded Backend Adapters:</span>
              {data.global_extensibility.provider_adapters.map((a) => (
                <span key={a} className="px-2 py-0.5 rounded bg-slate-950 border border-slate-800 font-mono text-slate-300">
                  {a}
                </span>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};
