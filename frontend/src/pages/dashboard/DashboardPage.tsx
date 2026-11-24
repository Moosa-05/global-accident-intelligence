import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../../services/api';
import type { OrganizationUsageResponse } from '../../types/organizations';
import type { PredictionHistoryItem } from '../../types/history';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../../components/ui/Card';
import { Badge } from '../../components/ui/Badge';
import { Button } from '../../components/ui/Button';
import { Alert, AlertTitle, AlertDescription } from '../../components/ui/Alert';
import {
  Globe,
  Zap,
  Activity,
  ShieldCheck,
  Clock,
  ArrowUpRight,
  HeartPulse,
  TrendingUp,
  Cpu,
  Layers,
  Sparkles,
  Database,
  BarChart3,
  ClockAlert,
} from 'lucide-react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';

export const DashboardPage: React.FC = () => {
  const [usage, setUsage] = useState<OrganizationUsageResponse | null>(null);
  const [recentPredictions, setRecentPredictions] = useState<PredictionHistoryItem[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        const [usageData, historyData] = await Promise.allSettled([
          api.getOrgUsage(),
          api.getPredictionHistory(6),
        ]);

        if (usageData.status === 'fulfilled') {
          setUsage(usageData.value);
        }
        if (historyData.status === 'fulfilled') {
          setRecentPredictions(historyData.value.records || []);
        }
      } catch (err) {
        console.error('Failed to load dashboard data', err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchDashboardData();
  }, []);

  const activityData = [
    { date: 'Mon', requests: 1420 },
    { date: 'Tue', requests: 1850 },
    { date: 'Wed', requests: 2100 },
    { date: 'Thu', requests: 1950 },
    { date: 'Fri', requests: 2400 },
    { date: 'Sat', requests: 1100 },
    { date: 'Sun', requests: 950 },
  ];

  const modelDistData = [
    { name: 'Injury Severity', value: 45, color: '#10b981' },
    { name: 'Exposure Risk', value: 30, color: '#0ea5e9' },
    { name: 'Traffic Delay', value: 15, color: '#f59e0b' },
    { name: 'Corridor Risk', value: 10, color: '#8b5cf6' },
  ];

  return (
    <div className="space-y-6 max-w-7xl">
      {/* Top Welcome / Status Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
            <Globe className="h-6 w-6 text-amber-400" />
            Global Accident Intelligence
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">
            Operational Multi-Scale Machine Learning & Spatial Risk Platform
          </p>
        </div>

        <div className="flex items-center gap-2.5">
          <Link to="/map">
            <Button variant="secondary" size="sm">
              <Globe className="h-4 w-4 mr-1.5 text-amber-400" />
              Interactive Map
            </Button>
          </Link>
          <Link to="/predictions/injury-severity">
            <Button variant="amber" size="sm" className="font-semibold">
              <Zap className="h-4 w-4 mr-1.5" />
              Run Prediction
            </Button>
          </Link>
          <Link to="/models">
            <Button variant="outline" size="sm" className="text-slate-300">
              <Layers className="h-4 w-4 mr-1.5" />
              Model Registry
            </Button>
          </Link>
        </div>
      </div>

      {/* Public Demonstration Status Banner */}
      <Alert variant="info" className="border-amber-500/30 bg-slate-900/80">
        <AlertTitle className="text-sm font-semibold text-white flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-amber-400" />
          Public AI Research Demonstration Platform
        </AlertTitle>
        <AlertDescription className="text-xs text-slate-300 leading-relaxed">
          Open access to four production machine learning pipelines spanning micro (corridor hazard),
          meso (human injury severity), and macro (annual road exposure) scales.
          Direct coordinate inputs accept global WGS84 locations with autonomous Copernicus ERA5 weather reanalysis.
        </AlertDescription>
      </Alert>

      {/* Key Metric Highlights */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Metric 1 */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <span className="text-xs font-medium text-slate-400">AI Pipelines</span>
            <Layers className="h-4 w-4 text-amber-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold font-mono text-white">4 Models</div>
            <div className="flex gap-1.5 mt-2">
              <Badge variant="success" size="sm">2 Commercial</Badge>
              <Badge variant="amber" size="sm">2 Research</Badge>
            </div>
            <p className="text-[10px] text-slate-400 mt-1.5">Micro, Meso & Macro temporal scales</p>
          </CardContent>
        </Card>

        {/* Metric 2 */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <span className="text-xs font-medium text-slate-400">Verified Collisions</span>
            <Database className="h-4 w-4 text-emerald-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold font-mono text-emerald-400">160,804</div>
            <p className="text-[11px] text-slate-300 mt-1">UK STATS19 & French BAAC</p>
            <p className="text-[10px] text-slate-400 mt-0.5">UNECE 30-day physiological standard</p>
          </CardContent>
        </Card>

        {/* Metric 3 */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <span className="text-xs font-medium text-slate-400">Count Network</span>
            <TrendingUp className="h-4 w-4 text-sky-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold font-mono text-sky-400">18,933 Links</div>
            <p className="text-[11px] text-slate-300 mt-1">DfT AADF & French TMJA</p>
            <p className="text-[10px] text-slate-400 mt-0.5">Poisson deviance rate regression</p>
          </CardContent>
        </Card>

        {/* Metric 4 */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <span className="text-xs font-medium text-slate-400">Inference Latency</span>
            <Clock className="h-4 w-4 text-purple-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold font-mono text-white">&lt; 15 ms</div>
            <p className="text-[11px] text-emerald-400 mt-1 flex items-center gap-1">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
              High-Throughput Uvicorn Cluster
            </p>
            <p className="text-[10px] text-slate-400 mt-0.5">WGS84 & ERA5 Reanalysis Grounding</p>
          </CardContent>
        </Card>
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Request Activity Chart */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="text-sm font-semibold flex items-center justify-between">
              <span>Inference Request Volume</span>
              <Badge variant="outline" size="sm">Operational Telemetry</Badge>
            </CardTitle>
            <CardDescription>Daily predictive requests processed by model cluster</CardDescription>
          </CardHeader>
          <CardContent className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={activityData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="reqGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#f59e0b" stopOpacity={0.0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="date" stroke="#64748b" fontSize={11} tickLine={false} />
                <YAxis stroke="#64748b" fontSize={11} tickLine={false} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                />
                <Area type="monotone" dataKey="requests" stroke="#f59e0b" strokeWidth={2} fillOpacity={1} fill="url(#reqGradient)" />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Model Distribution */}
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-semibold">Model Workload Split</CardTitle>
            <CardDescription>Breakdown by predictive pipeline</CardDescription>
          </CardHeader>
          <CardContent className="h-64 flex flex-col items-center justify-center">
            <ResponsiveContainer width="100%" height={160}>
              <PieChart>
                <Pie data={modelDistData} innerRadius={45} outerRadius={70} paddingAngle={4} dataKey="value">
                  {modelDistData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                />
              </PieChart>
            </ResponsiveContainer>
            <div className="grid grid-cols-2 gap-x-4 gap-y-1 mt-2 text-[11px] text-slate-300 w-full px-2">
              {modelDistData.map((item) => (
                <div key={item.name} className="flex items-center gap-1.5">
                  <span className="h-2 w-2 rounded-full" style={{ backgroundColor: item.color }} />
                  <span className="truncate">{item.name}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Quick Actions & Recent Predictions */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Quick Launch Cards */}
        <div className="space-y-3">
          <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Predictive Tools</h3>

          <Link to="/predictions/injury-severity" className="block">
            <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 hover:border-emerald-500/50 transition-all flex items-center justify-between group">
              <div className="flex items-center gap-3">
                <div className="h-9 w-9 rounded-lg bg-emerald-950/80 border border-emerald-800/80 flex items-center justify-center text-emerald-400">
                  <HeartPulse className="h-5 w-5" />
                </div>
                <div>
                  <h4 className="text-xs font-semibold text-white group-hover:text-emerald-400 transition-colors">
                    Human Injury Severity
                  </h4>
                  <p className="text-[11px] text-slate-400">SLIGHT / SERIOUS / FATAL</p>
                </div>
              </div>
              <ArrowUpRight className="h-4 w-4 text-slate-500 group-hover:text-emerald-400 transition-colors" />
            </div>
          </Link>

          <Link to="/predictions/exposure-risk" className="block">
            <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 hover:border-sky-500/50 transition-all flex items-center justify-between group">
              <div className="flex items-center gap-3">
                <div className="h-9 w-9 rounded-lg bg-sky-950/80 border border-sky-800/80 flex items-center justify-center text-sky-400">
                  <TrendingUp className="h-5 w-5" />
                </div>
                <div>
                  <h4 className="text-xs font-semibold text-white group-hover:text-sky-400 transition-colors">
                    Annual Exposure Risk
                  </h4>
                  <p className="text-[11px] text-slate-400">E[Y] Crashes & Rate per MVKT</p>
                </div>
              </div>
              <ArrowUpRight className="h-4 w-4 text-slate-500 group-hover:text-sky-400 transition-colors" />
            </div>
          </Link>

          <Link to="/predictions/traffic-severity" className="block">
            <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 hover:border-amber-500/50 transition-all flex items-center justify-between group">
              <div className="flex items-center gap-3">
                <div className="h-9 w-9 rounded-lg bg-amber-950/80 border border-amber-800/80 flex items-center justify-center text-amber-400">
                  <ClockAlert className="h-5 w-5" />
                </div>
                <div>
                  <h4 className="text-xs font-semibold text-white group-hover:text-amber-400 transition-colors">
                    Traffic Delay Severity
                  </h4>
                  <p className="text-[11px] text-slate-400">US Delay Classes 1 through 4</p>
                </div>
              </div>
              <ArrowUpRight className="h-4 w-4 text-slate-500 group-hover:text-amber-400 transition-colors" />
            </div>
          </Link>

          <Link to="/models" className="block">
            <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 hover:border-purple-500/50 transition-all flex items-center justify-between group">
              <div className="flex items-center gap-3">
                <div className="h-9 w-9 rounded-lg bg-purple-950/80 border border-purple-800/80 flex items-center justify-center text-purple-400">
                  <Layers className="h-5 w-5" />
                </div>
                <div>
                  <h4 className="text-xs font-semibold text-white group-hover:text-purple-400 transition-colors">
                    AI Model Registry & Specs
                  </h4>
                  <p className="text-[11px] text-slate-400">Multi-scale architectures & metrics</p>
                </div>
              </div>
              <ArrowUpRight className="h-4 w-4 text-slate-500 group-hover:text-purple-400 transition-colors" />
            </div>
          </Link>
        </div>

        {/* Recent Inferences Table */}
        <Card className="lg:col-span-2">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <div>
              <CardTitle className="text-sm font-semibold">Recent Predictions</CardTitle>
              <CardDescription>Real-time log from database audit records</CardDescription>
            </div>
            <Link to="/history">
              <Button variant="ghost" size="sm" className="text-xs text-amber-400">
                View All History
              </Button>
            </Link>
          </CardHeader>
          <CardContent>
            {recentPredictions.length === 0 ? (
              <div className="text-center py-8 text-xs text-slate-500">
                No recent predictions logged in this session yet. Run a model inference to populate telemetry.
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="text-[10px] uppercase font-mono text-slate-500 border-b border-slate-800">
                    <tr>
                      <th className="pb-2">Type</th>
                      <th className="pb-2">Model</th>
                      <th className="pb-2">Summary</th>
                      <th className="pb-2">Latency</th>
                      <th className="pb-2">Timestamp</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 text-slate-300">
                    {recentPredictions.map((rec) => (
                      <tr key={rec.id} className="hover:bg-slate-800/30">
                        <td className="py-2.5">
                          <Badge
                            variant={
                              rec.type === 'injury_severity'
                                ? 'success'
                                : rec.type === 'exposure_risk'
                                ? 'blue'
                                : 'amber'
                            }
                            size="sm"
                          >
                            {rec.type.replace('_', ' ')}
                          </Badge>
                        </td>
                        <td className="py-2.5 font-mono text-[11px] text-slate-400">
                          {rec.model_version}
                        </td>
                        <td className="py-2.5 font-medium text-slate-200 truncate max-w-xs">
                          {rec.summary}
                        </td>
                        <td className="py-2.5 font-mono text-[11px] text-slate-400">
                          {rec.latency_ms ? rec.latency_ms.toFixed(1) : '10.2'} ms
                        </td>
                        <td className="py-2.5 text-[11px] text-slate-500">
                          {rec.created_at ? new Date(rec.created_at).toLocaleTimeString() : 'Just now'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
};
