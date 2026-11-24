import React, { useState, useEffect } from 'react';
import { api } from '../../services/api';
import type { PredictionHistoryItem, AuditLogItem } from '../../types/history';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';
import { Input } from '../../components/ui/Input';
import { Select } from '../../components/ui/Select';
import { Badge } from '../../components/ui/Badge';
import { Modal } from '../../components/ui/Modal';
import {
  History,
  ShieldCheck,
  Search,
  Filter,
  Eye,
  RefreshCw,
  FileText,
  Activity,
} from 'lucide-react';

export const HistoryPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'predictions' | 'audit'>('predictions');
  const [predictionRecords, setPredictionRecords] = useState<PredictionHistoryItem[]>([]);
  const [auditRecords, setAuditRecords] = useState<AuditLogItem[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [typeFilter, setTypeFilter] = useState<string>('');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [selectedRecord, setSelectedRecord] = useState<any | null>(null);

  const fetchData = async () => {
    setIsLoading(true);
    try {
      if (activeTab === 'predictions') {
        const res = await api.getPredictionHistory(100, typeFilter || undefined);
        setPredictionRecords(res.records);
      } else {
        const res = await api.getAuditLogs(100);
        setAuditRecords(res);
      }
    } catch (err) {
      console.error('Failed to load history records', err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [activeTab, typeFilter]);

  const filteredPredictions = predictionRecords.filter((rec) => {
    if (!searchQuery) return true;
    const q = searchQuery.toLowerCase();
    return (
      rec.summary.toLowerCase().includes(q) ||
      rec.model_version.toLowerCase().includes(q) ||
      rec.type.toLowerCase().includes(q)
    );
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
            <History className="h-6 w-6 text-amber-400" />
            Prediction History & Audit Trail
          </h1>
          <p className="text-xs text-slate-400">
            Immutable database records of all inference invocations, payload summaries, and latency metrics
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={fetchData}
            isLoading={isLoading}
            className="text-xs"
          >
            <RefreshCw className="h-3.5 w-3.5 mr-1" />
            Refresh Logs
          </Button>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-slate-800 space-x-6 text-sm">
        <button
          onClick={() => setActiveTab('predictions')}
          className={`pb-3 font-medium transition-colors border-b-2 cursor-pointer ${
            activeTab === 'predictions'
              ? 'border-amber-400 text-amber-400'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          Prediction Records ({predictionRecords.length})
        </button>
        <button
          onClick={() => setActiveTab('audit')}
          className={`pb-3 font-medium transition-colors border-b-2 cursor-pointer ${
            activeTab === 'audit'
              ? 'border-amber-400 text-amber-400'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          API Audit Trail ({auditRecords.length})
        </button>
      </div>

      {/* Prediction Records Tab */}
      {activeTab === 'predictions' && (
        <Card>
          <CardHeader className="pb-3 flex flex-col md:flex-row md:items-center justify-between gap-3">
            <div className="flex items-center gap-3 flex-1 max-w-md">
              <div className="relative w-full">
                <Search className="h-4 w-4 absolute left-3 top-2.5 text-slate-500" />
                <Input
                  placeholder="Search by summary or model version..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-9 h-9 text-xs"
                />
              </div>
            </div>

            <div className="flex items-center gap-3">
              <Select
                value={typeFilter}
                onChange={(e) => setTypeFilter(e.target.value)}
                className="h-9 text-xs w-48"
              >
                <option value="">All Model Types</option>
                <option value="injury">Injury Severity (Commercial)</option>
                <option value="exposure">Exposure Risk (Commercial)</option>
                <option value="risk">Corridor Risk (Research)</option>
                <option value="severity">Traffic Delay (Research)</option>
              </Select>
            </div>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="py-12 text-center text-xs text-slate-400">Loading prediction records...</div>
            ) : filteredPredictions.length === 0 ? (
              <div className="py-12 text-center text-xs text-slate-500">
                No prediction history found matching filter.
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="text-[10px] uppercase font-mono text-slate-500 border-b border-slate-800">
                    <tr>
                      <th className="pb-2.5">Timestamp</th>
                      <th className="pb-2.5">Pipeline</th>
                      <th className="pb-2.5">Model Version</th>
                      <th className="pb-2.5">Summary / Outcome</th>
                      <th className="pb-2.5">Coordinates</th>
                      <th className="pb-2.5">Latency</th>
                      <th className="pb-2.5 text-right">Payload</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 text-slate-300 font-mono text-[11px]">
                    {filteredPredictions.map((rec) => (
                      <tr key={rec.id} className="hover:bg-slate-800/30">
                        <td className="py-3 text-slate-400 whitespace-nowrap">
                          {rec.created_at ? new Date(rec.created_at).toLocaleString() : 'N/A'}
                        </td>
                        <td className="py-3">
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
                        <td className="py-3 text-slate-400">{rec.model_version}</td>
                        <td className="py-3 font-sans font-medium text-slate-200 truncate max-w-sm">
                          {rec.summary}
                        </td>
                        <td className="py-3 text-slate-400">
                          {rec.latitude && rec.longitude
                            ? `${rec.latitude.toFixed(3)}, ${rec.longitude.toFixed(3)}`
                            : 'N/A'}
                        </td>
                        <td className="py-3 text-emerald-400">{rec.latency_ms.toFixed(1)} ms</td>
                        <td className="py-3 text-right">
                          <Button
                            variant="ghost"
                            size="sm"
                            className="p-1 text-slate-400 hover:text-amber-400 cursor-pointer"
                            onClick={() => setSelectedRecord(rec)}
                          >
                            <Eye className="h-4 w-4" />
                          </Button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Audit Logs Tab */}
      {activeTab === 'audit' && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-semibold">Tenant API Request Audit Trail</CardTitle>
            <CardDescription>Security and compliance log of all HTTP API interactions</CardDescription>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="py-12 text-center text-xs text-slate-400">Loading audit trail...</div>
            ) : auditRecords.length === 0 ? (
              <div className="py-12 text-center text-xs text-slate-500">No audit logs recorded.</div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="text-[10px] uppercase font-mono text-slate-500 border-b border-slate-800">
                    <tr>
                      <th className="pb-2.5">Timestamp</th>
                      <th className="pb-2.5">Method</th>
                      <th className="pb-2.5">Endpoint</th>
                      <th className="pb-2.5">Model</th>
                      <th className="pb-2.5">Status</th>
                      <th className="pb-2.5">Latency</th>
                      <th className="pb-2.5">Client IP</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 text-slate-300 font-mono text-[11px]">
                    {auditRecords.map((log) => (
                      <tr key={log.id} className="hover:bg-slate-800/30">
                        <td className="py-3 text-slate-400 whitespace-nowrap">
                          {log.created_at ? new Date(log.created_at).toLocaleString() : 'N/A'}
                        </td>
                        <td className="py-3">
                          <span className="px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 text-[10px] font-bold">
                            {log.method}
                          </span>
                        </td>
                        <td className="py-3 font-sans text-slate-200">{log.endpoint}</td>
                        <td className="py-3 text-slate-400">{log.model_id || 'system'}</td>
                        <td className="py-3">
                          <span
                            className={`font-bold ${
                              log.status_code >= 400
                                ? 'text-rose-400'
                                : log.status_code >= 300
                                ? 'text-amber-400'
                                : 'text-emerald-400'
                            }`}
                          >
                            {log.status_code}
                          </span>
                        </td>
                        <td className="py-3 text-slate-400">{log.latency_ms.toFixed(1)} ms</td>
                        <td className="py-3 text-slate-500">{log.ip_address || '127.0.0.1'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Record Inspect Modal */}
      <Modal
        isOpen={!!selectedRecord}
        onClose={() => setSelectedRecord(null)}
        title="Prediction Record Details"
        maxWidth="lg"
      >
        {selectedRecord && (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-3 text-xs p-3 rounded-lg bg-slate-950 border border-slate-800 font-mono">
              <div>
                <span className="text-slate-500">Record ID:</span>
                <p className="text-slate-300 truncate">{selectedRecord.id}</p>
              </div>
              <div>
                <span className="text-slate-500">Model Pipeline:</span>
                <p className="text-amber-400">{selectedRecord.model_id}</p>
              </div>
            </div>

            <div>
              <span className="text-xs font-semibold text-slate-300 block mb-1">Parsed Output:</span>
              <pre className="p-4 rounded-lg bg-slate-950 border border-slate-800 text-emerald-400 text-xs font-mono overflow-x-auto max-h-72">
                {JSON.stringify(selectedRecord.result, null, 2)}
              </pre>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
};
