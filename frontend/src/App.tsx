import React, { Suspense, lazy } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AuthProvider } from './context/AuthContext';
import { AppLayout } from './components/layout/AppLayout';

// Direct code-split modules
const DashboardPage = lazy(() => import('./pages/dashboard/DashboardPage').then(m => ({ default: m.DashboardPage })));
const MapIntelligencePage = lazy(() => import('./pages/map/MapIntelligencePage').then(m => ({ default: m.MapIntelligencePage })));
const InjuryPredictionPage = lazy(() => import('./pages/predictions/InjuryPredictionPage').then(m => ({ default: m.InjuryPredictionPage })));
const ExposureRiskPage = lazy(() => import('./pages/predictions/ExposureRiskPage').then(m => ({ default: m.ExposureRiskPage })));
const CorridorRiskPage = lazy(() => import('./pages/predictions/CorridorRiskPage').then(m => ({ default: m.CorridorRiskPage })));
const TrafficSeverityPage = lazy(() => import('./pages/predictions/TrafficSeverityPage').then(m => ({ default: m.TrafficSeverityPage })));
const HistoryPage = lazy(() => import('./pages/history/HistoryPage').then(m => ({ default: m.HistoryPage })));
const DeveloperDocsPage = lazy(() => import('./pages/docs/DeveloperDocsPage').then(m => ({ default: m.DeveloperDocsPage })));
const UsageAnalyticsPage = lazy(() => import('./pages/usage/UsageAnalyticsPage').then(m => ({ default: m.UsageAnalyticsPage })));
const ModelRegistryPage = lazy(() => import('./pages/models/ModelRegistryPage').then(m => ({ default: m.ModelRegistryPage })));

const PageLoader: React.FC = () => (
  <div className="h-64 flex flex-col items-center justify-center space-y-3">
    <div className="h-8 w-8 border-2 border-amber-500/20 border-t-amber-500 rounded-full animate-spin" />
    <span className="text-xs font-mono text-slate-400">Loading Global Accident Intelligence module...</span>
  </div>
);

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

export const App: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <BrowserRouter>
          <Suspense fallback={<PageLoader />}>
            <Routes>
              {/* Public Demonstration App Shell */}
              <Route element={<AppLayout />}>
                <Route path="/" element={<DashboardPage />} />
                <Route path="/dashboard" element={<Navigate to="/" replace />} />
                <Route path="/map" element={<MapIntelligencePage />} />
                <Route path="/predictions/injury-severity" element={<InjuryPredictionPage />} />
                <Route path="/predictions/exposure-risk" element={<ExposureRiskPage />} />
                <Route path="/predictions/traffic-severity" element={<TrafficSeverityPage />} />
                <Route path="/predictions/corridor-risk" element={<CorridorRiskPage />} />
                <Route path="/history" element={<HistoryPage />} />
                <Route path="/models" element={<ModelRegistryPage />} />
                <Route path="/usage" element={<UsageAnalyticsPage />} />
                <Route path="/docs" element={<DeveloperDocsPage />} />
              </Route>

              {/* Catch-all redirect to direct dashboard */}
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </Suspense>
        </BrowserRouter>
      </AuthProvider>
    </QueryClientProvider>
  );
};

export default App;
