import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Globe,
  HeartPulse,
  TrendingUp,
  Cpu,
  ClockAlert,
  History,
  FileCode2,
  BarChart3,
  Layers,
  Sparkles,
} from 'lucide-react';
import { Badge } from '../ui/Badge';

export const Sidebar: React.FC = () => {
  const navItemClass = ({ isActive }: { isActive: boolean }) =>
    `flex items-center gap-3 px-3 py-2 text-xs font-medium rounded-lg transition-colors ${
      isActive
        ? 'bg-slate-800 text-amber-400 font-semibold border-l-2 border-amber-400 pl-2.5'
        : 'text-slate-400 hover:text-slate-100 hover:bg-slate-800/50'
    }`;

  return (
    <aside className="w-64 bg-slate-950 border-r border-slate-800/80 flex flex-col h-screen shrink-0 sticky top-0 select-none">
      {/* Official Platform Brand */}
      <div className="p-4 border-b border-slate-800 flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="h-8 w-8 rounded-lg bg-gradient-to-tr from-amber-600 to-amber-400 flex items-center justify-center shadow-md shadow-amber-500/20">
            <Globe className="h-5 w-5 text-slate-950" />
          </div>
          <div>
            <span className="font-bold text-sm tracking-tight text-white block">
              GLOBAL ACCIDENT
            </span>
            <span className="text-[10px] text-amber-400 tracking-wider font-mono uppercase block -mt-0.5">
              INTELLIGENCE
            </span>
          </div>
        </div>
      </div>

      {/* Demonstration Status Pill */}
      <div className="px-4 py-2.5 bg-slate-900/60 border-b border-slate-800/50">
        <div className="flex items-center justify-between text-xs mb-1">
          <span className="text-slate-300 font-medium truncate">
            Public Demonstration
          </span>
          <Badge variant="amber" size="sm">
            LIVE DEMO
          </Badge>
        </div>
        <div className="flex items-center gap-1.5 text-[11px] text-slate-400">
          <Sparkles className="h-3.5 w-3.5 text-amber-400" />
          <span>4 ML Models • Open Public Access</span>
        </div>
      </div>

      {/* Nav List */}
      <div className="flex-1 overflow-y-auto p-3 space-y-5 scrollbar-thin scrollbar-thumb-slate-800">
        {/* Core section */}
        <div>
          <p className="px-3 text-[10px] font-semibold text-slate-400 uppercase tracking-wider mb-1.5">
            Overview
          </p>
          <div className="space-y-0.5">
            <NavLink to="/" end className={navItemClass}>
              <LayoutDashboard className="h-4 w-4" />
              <span>Operations Dashboard</span>
            </NavLink>
            <NavLink to="/map" className={navItemClass}>
              <Globe className="h-4 w-4" />
              <span>Global Intelligence Map</span>
            </NavLink>
          </div>
        </div>

        {/* Prediction Intelligence */}
        <div>
          <p className="px-3 text-[10px] font-semibold text-slate-400 uppercase tracking-wider mb-1.5">
            Model Inference
          </p>
          <div className="space-y-0.5">
            <NavLink to="/predictions/injury-severity" className={navItemClass}>
              <HeartPulse className="h-4 w-4 text-emerald-400" />
              <div className="flex-1 flex items-center justify-between">
                <span>Injury Severity</span>
                <span className="text-[9px] bg-emerald-950 text-emerald-400 px-1 rounded border border-emerald-800/50">
                  COMMERCIAL
                </span>
              </div>
            </NavLink>

            <NavLink to="/predictions/exposure-risk" className={navItemClass}>
              <TrendingUp className="h-4 w-4 text-emerald-400" />
              <div className="flex-1 flex items-center justify-between">
                <span>Exposure Risk</span>
                <span className="text-[9px] bg-emerald-950 text-emerald-400 px-1 rounded border border-emerald-800/50">
                  COMMERCIAL
                </span>
              </div>
            </NavLink>

            <NavLink to="/predictions/traffic-severity" className={navItemClass}>
              <ClockAlert className="h-4 w-4 text-amber-400" />
              <div className="flex-1 flex items-center justify-between">
                <span>Traffic Impact</span>
                <span className="text-[9px] bg-amber-950 text-amber-400 px-1 rounded border border-amber-800/50">
                  CC-BY-NC
                </span>
              </div>
            </NavLink>

            <NavLink to="/predictions/corridor-risk" className={navItemClass}>
              <Cpu className="h-4 w-4 text-amber-400" />
              <div className="flex-1 flex items-center justify-between">
                <span>Corridor Risk</span>
                <span className="text-[9px] bg-amber-950 text-amber-400 px-1 rounded border border-amber-800/50">
                  RESEARCH
                </span>
              </div>
            </NavLink>

            <NavLink to="/history" className={navItemClass}>
              <History className="h-4 w-4" />
              <span>Prediction History</span>
            </NavLink>
          </div>
        </div>

        {/* Intelligence & Research */}
        <div>
          <p className="px-3 text-[10px] font-semibold text-slate-400 uppercase tracking-wider mb-1.5">
            Intelligence & Research
          </p>
          <div className="space-y-0.5">
            <NavLink to="/models" className={navItemClass}>
              <Layers className="h-4 w-4 text-amber-400" />
              <span>Model Registry</span>
            </NavLink>
            <NavLink to="/usage" className={navItemClass}>
              <BarChart3 className="h-4 w-4" />
              <span>Platform Analytics</span>
            </NavLink>
            <NavLink to="/docs" className={navItemClass}>
              <FileCode2 className="h-4 w-4" />
              <span>Developer Specs</span>
            </NavLink>
          </div>
        </div>
      </div>

      {/* Footer Info */}
      <div className="p-3 border-t border-slate-800/80 bg-slate-950/80 text-[10px] text-slate-400 flex flex-col gap-1">
        <div className="flex items-center justify-between">
          <span>Engine v2.1.0</span>
          <span className="text-emerald-400 flex items-center gap-1">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
            Operational
          </span>
        </div>
        <span>Multi-Country Spatial AI Systems</span>
      </div>
    </aside>
  );
};
