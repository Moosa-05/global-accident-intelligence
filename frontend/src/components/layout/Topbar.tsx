import React from 'react';
import { Badge } from '../ui/Badge';
import { Globe, Cpu, CheckCircle2, Layers } from 'lucide-react';
import { Link } from 'react-router-dom';

export const Topbar: React.FC = () => {
  return (
    <header className="h-14 bg-slate-900/90 backdrop-blur border-b border-slate-800 px-6 flex items-center justify-between sticky top-0 z-30 select-none">
      {/* Left: Platform Context */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2">
          <Globe className="h-4 w-4 text-amber-400" />
          <span className="text-xs font-bold tracking-tight text-white hidden sm:inline">
            GLOBAL ACCIDENT INTELLIGENCE
          </span>
        </div>
        <Badge variant="amber" size="sm">
          PUBLIC DEMONSTRATION MODE
        </Badge>
      </div>

      {/* Right: Operational Status & Model Badges */}
      <div className="flex items-center gap-4 text-xs">
        <div className="hidden lg:flex items-center gap-2 text-[11px] text-slate-400 font-mono">
          <span className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-slate-950 border border-slate-800">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
            4 ML Pipelines Active
          </span>
          <span className="px-2 py-0.5 rounded bg-slate-950 border border-slate-800 text-slate-300">
            WGS84 Global Grid
          </span>
          <span className="px-2 py-0.5 rounded bg-slate-950 border border-slate-800 text-slate-300">
            ECMWF ERA5 Grounding
          </span>
        </div>

        <Link
          to="/models"
          className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-amber-500/10 hover:bg-amber-500/20 border border-amber-500/30 text-amber-300 transition-colors text-xs font-medium"
        >
          <Layers className="h-3.5 w-3.5" />
          <span>Model Registry</span>
        </Link>
      </div>
    </header>
  );
};
