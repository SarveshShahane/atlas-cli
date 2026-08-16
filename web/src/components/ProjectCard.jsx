import React from "react";
import { 
  ArrowUpRight, 
  Trash2, 
  Database, 
  Trophy, 
  Sparkles, 
  CheckCircle2,
  Cpu
} from "lucide-react";

export function ProjectCard({ project, onOpen, onDelete }) {
  const winner = project.winner_model || project.results?.winner || (project.models && project.models[0]);
  const modelCount = project.models?.length || project.results?.experiments?.length || 0;
  const rowCount = project.schema?.row_count ?? project.dataset?.rows;
  const datasetName = project.dataset_filename || project.dataset?.name || "Dataset";

  return (
    <div 
      onClick={() => onOpen(project)}
      className="group relative flex flex-col justify-between rounded-2xl border border-white/10 bg-gradient-to-b from-slate-900/90 to-slate-950/90 p-5 shadow-xl backdrop-blur-xl transition-all duration-300 hover:-translate-y-1 hover:border-cyan-500/50 hover:shadow-2xl hover:shadow-cyan-500/10 cursor-pointer"
    >
      {/* Top Bar: Stage & Actions */}
      <div>
        <div className="flex items-center justify-between gap-2">
          <span className="rounded-full bg-cyan-500/10 px-2.5 py-1 font-mono text-[10px] font-bold uppercase tracking-widest text-cyan-300 ring-1 ring-cyan-500/30">
            {project.stage || "Data Health"}
          </span>

          <button
            onClick={(e) => onDelete(project.id, e)}
            className="opacity-0 group-hover:opacity-100 rounded-lg p-1.5 text-slate-500 hover:bg-red-500/10 hover:text-red-400 transition-all"
            title="Delete project"
          >
            <Trash2 className="h-4 w-4" />
          </button>
        </div>

        {/* Project Name & Goal */}
        <h3 className="mt-3 text-lg font-black text-white group-hover:text-cyan-300 transition-colors truncate">
          {project.name || project.id}
        </h3>

        <p className="mt-1.5 line-clamp-2 text-xs text-slate-400">
          {project.goal || "Autonomous ML prediction pipeline"}
        </p>
      </div>

      {/* Center Details & Metrics */}
      <div className="my-4 space-y-3 border-y border-white/10 py-3">
        {/* Dataset Meta */}
        <div className="flex items-center justify-between text-xs">
          <span className="flex items-center gap-1.5 text-slate-400">
            <Database className="h-3.5 w-3.5 text-cyan-400" />
            <span className="truncate max-w-[140px] font-mono">{datasetName}</span>
          </span>
          <span className="font-mono text-slate-300">
            {rowCount != null ? `${rowCount} rows` : "—"}
          </span>
        </div>

        {/* Target Variable Tag */}
        <div className="flex items-center justify-between text-xs">
          <span className="text-slate-400">Target Variable</span>
          <span className="rounded bg-slate-800 px-2 py-0.5 font-mono text-[11px] font-bold text-cyan-300 border border-cyan-500/20">
            {project.target || "Auto-detect"}
          </span>
        </div>

        {/* Winner Model Banner (if available) */}
        {winner ? (
          <div className="flex items-center justify-between rounded-lg border border-amber-500/30 bg-amber-500/10 p-2 text-xs">
            <div className="flex items-center gap-1.5 text-amber-300 font-bold">
              <Trophy className="h-4 w-4 text-amber-400" />
              <span className="truncate max-w-[130px]">{winner.classifier || winner.name || "Top Candidate"}</span>
            </div>
            <span className="font-mono font-extrabold text-amber-400">
              {winner.metric_value != null ? winner.metric_value.toFixed(3) : "Winner"}
            </span>
          </div>
        ) : (
          <div className="flex items-center justify-between rounded-lg border border-white/5 bg-slate-900/50 p-2 text-xs text-slate-500">
            <span className="flex items-center gap-1.5">
              <Cpu className="h-3.5 w-3.5" />
              <span>Models</span>
            </span>
            <span className="font-mono text-slate-400">{modelCount} Trained</span>
          </div>
        )}
      </div>

      {/* Footer CTA */}
      <div className="flex items-center justify-between text-xs">
        <span className="font-mono text-[11px] text-slate-500">
          ID: {project.id.slice(0, 8)}
        </span>
        <div className="flex items-center gap-1 font-bold text-cyan-400 group-hover:translate-x-0.5 transition-transform">
          <span>Open Workspace</span>
          <ArrowUpRight className="h-4 w-4" />
        </div>
      </div>
    </div>
  );
}
