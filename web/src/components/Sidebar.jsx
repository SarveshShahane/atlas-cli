import React from "react";
import { 
  Activity, 
  Sparkles, 
  BrainCircuit, 
  Trophy, 
  PieChart, 
  Rocket, 
  CheckCircle2, 
  Lock, 
  Play, 
  Trash2, 
  Database,
  ArrowRight,
  Zap,
  Sliders
} from "lucide-react";

const STAGE_CONFIG = [
  { id: "data_health", label: "1. Data Health & Profiler", icon: Activity, desc: "Schema, outliers & VIF collinearity" },
  { id: "cleaning", label: "2. Hygiene & Cleaning", icon: Sliders, desc: "IQR Winsorization & anomaly drops" },
  { id: "model_plan", label: "3. Strategy & Execution Plan", icon: BrainCircuit, desc: "LLM synthesis & candidate models" },
  { id: "training", label: "4. Training & Comparator", icon: Trophy, desc: "Parallel benchmark & gold winner" },
  { id: "explainability", label: "5. Explainability & SHAP", icon: PieChart, desc: "Global attributions & SHAP plots" },
  { id: "delivery", label: "6. Share, Launch & Exporter", icon: Rocket, desc: "Microservices, ONNX & PDF reports" },
];

export function Sidebar({ project, activeStage, onSelectStage, onRunAction, onDeleteProject, busy }) {
  if (!project) return null;

  const currentStageIndex = STAGE_CONFIG.findIndex((s) => s.id === (project.stage || "data_health"));
  const activeStageIndex = STAGE_CONFIG.findIndex((s) => s.id === activeStage);

  return (
    <aside className="w-full flex-shrink-0 border-r border-white/10 bg-slate-950/60 p-4 lg:w-72">
      {/* Project Overview Card */}
      <div className="mb-6 rounded-xl border border-white/10 bg-slate-900/80 p-4 shadow-xl backdrop-blur-md">
        <div className="mb-2 flex items-center justify-between">
          <span className="font-mono text-[10px] font-bold uppercase tracking-wider text-cyan-400">Workspace</span>
          <span className="rounded-full bg-cyan-500/10 px-2 py-0.5 font-mono text-[10px] font-semibold text-cyan-300 ring-1 ring-cyan-500/20">
            {project.stage || "Init"}
          </span>
        </div>

        <h3 className="truncate text-base font-extrabold text-white" title={project.name || project.id}>
          {project.name || project.id}
        </h3>
        
        <p className="mt-1 line-clamp-2 text-xs text-slate-400" title={project.goal}>
          {project.goal || "No goal specified"}
        </p>

        <div className="mt-4 grid grid-cols-2 gap-2 border-t border-white/10 pt-3 text-[11px]">
          <div>
            <span className="block text-slate-500 font-medium">Dataset</span>
            <span className="font-mono font-semibold text-slate-300 truncate block">
              {project.dataset_filename || project.dataset?.name || "Dataset"}
            </span>
          </div>
          <div>
            <span className="block text-slate-500 font-medium">Target</span>
            <span className="font-mono font-semibold text-cyan-400 truncate block">
              {project.target || "Auto"}
            </span>
          </div>
          <div>
            <span className="block text-slate-500 font-medium">Dimensions</span>
            <span className="font-mono text-slate-300">
              {(project.schema?.row_count ?? project.dataset?.rows) != null
                ? `${project.schema?.row_count ?? project.dataset?.rows} × ${project.schema?.col_count ?? project.dataset?.columns ?? 0}`
                : "—"}
            </span>
          </div>
          <div>
            <span className="block text-slate-500 font-medium">Models Trained</span>
            <span className="font-mono font-bold text-amber-400">
              {project.models?.length || project.results?.experiments?.length || 0}
            </span>
          </div>
        </div>

        {/* Global Action: Run Autonomous Pipeline */}
        <button
          onClick={() => onRunAction("autonomous")}
          disabled={busy === "autonomous"}
          className="mt-4 flex w-full items-center justify-center gap-2 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-600 px-3 py-2 text-xs font-bold text-white shadow-md shadow-cyan-500/20 transition-all hover:brightness-110 active:scale-95 disabled:opacity-50"
        >
          {busy === "autonomous" ? (
            <Zap className="h-4 w-4 animate-spin text-amber-300" />
          ) : (
            <Play className="h-4 w-4 fill-white" />
          )}
          <span>Run Auto Pipeline</span>
        </button>
      </div>

      {/* Pipeline Stages Navigation */}
      <div className="space-y-1.5">
        <span className="px-2 text-[10px] font-bold uppercase tracking-wider text-slate-500 font-mono">
          AutoML Pipeline Stages
        </span>

        {STAGE_CONFIG.map((stage, idx) => {
          const Icon = stage.icon;
          const isDone = idx < currentStageIndex || (idx === currentStageIndex && project.models?.length > 0);
          const isCurrent = activeStage === stage.id;
          const isLocked = idx > currentStageIndex + 1;

          return (
            <button
              key={stage.id}
              onClick={() => onSelectStage(stage.id)}
              disabled={isLocked}
              className={`group flex w-full items-center justify-between rounded-xl border p-3 text-left transition-all ${
                isCurrent
                  ? "border-cyan-500/50 bg-slate-900/90 text-white shadow-lg shadow-cyan-500/10 ring-1 ring-cyan-500/30"
                  : isDone
                  ? "border-white/5 bg-slate-900/40 text-slate-200 hover:border-white/20 hover:bg-slate-900/70"
                  : "border-transparent text-slate-500 hover:bg-slate-900/30"
              } ${isLocked ? "cursor-not-allowed opacity-40" : "cursor-pointer"}`}
            >
              <div className="flex items-center gap-3">
                <div
                  className={`flex h-8 w-8 items-center justify-center rounded-lg border transition-colors ${
                    isCurrent
                      ? "border-cyan-400 bg-cyan-500/20 text-cyan-300"
                      : isDone
                      ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-400"
                      : "border-white/10 bg-slate-900 text-slate-500"
                  }`}
                >
                  <Icon className="h-4 w-4" />
                </div>
                <div className="flex flex-col">
                  <span className={`text-xs font-bold ${isCurrent ? "text-cyan-300" : isDone ? "text-slate-200" : "text-slate-400"}`}>
                    {stage.label}
                  </span>
                  <span className="text-[10px] text-slate-500">{stage.desc}</span>
                </div>
              </div>

              <div>
                {isDone ? (
                  <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                ) : isLocked ? (
                  <Lock className="h-3.5 w-3.5 text-slate-600" />
                ) : isCurrent ? (
                  <span className="flex h-2 w-2 rounded-full bg-cyan-400 ring-4 ring-cyan-400/20" />
                ) : null}
              </div>
            </button>
          );
        })}
      </div>

      {/* Delete Project Danger Button */}
      <div className="mt-8 border-t border-white/10 pt-4">
        <button
          onClick={(e) => onDeleteProject(project.id, e)}
          className="flex w-full items-center justify-center gap-2 rounded-lg border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs font-semibold text-red-400 transition-colors hover:border-red-500/40 hover:bg-red-500/20 active:scale-95"
        >
          <Trash2 className="h-4 w-4" />
          <span>Delete Workspace</span>
        </button>
      </div>
    </aside>
  );
}
