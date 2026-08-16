import React from "react";
import {
  Sliders,
  Download,
  CheckCircle2,
  Zap,
  ArrowRight,
  FileSpreadsheet,
  ShieldCheck,
  AlertTriangle,
  Scissors
} from "lucide-react";
import { atlasService } from "../api";

function StatCard({ label, value, color = "text-white" }) {
  return (
    <div className="rounded-xl border border-white/5 bg-slate-950/60 p-3">
      <span className="text-[10px] font-mono font-semibold uppercase text-slate-500">{label}</span>
      <p className={`mt-1 font-mono text-lg font-bold ${color} truncate`}>{value}</p>
    </div>
  );
}

export function StageCleaning({ project, onRunAction, busy }) {
  const cleaning = project?.cleaning || {};
  const cleanedUrl = atlasService.cleanedDatasetUrl(project.id);
  const prunedFeatures = cleaning.pruned_features || project?.data_health?.vif_metrics || [];
  const droppedAnomalies = cleaning.anomalies_dropped || 0;

  const hasCleaned =
    cleaning.has_cleaned_data ||
    cleaning.clean_row_count != null ||
    cleaning.cleaned_rows != null ||
    cleaning.outliers_clipped != null;

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="rounded-2xl border border-white/10 bg-slate-900/90 p-6 shadow-xl backdrop-blur-xl">
        <span className="rounded-full bg-cyan-500/10 px-3 py-1 font-mono text-xs font-bold uppercase tracking-wider text-cyan-400 ring-1 ring-cyan-500/20">
          Stage 2 • Hygiene & Cleaning
        </span>
        <h2 className="mt-2 text-2xl font-black text-white">
          Data Hygiene & Automated Feature Preprocessing
        </h2>
        <p className="mt-1 text-xs text-slate-400">
          IQR Winsorization outlier clipping, median/mode missing imputation, and multicollinearity feature pruning.
        </p>

        {/* Cleaned Metrics Cards */}
        <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4 border-t border-white/10 pt-4">
          <StatCard
            label="Rows Preserved"
            value={hasCleaned ? (cleaning.clean_row_count || project?.schema?.row_count || 0) : "—"}
            color="text-emerald-400"
          />
          <StatCard
            label="Outliers Clipped"
            value={hasCleaned ? `${cleaning.outliers_clipped || 0}` : "—"}
            color="text-amber-400"
          />
          <StatCard
            label="VIF Features Pruned"
            value={hasCleaned ? `${prunedFeatures.length}` : "—"}
            color="text-cyan-400"
          />
          <StatCard
            label="Imputed Cells"
            value={hasCleaned ? (cleaning.imputed_count || 0) : "—"}
            color="text-indigo-400"
          />
        </div>
      </div>

      {/* Pruned Features List */}
      {prunedFeatures.length > 0 && (
        <div className="rounded-2xl border border-white/10 bg-slate-900/80 p-5 shadow-xl backdrop-blur-xl">
          <h3 className="mb-3 text-sm font-bold text-white flex items-center gap-2">
            <Scissors className="h-4 w-4 text-amber-400" />
            <span>VIF Pruned Features ({prunedFeatures.length} removed)</span>
          </h3>
          <div className="flex flex-wrap gap-2">
            {prunedFeatures.map((feat, i) => {
              const name = typeof feat === "string" ? feat : feat.feature || feat.column || feat.name;
              return (
                <span
                  key={i}
                  className="flex items-center gap-1.5 rounded-lg border border-amber-500/20 bg-amber-500/10 px-2.5 py-1 font-mono text-[11px] text-amber-300"
                >
                  <AlertTriangle className="h-3 w-3" />
                  {name}
                </span>
              );
            })}
          </div>
          <p className="mt-3 text-xs text-slate-500">
            These features were removed due to high VIF collinearity to prevent multicollinearity in downstream model training.
          </p>
        </div>
      )}

      {/* Anomalies dropped notice */}
      {droppedAnomalies > 0 && (
        <div className="flex items-start gap-3 rounded-2xl border border-red-500/20 bg-red-500/10 p-4">
          <AlertTriangle className="h-4 w-4 text-red-400 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-xs font-bold text-red-300">
              {droppedAnomalies} Isolation Forest anomalies filtered
            </p>
            <p className="mt-0.5 text-[11px] text-slate-400">
              These rows were identified as statistical anomalies and excluded from the training dataset.
            </p>
          </div>
        </div>
      )}

      {/* Download Clean Dataset Card */}
      <div className="flex flex-col items-start justify-between gap-4 rounded-2xl border border-cyan-500/30 bg-gradient-to-r from-cyan-950/30 via-slate-900 to-slate-950 p-6 shadow-xl backdrop-blur-xl sm:flex-row sm:items-center">
        <div className="flex items-center gap-4">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-cyan-500/20 text-cyan-300 ring-1 ring-cyan-400">
            <FileSpreadsheet className="h-6 w-6" />
          </div>
          <div>
            <h3 className="text-base font-bold text-white">Cleaned Dataset Artifact</h3>
            <p className="text-xs text-slate-400">
              Export the fully preprocessed CSV dataset ready for downstream analysis or custom ML models.
            </p>
          </div>
        </div>

        <a
          href={cleanedUrl}
          download
          className="flex items-center gap-2 rounded-xl border border-cyan-500/40 bg-cyan-500/10 px-5 py-2.5 text-xs font-bold text-cyan-300 hover:bg-cyan-500/20 transition-all active:scale-95"
        >
          <Download className="h-4 w-4" />
          <span>Download Clean CSV</span>
        </a>
      </div>

      {/* Pipeline health summary */}
      <div className="flex items-center gap-3 rounded-2xl border border-emerald-500/20 bg-emerald-950/10 px-5 py-3">
        <ShieldCheck className="h-5 w-5 text-emerald-400 flex-shrink-0" />
        <p className="text-xs text-slate-300">
          <span className="font-bold text-emerald-300">Data hygiene complete.</span>{" "}
          The cleaned dataset is dimensionality-optimized and ready for LLM strategy synthesis & model training.
        </p>
      </div>

      {/* Next Stage Trigger */}
      <div className="flex items-center justify-end gap-3">
        <button
          onClick={() => onRunAction("plan")}
          disabled={busy === "plan"}
          className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 px-6 py-3 text-xs font-extrabold text-white shadow-lg shadow-cyan-500/25 transition-all hover:brightness-110 active:scale-95 disabled:opacity-50"
        >
          {busy === "plan" ? (
            <Zap className="h-4 w-4 animate-spin text-amber-300" />
          ) : (
            <>
              <span>Execute Stage 3: Synthesize Strategy & Plan</span>
              <ArrowRight className="h-4 w-4" />
            </>
          )}
        </button>
      </div>
    </div>
  );
}
