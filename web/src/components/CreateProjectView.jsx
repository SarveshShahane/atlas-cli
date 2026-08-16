import React, { useRef } from "react";
import { 
  UploadCloud, 
  FileText, 
  Sparkles, 
  ShieldCheck, 
  CheckCircle2, 
  ArrowRight,
  Sliders,
  Target,
  FileCheck
} from "lucide-react";

export function CreateProjectView({
  file,
  setFile,
  name,
  setName,
  goal,
  setGoal,
  target,
  setTarget,
  testSize,
  setTestSize,
  outlierAction,
  setOutlierAction,
  anomalyAction,
  setAnomalyAction,
  onSubmit,
  busy,
  onCancel
}) {
  const fileInputRef = useRef(null);

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
      if (!name) {
        setName(e.dataTransfer.files[0].name.replace(/\.[^/.]+$/, ""));
      }
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      if (!name) {
        setName(e.target.files[0].name.replace(/\.[^/.]+$/, ""));
      }
    }
  };

  return (
    <div className="mx-auto max-w-4xl px-4 py-8">
      <div className="mb-8 text-center sm:text-left">
        <div className="inline-flex items-center gap-2 rounded-full border border-cyan-500/30 bg-cyan-500/10 px-3.5 py-1 text-xs font-bold text-cyan-400">
          <Sparkles className="h-3.5 w-3.5" />
          <span>New Experiment Orchestrator</span>
        </div>
        <h1 className="mt-3 text-3xl font-black text-white sm:text-4xl">
          Initialize Autonomous ML Project
        </h1>
        <p className="mt-2 text-sm text-slate-400">
          Upload your dataset (CSV, Parquet, JSON) and state your business goal. Atlas handles schema inference, outlier handling, feature engineering, parallel model training, and SHAP explainability.
        </p>
      </div>

      <form onSubmit={onSubmit} className="space-y-6">
        {/* Dropzone File Upload */}
        <div className="rounded-2xl border border-white/10 bg-slate-900/80 p-6 shadow-xl backdrop-blur-xl">
          <label className="mb-2 block text-xs font-bold uppercase tracking-wider text-slate-400 font-mono">
            Step 1: Dataset Upload
          </label>

          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept=".csv,.parquet,.json"
            className="hidden"
          />

          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`group relative flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-8 text-center transition-all ${
              file
                ? "border-cyan-500 bg-cyan-500/10"
                : "border-white/15 bg-slate-950/60 hover:border-cyan-500/50 hover:bg-slate-900/90"
            }`}
          >
            {file ? (
              <div className="flex flex-col items-center gap-2">
                <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-cyan-500/20 text-cyan-300 ring-1 ring-cyan-400">
                  <FileCheck className="h-6 w-6" />
                </div>
                <div className="text-center">
                  <span className="font-mono text-sm font-bold text-white block">{file.name}</span>
                  <span className="text-xs text-slate-400">
                    {(file.size / 1024).toFixed(1)} KB • Click or drag to change
                  </span>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-3">
                <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-cyan-500/10 text-cyan-400 ring-1 ring-cyan-500/30 group-hover:scale-110 transition-transform">
                  <UploadCloud className="h-7 w-7" />
                </div>
                <div>
                  <p className="text-sm font-bold text-white">
                    Drop CSV, Parquet, or JSON dataset here, or <span className="text-cyan-400 underline">browse</span>
                  </p>
                  <p className="mt-1 text-xs text-slate-500">
                    Maximum file size 100MB • Automatic column & missingness profiling
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Project Setup & Goal */}
        <div className="rounded-2xl border border-white/10 bg-slate-900/80 p-6 shadow-xl backdrop-blur-xl space-y-5">
          <label className="block text-xs font-bold uppercase tracking-wider text-slate-400 font-mono">
            Step 2: Project Parameters & Goal
          </label>

          <div className="grid grid-cols-1 gap-5 sm:grid-cols-2">
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1.5">Project Name</label>
              <input
                type="text"
                placeholder="e.g., Customer Churn Classifier"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full rounded-xl border border-white/10 bg-slate-950 px-4 py-2.5 text-sm text-white placeholder-slate-500 focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1.5">
                Target Variable <span className="text-slate-500 font-normal">(Leave blank for auto-inference)</span>
              </label>
              <input
                type="text"
                placeholder="e.g., churn / target / price"
                value={target}
                onChange={(e) => setTarget(e.target.value)}
                className="w-full rounded-xl border border-white/10 bg-slate-950 px-4 py-2.5 text-sm text-cyan-300 font-mono placeholder-slate-500 focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1.5">Business Goal / Modeling Intent</label>
            <textarea
              rows={3}
              placeholder="e.g., Predict whether a customer will churn based on tenure, monthly charges, and contract type."
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              className="w-full rounded-xl border border-white/10 bg-slate-950 px-4 py-2.5 text-sm text-white placeholder-slate-500 focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500"
            />

            {/* Quick Sample Goal Buttons */}
            <div className="mt-2.5 flex flex-wrap gap-2">
              <span className="text-[11px] text-slate-500 self-center">Quick presets:</span>
              {[
                "Classify Iris species accurately",
                "Predict patient heart disease risk",
                "Forecast housing median price"
              ].map((sample) => (
                <button
                  key={sample}
                  type="button"
                  onClick={() => setGoal(sample)}
                  className="rounded-full border border-white/10 bg-slate-950 px-3 py-1 text-[11px] font-medium text-slate-400 hover:border-cyan-500/40 hover:text-cyan-300 transition-colors"
                >
                  {sample}
                </button>
              ))}
            </div>
          </div>

          {/* Advanced Pipeline Settings */}
          <div className="grid grid-cols-1 gap-4 border-t border-white/10 pt-4 sm:grid-cols-3">
            <div>
              <label className="block text-[11px] font-semibold text-slate-400 mb-1">
                Validation Test Split ({Math.round(parseFloat(testSize) * 100)}%)
              </label>
              <input
                type="range"
                min="0.1"
                max="0.4"
                step="0.05"
                value={testSize}
                onChange={(e) => setTestSize(e.target.value)}
                className="w-full accent-cyan-400"
              />
            </div>

            <div>
              <label className="block text-[11px] font-semibold text-slate-400 mb-1">Outlier Strategy</label>
              <select
                value={outlierAction}
                onChange={(e) => setOutlierAction(e.target.value)}
                className="w-full rounded-lg border border-white/10 bg-slate-950 px-3 py-1.5 text-xs text-slate-300 focus:border-cyan-500 focus:outline-none"
              >
                <option value="report">Report & Flag Only</option>
                <option value="winsorize">IQR Winsorization (Clip)</option>
                <option value="drop">Drop Outlier Rows</option>
              </select>
            </div>

            <div>
              <label className="block text-[11px] font-semibold text-slate-400 mb-1">Anomaly Action</label>
              <select
                value={anomalyAction}
                onChange={(e) => setAnomalyAction(e.target.value)}
                className="w-full rounded-lg border border-white/10 bg-slate-950 px-3 py-1.5 text-xs text-slate-300 focus:border-cyan-500 focus:outline-none"
              >
                <option value="report">Report & Flag Only</option>
                <option value="filter">Filter Isolation Forest Anomalies</option>
              </select>
            </div>
          </div>
        </div>

        {/* Submit Actions */}
        <div className="flex items-center justify-end gap-3">
          <button
            type="button"
            onClick={onCancel}
            className="rounded-xl border border-white/10 bg-slate-900 px-5 py-2.5 text-xs font-bold text-slate-300 hover:bg-slate-800 hover:text-white transition-colors"
          >
            Cancel
          </button>

          <button
            type="submit"
            disabled={busy === "create" || !file || !goal.trim()}
            className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-cyan-500 via-blue-600 to-indigo-600 px-6 py-2.5 text-xs font-extrabold text-white shadow-lg shadow-cyan-500/25 transition-all hover:brightness-110 active:scale-95 disabled:opacity-50"
          >
            {busy === "create" ? (
              <>
                <Sparkles className="h-4 w-4 animate-spin text-amber-300" />
                <span>Initializing Workspace...</span>
              </>
            ) : (
              <>
                <span>Launch AutoML Workspace</span>
                <ArrowRight className="h-4 w-4" />
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}
