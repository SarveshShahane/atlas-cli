import React from "react";
import { 
  BrainCircuit, 
  Cpu, 
  Sparkles, 
  Zap, 
  ArrowRight, 
  CheckCircle2, 
  Layers,
  Code
} from "lucide-react";

export function StageModelPlan({ project, onRunAction, busy }) {
  const plan = project?.model_plan || project?.plan || {};
  const reasoning = plan.reasoning || plan.llm_synthesis || "LLM Strategy engine has analyzed your dataset structure and objective.";
  const candidateList = (plan.candidate_models && plan.candidate_models.length > 0)
    ? plan.candidate_models
    : (plan.models && plan.models.length > 0)
    ? plan.models
    : null;
  const candidates = candidateList || [
    { name: "XGBoost Classifier", type: "Gradient Boosting", priority: "High", rationale: "Handles non-linear feature interactions with built-in regularization." },
    { name: "Random Forest Classifier", type: "Bagging Ensemble", priority: "High", rationale: "Robust against overfitting; provides fast feature importance benchmarks." },
    { name: "CatBoost Classifier", type: "Categorical Gradient Boosting", priority: "Medium", rationale: "Optimal handling of high-cardinality categorical variables." },
    { name: "LightGBM Classifier", type: "Histogram Boosting", priority: "Medium", rationale: "Extremely fast leaf-wise tree growth for quick iteration." }
  ];

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="rounded-2xl border border-white/10 bg-slate-900/90 p-6 shadow-xl backdrop-blur-xl">
        <span className="rounded-full bg-cyan-500/10 px-3 py-1 font-mono text-xs font-bold uppercase tracking-wider text-cyan-400 ring-1 ring-cyan-500/20">
          Stage 3 • Strategy & Execution Plan
        </span>
        <h2 className="mt-2 text-2xl font-black text-white">
          LLM Model Strategy & Feature Engineering Blueprint
        </h2>
        <p className="mt-1 text-xs text-slate-400">
          Autonomous selection of candidate algorithms, hyperparameter spaces, and feature synthesis tailored to your target variable.
        </p>

        {/* LLM Reasoning Callout */}
        <div className="mt-5 rounded-xl border border-indigo-500/30 bg-indigo-950/30 p-4">
          <div className="flex items-center gap-2 text-indigo-300 font-bold text-xs mb-2">
            <Sparkles className="h-4 w-4 text-purple-400" />
            <span>AI Reasoning Synthesis</span>
          </div>
          <p className="text-xs text-slate-200 leading-relaxed font-sans">
            {reasoning}
          </p>
        </div>
      </div>

      {/* Candidate Models Cards Grid */}
      <div className="rounded-2xl border border-white/10 bg-slate-900/80 p-6 shadow-xl backdrop-blur-xl">
        <h3 className="mb-4 text-base font-extrabold text-white flex items-center gap-2">
          <Cpu className="h-4 w-4 text-cyan-400" />
          <span>Selected Candidate Model Architecture ({candidates.length})</span>
        </h3>

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          {candidates.map((model, idx) => (
            <div
              key={idx}
              className="flex flex-col justify-between rounded-xl border border-white/10 bg-slate-950/60 p-4 transition-all hover:border-cyan-500/40 hover:bg-slate-900"
            >
              <div>
                <div className="flex items-center justify-between">
                  <span className="rounded bg-cyan-500/10 px-2 py-0.5 font-mono text-[10px] font-bold text-cyan-300 border border-cyan-500/20">
                    {model.type || "Ensemble Model"}
                  </span>
                  <span className="font-mono text-[10px] font-bold text-amber-400">
                    Rank #{idx + 1}
                  </span>
                </div>

                <h4 className="mt-2 text-sm font-extrabold text-white">
                  {model.name || model.classifier}
                </h4>

                <p className="mt-1.5 text-xs text-slate-400">
                  {model.rationale || model.reason || "High performing algorithm selected for this dataset size and target distribution."}
                </p>
              </div>

              <div className="mt-4 border-t border-white/5 pt-2 flex items-center justify-between text-[11px] text-slate-500">
                <span>Metric Goal: <strong className="text-slate-300">{project.target_metric || "Accuracy / ROC-AUC"}</strong></span>
                <span className="text-emerald-400 font-semibold flex items-center gap-1">
                  <CheckCircle2 className="h-3 w-3" />
                  <span>Ready</span>
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Next Stage Action */}
      <div className="flex items-center justify-end">
        <button
          onClick={() => onRunAction("train")}
          disabled={busy === "train"}
          className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-cyan-500 via-blue-600 to-indigo-600 px-6 py-3 text-xs font-extrabold text-white shadow-lg shadow-cyan-500/25 transition-all hover:brightness-110 active:scale-95 disabled:opacity-50"
        >
          {busy === "train" ? (
            <Zap className="h-4 w-4 animate-spin text-amber-300" />
          ) : (
            <>
              <span>Execute Stage 4: Parallel Model Training & Benchmark</span>
              <ArrowRight className="h-4 w-4" />
            </>
          )}
        </button>
      </div>
    </div>
  );
}
