import React, { useEffect, useState } from "react";
import {
  History,
  RefreshCw,
  CheckCircle2,
  AlertCircle,
  Clock,
  Terminal,
  ArrowLeft,
  Inbox,
  Trophy
} from "lucide-react";
import { atlasService } from "../api";

const STATUS_CONFIG = {
  completed: { label: "COMPLETED", cls: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20" },
  success: { label: "SUCCESS", cls: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20" },
  failed: { label: "FAILED", cls: "bg-red-500/10 text-red-400 border-red-500/20" },
  error: { label: "ERROR", cls: "bg-red-500/10 text-red-400 border-red-500/20" },
  running: { label: "RUNNING", cls: "bg-cyan-500/10 text-cyan-400 border-cyan-500/20" },
  pending: { label: "PENDING", cls: "bg-amber-500/10 text-amber-400 border-amber-500/20" },
};

function StatusBadge({ status }) {
  const key = (status || "completed").toLowerCase();
  const cfg = STATUS_CONFIG[key] || STATUS_CONFIG.completed;
  return (
    <span className={`rounded px-2 py-0.5 text-[10px] font-bold border font-mono ${cfg.cls}`}>
      {cfg.label}
    </span>
  );
}

export function RunsHistory({ onBack }) {
  const [runs, setRuns] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const loadRuns = async () => {
    setBusy(true);
    setError("");
    try {
      const data = await atlasService.listRuns();
      setRuns(data || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    loadRuns();
  }, []);

  return (
    <div className="mx-auto max-w-6xl px-4 py-8 space-y-6">
      {/* Page Header */}
      <div className="flex flex-col gap-4 border-b border-white/10 pb-6 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <button
            onClick={onBack}
            className="mb-3 flex items-center gap-1.5 text-xs font-semibold text-slate-400 hover:text-cyan-400 transition-colors"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            <span>Back to Workspaces</span>
          </button>

          <div className="inline-flex items-center gap-2 rounded-full border border-cyan-500/30 bg-cyan-500/10 px-3 py-1 text-xs font-bold text-cyan-400 mb-2">
            <Terminal className="h-3.5 w-3.5" />
            <span>Audit Log & History</span>
          </div>

          <h1 className="text-3xl font-black text-white">
            System Execution Audit Runs
          </h1>
          <p className="mt-1 text-xs text-slate-400">
            Historical trace of all CLI and GUI pipeline executions, hyperparameters, and generated model artifacts.
          </p>
        </div>

        <button
          onClick={loadRuns}
          disabled={busy}
          className="flex items-center gap-2 rounded-xl border border-white/10 bg-slate-900 px-4 py-2 text-xs font-bold text-slate-300 hover:bg-slate-800 hover:text-white transition-colors disabled:opacity-50 self-start sm:self-auto"
        >
          <RefreshCw className={`h-4 w-4 ${busy ? "animate-spin text-cyan-400" : ""}`} />
          <span>Refresh Logs</span>
        </button>
      </div>

      {/* Error State */}
      {error && (
        <div className="flex items-center gap-3 rounded-xl border border-red-500/30 bg-red-500/10 p-4">
          <AlertCircle className="h-4 w-4 text-red-400 flex-shrink-0" />
          <p className="text-xs font-semibold text-red-300">{error}</p>
        </div>
      )}

      {/* Runs Table Card */}
      <div className="rounded-2xl border border-white/10 bg-slate-900/80 p-6 shadow-xl backdrop-blur-xl">
        {busy && runs.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-16 text-center">
            <RefreshCw className="h-8 w-8 text-cyan-400 animate-spin mb-3" />
            <p className="text-sm font-bold text-slate-300">Loading audit logs...</p>
          </div>
        ) : runs.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-16 text-center">
            <div className="flex h-16 w-16 items-center justify-center rounded-2xl border border-white/10 bg-slate-950/80 mb-4">
              <Inbox className="h-8 w-8 text-slate-600" />
            </div>
            <h3 className="text-base font-bold text-slate-300">No Audit Runs Found</h3>
            <p className="mt-1 text-xs text-slate-500 max-w-sm">
              Run a pipeline project to populate system run logs. Runs will appear here after any stage execution.
            </p>
          </div>
        ) : (
          <>
            <div className="mb-4 flex items-center justify-between">
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <History className="h-4 w-4 text-cyan-400" />
                <span>Execution History</span>
              </h3>
              <span className="font-mono text-xs text-slate-500">
                {runs.length} {runs.length === 1 ? "run" : "runs"}
              </span>
            </div>

            <div className="overflow-x-auto rounded-xl border border-white/10">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-950/80 text-slate-400 font-mono uppercase text-[10px]">
                  <tr>
                    <th className="px-4 py-3">Run ID</th>
                    <th className="px-4 py-3">Status</th>
                    <th className="px-4 py-3">Stage</th>
                    <th className="px-4 py-3">Winner Model</th>
                    <th className="px-4 py-3">Primary Metric</th>
                    <th className="px-4 py-3">Timestamp</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5 bg-slate-900/50">
                  {runs.map((run, idx) => (
                    <tr key={run.run_id || run.id || idx} className="hover:bg-slate-800/40 transition-colors">
                      <td className="px-4 py-3 font-mono font-bold text-cyan-400">
                        {(run.run_id || run.id || `run-${idx}`).slice(0, 12)}
                      </td>
                      <td className="px-4 py-3">
                        <StatusBadge status={run.status} />
                      </td>
                      <td className="px-4 py-3 font-mono text-slate-300">
                        {run.stage || "delivery"}
                      </td>
                      <td className="px-4 py-3 font-mono text-white">
                        {run.winner_model ? (
                          <span className="flex items-center gap-1.5">
                            <Trophy className="h-3 w-3 text-amber-400" />
                            {run.winner_model}
                          </span>
                        ) : (
                          <span className="text-slate-500">—</span>
                        )}
                      </td>
                      <td className="px-4 py-3 font-mono text-cyan-300 font-bold">
                        {run.metric_value != null ? run.metric_value.toFixed(4) : "—"}
                      </td>
                      <td className="px-4 py-3 font-mono text-slate-500 flex items-center gap-1.5">
                        <Clock className="h-3 w-3 flex-shrink-0" />
                        {run.created_at
                          ? new Date(run.created_at).toLocaleString()
                          : "Just now"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
