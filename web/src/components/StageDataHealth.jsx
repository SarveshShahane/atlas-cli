import React from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  BarChart2,
  Zap,
  ArrowRight,
  Database,
  Layers,
  Info
} from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";

function StatCard({ label, value, color = "text-white" }) {
  return (
    <div className="rounded-xl border border-white/5 bg-slate-950/60 p-3">
      <span className="text-[10px] font-mono font-semibold uppercase text-slate-500">{label}</span>
      <p className={`mt-1 font-mono text-lg font-bold ${color} truncate`}>{value}</p>
    </div>
  );
}

export function StageDataHealth({ project, onRunAction, busy }) {
  const schema = project?.schema || {};
  const columns = schema.columns || project?.data_health?.columns || [];
  const health = schema.health_assessment || {};
  const vifList = schema.vif_collinearity || project?.data_health?.vif_metrics || [];
  const risks = health.risk_assessment || project?.data_health?.risks || [];

  const rowCount = schema.row_count ?? project?.dataset?.rows ?? 0;
  const colCount = schema.col_count ?? project?.dataset?.columns ?? columns.length;
  const totalMissingRatio = schema.total_missing_ratio ?? (
    columns.length > 0
      ? columns.reduce((acc, c) => acc + (c.missing_ratio || 0), 0) / columns.length
      : 0
  );

  // Data Quality Score Calculation
  const totalCols = columns.length || 1;
  const highMissingCols = columns.filter((c) => (c.missing_ratio || 0) > 0.2).length;
  const qualityScore = Math.max(
    20,
    Math.min(100, Math.round(100 - (highMissingCols / totalCols) * 40 - (vifList.length / totalCols) * 20))
  );

  const scoreColor =
    qualityScore >= 80 ? "text-emerald-400" : qualityScore >= 50 ? "text-amber-400" : "text-red-400";
  const scoreStroke =
    qualityScore >= 80 ? "#34d399" : qualityScore >= 50 ? "#fbbf24" : "#f87171";

  // Chart Data for Missing Rates
  const chartData = columns
    .map((col) => ({
      name: col.name || col.column,
      missing: Math.round((col.missing_ratio || 0) * 100),
      cardinality: col.unique_count || 0
    }))
    .slice(0, 12);

  const hasData = columns.length > 0 || rowCount > 0;

  // Parsed Risk Assessment Entries
  const parsedRisks = risks.map((r) => {
    let item = r;
    if (typeof r === "string") {
      try {
        item = JSON.parse(r);
      } catch (e) {
        item = { description: r };
      }
    }
    return {
      severity: item.severity || "WARNING",
      category: item.category || item.type || "Quality Risk",
      column: item.column || item.feature || "Dataset-wide",
      description: item.description || item.message || (typeof r === "string" ? r : JSON.stringify(r)),
      recommendation: item.recommendation || item.action || "Review feature during cleaning step."
    };
  });

  return (
    <div className="space-y-6">
      {/* Top Banner & Quality Score */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-4">
        <div className="col-span-3 rounded-2xl border border-white/10 bg-slate-900/90 p-6 shadow-xl backdrop-blur-xl">
          <div>
            <span className="rounded-full bg-cyan-500/10 px-3 py-1 font-mono text-xs font-bold uppercase tracking-wider text-cyan-400 ring-1 ring-cyan-500/20">
              Stage 1 • Data Health & Profiler
            </span>
            <h2 className="mt-2 text-2xl font-black text-white">
              Dataset Health & Collinearity Assessment
            </h2>
            <p className="mt-1 text-xs text-slate-400">
              Automated multi-method outlier detection, VIF variance inflation profiling, and feature cardinality analysis.
            </p>
          </div>

          {/* Key Quick Metrics Grid */}
          <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4 border-t border-white/10 pt-4">
            <StatCard
              label="Rows × Cols"
              value={hasData ? `${rowCount} × ${colCount}` : "—"}
              color="text-white"
            />
            <StatCard
              label="Total Missing Rate"
              value={hasData ? `${(totalMissingRatio * 100).toFixed(1)}%` : "—"}
              color="text-cyan-400"
            />
            <StatCard
              label="VIF Risk Features"
              value={hasData ? `${vifList.length} High Risk` : "—"}
              color="text-amber-400"
            />
            <StatCard
              label="Target Column"
              value={project.target || "Inferred"}
              color="text-emerald-400"
            />
          </div>
        </div>

        {/* Quality Score Ring */}
        <div className="flex flex-col items-center justify-center rounded-2xl border border-cyan-500/30 bg-gradient-to-br from-cyan-950/40 via-slate-900 to-slate-950 p-6 text-center shadow-xl backdrop-blur-xl">
          <span className="text-xs font-mono font-bold uppercase tracking-wider text-slate-400">
            Health Index
          </span>

          <div className="relative my-3 flex h-24 w-24 items-center justify-center">
            <svg className="h-full w-full -rotate-90" viewBox="0 0 36 36">
              <path
                className="text-slate-800"
                strokeWidth="3.5"
                stroke="currentColor"
                fill="none"
                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
              />
              <path
                strokeDasharray={`${hasData ? qualityScore : 0}, 100`}
                strokeWidth="3.5"
                strokeLinecap="round"
                stroke={scoreStroke}
                fill="none"
                className="transition-all duration-1000"
                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
              />
            </svg>
            <div className="absolute flex flex-col items-center">
              <span className={`font-mono text-2xl font-black ${hasData ? scoreColor : "text-slate-600"}`}>
                {hasData ? `${qualityScore}%` : "—"}
              </span>
            </div>
          </div>

          <span className={`text-[11px] font-medium flex items-center gap-1 ${hasData ? "text-emerald-400" : "text-slate-500"}`}>
            <CheckCircle2 className="h-3.5 w-3.5" />
            <span>{hasData ? "Ready for Hygiene" : "Awaiting data"}</span>
          </span>
        </div>
      </div>

      {/* Risk Assessment Table */}
      {parsedRisks.length > 0 && (
        <div className="rounded-2xl border border-amber-500/20 bg-slate-900/80 p-6 shadow-xl backdrop-blur-xl">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="text-base font-extrabold text-amber-300 flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-amber-400" />
              <span>Risk Assessment & Quality Flags ({parsedRisks.length} flags)</span>
            </h3>
            <span className="font-mono text-xs text-slate-500">
              Automated Severity & Remediation Matrix
            </span>
          </div>

          <div className="overflow-x-auto rounded-xl border border-amber-500/20">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-950/80 text-amber-300/80 font-mono uppercase text-[10px]">
                <tr>
                  <th className="px-4 py-3">Severity</th>
                  <th className="px-4 py-3">Category</th>
                  <th className="px-4 py-3">Feature / Target</th>
                  <th className="px-4 py-3">Issue Description</th>
                  <th className="px-4 py-3">Recommended Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5 bg-slate-900/50">
                {parsedRisks.map((risk, idx) => {
                  const isCritical = risk.severity.toUpperCase() === "CRITICAL" || risk.severity.toUpperCase() === "HIGH";
                  return (
                    <tr key={idx} className="hover:bg-slate-800/40 transition-colors">
                      <td className="px-4 py-3 font-mono">
                        <span className={`inline-flex items-center gap-1 rounded px-2 py-0.5 text-[10px] font-bold border ${
                          isCritical
                            ? "bg-red-500/20 text-red-400 border-red-500/30"
                            : "bg-amber-500/20 text-amber-300 border-amber-500/30"
                        }`}>
                          {risk.severity}
                        </span>
                      </td>
                      <td className="px-4 py-3 font-mono font-semibold text-white">
                        {risk.category}
                      </td>
                      <td className="px-4 py-3 font-mono text-cyan-300">
                        {risk.column || "Dataset-wide"}
                      </td>
                      <td className="px-4 py-3 text-slate-300 max-w-xs leading-relaxed">
                        {risk.description}
                      </td>
                      <td className="px-4 py-3 text-emerald-300/90 font-mono text-[11px] max-w-xs leading-relaxed">
                        {risk.recommendation}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Interactive Recharts Missingness Distribution */}
      {chartData.length > 0 ? (
        <div className="rounded-2xl border border-white/10 bg-slate-900/80 p-6 shadow-xl backdrop-blur-xl">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="text-base font-extrabold text-white flex items-center gap-2">
              <BarChart2 className="h-4 w-4 text-cyan-400" />
              <span>Feature Missing Ratio Profiling (%)</span>
            </h3>
            <span className="text-xs text-slate-400 font-mono">Top {chartData.length} Features</span>
          </div>

          <div className="h-56 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                <XAxis
                  dataKey="name"
                  stroke="#64748b"
                  fontSize={11}
                  tickLine={false}
                  angle={-25}
                  textAnchor="end"
                />
                <YAxis stroke="#64748b" fontSize={11} tickLine={false} unit="%" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#0f172a",
                    borderColor: "#1e293b",
                    borderRadius: "12px",
                    color: "#f8fafc"
                  }}
                  formatter={(val) => [`${val}%`, "Missing Rate"]}
                />
                <Bar dataKey="missing" radius={[6, 6, 0, 0]}>
                  {chartData.map((entry, idx) => (
                    <Cell
                      key={idx}
                      fill={entry.missing > 20 ? "#ef4444" : entry.missing > 5 ? "#f59e0b" : "#06b6d4"}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      ) : hasData ? null : (
        <div className="rounded-2xl border border-dashed border-white/10 bg-slate-900/40 p-8 text-center">
          <Info className="h-8 w-8 text-slate-600 mx-auto mb-2" />
          <p className="text-sm font-bold text-slate-400">No column data yet</p>
          <p className="text-xs text-slate-600 mt-1">Run the data health stage to profile this dataset.</p>
        </div>
      )}

      {/* Columns Schema Detailed Table */}
      {columns.length > 0 && (
        <div className="rounded-2xl border border-white/10 bg-slate-900/80 p-6 shadow-xl backdrop-blur-xl">
          <h3 className="mb-4 text-base font-extrabold text-white flex items-center gap-2">
            <Layers className="h-4 w-4 text-cyan-400" />
            <span>Feature Schema & Outlier Profiler</span>
            <span className="ml-auto font-mono text-xs text-slate-500 font-normal">
              {columns.length} columns
            </span>
          </h3>

          <div className="overflow-x-auto rounded-xl border border-white/10">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-950/80 text-slate-400 font-mono uppercase text-[10px]">
                <tr>
                  <th className="px-4 py-3">Feature Name</th>
                  <th className="px-4 py-3">Inferred Type</th>
                  <th className="px-4 py-3">Missing Rate</th>
                  <th className="px-4 py-3">Unique Values</th>
                  <th className="px-4 py-3">Outlier Status</th>
                  <th className="px-4 py-3">Role</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5 bg-slate-900/50">
                {columns.map((col, idx) => {
                  const colName = col.name || col.column;
                  const isTarget = colName === project.target;
                  const isHighMissing = (col.missing_ratio || 0) > 0.15;

                  return (
                    <tr key={idx} className={`hover:bg-slate-800/40 transition-colors ${isTarget ? "bg-emerald-950/20" : ""}`}>
                      <td className="px-4 py-3 font-mono font-bold text-white">
                        <div className="flex items-center gap-2">
                          {colName}
                          {isTarget && (
                            <span className="rounded bg-emerald-500/20 px-1.5 py-0.5 text-[9px] font-bold text-emerald-400 border border-emerald-500/30">
                              TARGET
                            </span>
                          )}
                        </div>
                      </td>
                      <td className="px-4 py-3 font-mono text-cyan-300">
                        {col.data_type || col.dtype || "numerical"}
                      </td>
                      <td className="px-4 py-3 font-mono">
                        <span className={isHighMissing ? "text-red-400 font-bold" : "text-slate-300"}>
                          {((col.missing_ratio || 0) * 100).toFixed(1)}%
                        </span>
                        {isHighMissing && (
                          <AlertTriangle className="ml-1 inline h-3 w-3 text-red-400" />
                        )}
                      </td>
                      <td className="px-4 py-3 font-mono text-slate-300">
                        {col.unique_count ?? "—"}
                      </td>
                      <td className="px-4 py-3 font-mono">
                        {col.outlier_count ? (
                          <span className="text-amber-400 font-semibold">
                            {col.outlier_count} detected
                          </span>
                        ) : (
                          <span className="text-emerald-400/70">Normal</span>
                        )}
                      </td>
                      <td className="px-4 py-3 font-mono text-slate-400">
                        {isTarget ? "Target Label" : "Predictor Feature"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* VIF Collinearity List */}
      {vifList.length > 0 && (
        <div className="rounded-2xl border border-amber-500/20 bg-slate-900/80 p-5 shadow-xl">
          <h3 className="mb-3 text-sm font-bold text-amber-300 flex items-center gap-2">
            <AlertTriangle className="h-4 w-4" />
            <span>High VIF Collinearity Risk ({vifList.length} features)</span>
          </h3>
          <div className="flex flex-wrap gap-2">
            {vifList.map((v, i) => {
              const name = typeof v === "string" ? v : v.feature || v.column || v.name;
              const vifVal = v.vif != null ? v.vif.toFixed(1) : null;
              return (
                <span key={i} className="flex items-center gap-1.5 rounded-lg border border-amber-500/20 bg-amber-500/10 px-2.5 py-1 font-mono text-[11px] text-amber-300">
                  {name}
                  {vifVal && <span className="text-amber-500 font-bold">VIF {vifVal}</span>}
                </span>
              );
            })}
          </div>
        </div>
      )}

      {/* Stage Action CTA */}
      <div className="flex items-center justify-end">
        <button
          onClick={() => onRunAction("clean")}
          disabled={busy === "clean"}
          className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 px-6 py-3 text-xs font-extrabold text-white shadow-lg shadow-cyan-500/25 transition-all hover:brightness-110 active:scale-95 disabled:opacity-50"
        >
          {busy === "clean" ? (
            <Zap className="h-4 w-4 animate-spin text-amber-300" />
          ) : (
            <>
              <span>Execute Stage 2: Data Hygiene & Cleaning</span>
              <ArrowRight className="h-4 w-4" />
            </>
          )}
        </button>
      </div>
    </div>
  );
}
