import React, { useState } from "react";
import {
  Rocket,
  Download,
  Copy,
  Check,
  Terminal,
  Zap,
  FileCode,
  Play,
  CheckCircle2,
  FileText,
  Code2,
  Sliders,
  AlertCircle,
  ToggleLeft,
  ToggleRight
} from "lucide-react";
import { atlasService } from "../api";

function CodeBlock({ label, code, copyKey, copiedKey, onCopy, color = "text-cyan-300", icon: Icon = FileCode }) {
  return (
    <div className="rounded-xl border border-white/10 bg-slate-950 p-4">
      <div className="flex items-center justify-between mb-3">
        <span className={`font-mono text-xs font-bold ${color} flex items-center gap-1.5`}>
          <Icon className="h-3.5 w-3.5" />
          {label}
        </span>
        <button
          onClick={() => onCopy(code, copyKey)}
          className="flex items-center gap-1 text-[11px] font-semibold text-slate-400 hover:text-white transition-colors"
        >
          {copiedKey === copyKey ? (
            <Check className="h-3.5 w-3.5 text-emerald-400" />
          ) : (
            <Copy className="h-3.5 w-3.5" />
          )}
          <span>{copiedKey === copyKey ? "Copied!" : "Copy"}</span>
        </button>
      </div>
      <pre className={`overflow-x-auto text-[11px] font-mono leading-relaxed whitespace-pre-wrap break-all ${color}`}>
        {code}
      </pre>
    </div>
  );
}

export function StageDelivery({ project, onRunAction, busy }) {
  const delivery = project?.delivery || {};
  const reportUrl = atlasService.reportUrl(project.id);
  const winner = project?.winner_model || project?.results?.winner || (project?.models && project.models[0]);

  // Code Copy State
  const [copiedKey, setCopiedKey] = useState(null);

  // Dynamic Live Prediction State
  const targetColName = (project?.target || project?.schema?.target || project?.dataset?.target || "").toLowerCase();
  const columns = project?.schema?.columns || project?.data_health?.columns || [];
  const predictorCols = columns.filter((c) => {
    const name = (c.name || c.column || "").toLowerCase();
    if (!name) return false;
    if (name === targetColName) return false;
    if (["id", "index", "_id", "id_col"].includes(name)) return false;
    return true;
  });

  const [inputValues, setInputValues] = useState({});
  const [predResult, setPredResult] = useState(null);
  const [predBusy, setPredBusy] = useState(false);
  const [predError, setPredError] = useState("");

  const copyToClipboard = (text, key) => {
    navigator.clipboard.writeText(text);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 2000);
  };

  const predictEndpoint = `${window.location.origin}/api/product/projects/${project?.id}/predict`;

  // Code Snippets for Deployment
  const samplePayload = Object.keys(inputValues).length > 0
    ? inputValues
    : Object.fromEntries(predictorCols.slice(0, 5).map((c) => [c.name || c.column, 1.0]));

  const escapedJsonPayload = JSON.stringify(samplePayload).replace(/"/g, '\\"');

  const pythonCode = `import requests

url = "${predictEndpoint}"
payload = ${JSON.stringify(samplePayload, null, 4)}

response = requests.post(url, json=payload)
print(response.json())`;

  const curlCode = `# Linux / macOS / PowerShell
curl -X POST "${predictEndpoint}" \\
  -H "Content-Type: application/json" \\
  -d '${JSON.stringify(samplePayload)}'

# Windows Command Prompt (cmd.exe)
curl -X POST "${predictEndpoint}" -H "Content-Type: application/json" -d "${escapedJsonPayload}"`;

  const dockerCode = `# Build and run the Atlas model microservice
docker build -t atlas-project-${project?.id?.slice(0, 8)} .
docker run -d -p 8000:8000 --name atlas-ml-service \\
  atlas-project-${project?.id?.slice(0, 8)}

# Test the deployed endpoint
curl -X POST "http://localhost:8000/predict" \\
  -H "Content-Type: application/json" \\
  -d '${JSON.stringify(samplePayload)}'`;

  // Run Live Prediction
  const handlePredict = async (e) => {
    e.preventDefault();
    setPredBusy(true);
    setPredError("");
    setPredResult(null);
    try {
      const payload = { ...inputValues };
      predictorCols.forEach((c) => {
        const name = c.name || c.column;
        if (payload[name] === undefined || payload[name] === "") {
          const inputType = getInputType(c);
          payload[name] = inputType === "number" ? 1.0 : "sample";
        }
      });
      const res = await atlasService.predictProject(project.id, payload);
      setPredResult(res);
    } catch (err) {
      setPredError(err.message);
    } finally {
      setPredBusy(false);
    }
  };

  // Determine input type per column
  const getInputType = (col) => {
    const dtype = (col.data_type || col.dtype || "").toLowerCase();
    if (dtype.includes("int") || dtype.includes("float") || dtype.includes("num")) return "number";
    return "text";
  };

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="rounded-2xl border border-white/10 bg-slate-900/90 p-6 shadow-xl backdrop-blur-xl">
        <span className="rounded-full bg-cyan-500/10 px-3 py-1 font-mono text-xs font-bold uppercase tracking-wider text-cyan-400 ring-1 ring-cyan-500/20">
          Stage 6 • Share, Launch & Exporter
        </span>
        <h2 className="mt-2 text-2xl font-black text-white">
          Production Deployment & Executive Delivery
        </h2>
        <p className="mt-1 text-xs text-slate-400">
          Export automated FastAPI microservices, ONNX model weights, executive HTML reports, and test real-time model inference.
        </p>

        {/* Executive Report Download */}
        <div className="mt-5 flex flex-wrap gap-3 border-t border-white/10 pt-4">
          <a
            href={reportUrl}
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-emerald-500 to-teal-600 px-5 py-2.5 text-xs font-extrabold text-white shadow-lg shadow-emerald-500/25 transition-all hover:brightness-110 active:scale-95"
          >
            <FileText className="h-4 w-4" />
            <span>Open Executive HTML Report</span>
          </a>

          {winner && (
            <div className="flex items-center gap-2 rounded-xl border border-white/10 bg-slate-950/60 px-4 py-2.5 text-xs font-semibold text-slate-300">
              <CheckCircle2 className="h-4 w-4 text-emerald-400" />
              <span>Winner: <span className="text-white font-bold">{winner.model_name || winner.classifier || winner.name || "Gradient Boosting Classifier"}</span></span>
              {winner.metric_value != null && (
                <span className="font-mono text-cyan-400 font-bold">
                  {winner.metric_value.toFixed(4)}
                </span>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Dynamic Model Inference Playground */}
      <div className="rounded-2xl border border-white/10 bg-slate-900/80 p-6 shadow-xl backdrop-blur-xl">
        <div className="mb-4 flex items-center justify-between">
          <h3 className="text-base font-extrabold text-white flex items-center gap-2">
            <Sliders className="h-4 w-4 text-cyan-400" />
            <span>Live Model Inference Playground</span>
          </h3>
          <span className="rounded bg-cyan-500/10 px-2 py-0.5 font-mono text-[10px] font-bold text-cyan-300 border border-cyan-500/20">
            Winner Model Active
          </span>
        </div>

        {predictorCols.length === 0 ? (
          <div className="flex items-center gap-3 rounded-xl border border-white/10 bg-slate-950/60 p-4">
            <AlertCircle className="h-4 w-4 text-slate-500 flex-shrink-0" />
            <p className="text-xs text-slate-400">
              No feature columns detected. Complete data profiling to enable live inference.
            </p>
          </div>
        ) : (
          <form onSubmit={handlePredict} className="space-y-4">
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {predictorCols.slice(0, 9).map((col, idx) => {
                const name = col.name || col.column;
                const inputType = getInputType(col);
                return (
                  <div key={idx}>
                    <label className="block text-[11px] font-semibold text-slate-300 font-mono mb-1 truncate">
                      {name}
                      <span className="ml-1 text-slate-600">({col.data_type || col.dtype || "num"})</span>
                    </label>
                    <input
                      type={inputType}
                      step={inputType === "number" ? "any" : undefined}
                      placeholder={inputType === "number" ? "0.0" : "value..."}
                      value={inputValues[name] ?? ""}
                      onChange={(e) =>
                        setInputValues({
                          ...inputValues,
                          [name]: inputType === "number" ? parseFloat(e.target.value) || 0 : e.target.value
                        })
                      }
                      className="w-full rounded-lg border border-white/10 bg-slate-950 px-3 py-1.5 text-xs text-cyan-300 font-mono focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500/40 transition-colors"
                    />
                  </div>
                );
              })}
            </div>

            {predictorCols.length > 9 && (
              <p className="text-[11px] text-slate-500 font-mono">
                Showing 9 of {predictorCols.length} features. Edit payload in the code snippet below for full inference.
              </p>
            )}

            <div className="flex items-center justify-between border-t border-white/10 pt-3">
              {predResult ? (
                <div className="flex flex-wrap items-center gap-3 rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-xs font-bold text-emerald-300">
                  <CheckCircle2 className="h-4 w-4 text-emerald-400 flex-shrink-0" />
                  <span>
                    Prediction:{" "}
                    <strong className="font-mono text-white text-sm">
                      {JSON.stringify(predResult.prediction ?? predResult.result)}
                    </strong>
                  </span>
                  {predResult.confidence != null && (
                    <span className="font-mono text-[11px] text-cyan-300">
                      Confidence: {(predResult.confidence * 100).toFixed(1)}%
                    </span>
                  )}
                </div>
              ) : predError ? (
                <div className="flex items-center gap-2 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs font-semibold text-red-300">
                  <AlertCircle className="h-4 w-4 text-red-400" />
                  <span>{predError}</span>
                </div>
              ) : (
                <span className="text-xs text-slate-500 font-mono">
                  Fill feature values above to test model inference
                </span>
              )}

              <button
                type="submit"
                disabled={predBusy}
                className="flex-shrink-0 flex items-center gap-2 rounded-xl bg-cyan-500 px-5 py-2 text-xs font-extrabold text-slate-950 hover:bg-cyan-400 transition-all active:scale-95 disabled:opacity-50"
              >
                {predBusy ? <Zap className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4 fill-slate-950" />}
                <span>Execute Inference</span>
              </button>
            </div>
          </form>
        )}
      </div>

      {/* Code Snippets Copy Box */}
      <div className="rounded-2xl border border-white/10 bg-slate-900/80 p-6 shadow-xl backdrop-blur-xl space-y-4">
        <h3 className="text-base font-extrabold text-white flex items-center gap-2">
          <Code2 className="h-4 w-4 text-cyan-400" />
          <span>REST API Integration Snippets</span>
        </h3>

        <CodeBlock
          label="Python Request"
          code={pythonCode}
          copyKey="python"
          copiedKey={copiedKey}
          onCopy={copyToClipboard}
          color="text-cyan-300"
          icon={FileCode}
        />

        <CodeBlock
          label="cURL Command"
          code={curlCode}
          copyKey="curl"
          copiedKey={copiedKey}
          onCopy={copyToClipboard}
          color="text-amber-200"
          icon={Terminal}
        />

        <CodeBlock
          label="Docker Deployment"
          code={dockerCode}
          copyKey="docker"
          copiedKey={copiedKey}
          onCopy={copyToClipboard}
          color="text-emerald-300"
          icon={Code2}
        />
      </div>
    </div>
  );
}
