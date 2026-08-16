const configuredUrl = import.meta.env.VITE_ATLAS_API_URL?.replace(/\/$/, "");
const API_BASE = configuredUrl || "/api";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });

  const data = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(data?.detail || `Service returned ${response.status}`);
  }
  return data;
}

export const atlasService = {
  health: () => request("/health"),
  info: () => request("/info"),
  init: () => request("/workspace/init", { method: "POST" }),

  // Runs browser
  listRuns: () => request("/runs"),
  getRun: (runId) => request(`/runs/${encodeURIComponent(runId)}`),
  downloadRunArtifactUrl: (runId, filename) => `${API_BASE}/runs/${encodeURIComponent(runId)}/files/${encodeURIComponent(filename)}`,

  // Dataset upload
  uploadDataset: async (file) => {
    const form = new FormData();
    form.append("file", file);
    const response = await fetch(`${API_BASE}/datasets/upload`, { method: "POST", body: form });
    const data = await response.json().catch(() => null);
    if (!response.ok) throw new Error(data?.detail || "Upload failed");
    return data;
  },

  // Raw CLI command proxy
  command: (command, body = {}) => request(`/atlas/${command}`, { method: "POST", body: JSON.stringify(body) }),

  // Product API
  listProjects: () => request("/product/projects"),
  getProject: (projectId) => request(`/product/projects/${projectId}`),
  deleteProject: (projectId) => request(`/product/projects/${encodeURIComponent(projectId)}`, { method: "DELETE" }),
  updateProject: (projectId, body) => request(`/product/projects/${encodeURIComponent(projectId)}`, { method: "PATCH", body: JSON.stringify(body) }),
  createProject: async ({ file, name, goal, target }) => {
    const form = new FormData();
    form.append("file", file);
    form.append("name", name);
    form.append("goal", goal);
    form.append("target", target);
    const response = await fetch(`${API_BASE}/product/projects`, { method: "POST", body: form });
    const data = await response.json().catch(() => null);
    if (!response.ok) throw new Error(data?.detail || "We couldn't create that project.");
    return data;
  },
  projectAction: (projectId, action, body = {}) => request(`/product/projects/${projectId}/${action}`, { method: "POST", body: JSON.stringify(body) }),

  // Delivery & dataset downloads
  reportUrl: (projectId) => `${API_BASE}/product/projects/${encodeURIComponent(projectId)}/report`,
  deploymentFileUrl: (projectId, filename) => `${API_BASE}/product/projects/${encodeURIComponent(projectId)}/deployment/${encodeURIComponent(filename)}`,
  exportDatasetUrl: (projectId, filename) => `${API_BASE}/product/projects/${encodeURIComponent(projectId)}/exports/${encodeURIComponent(filename)}`,
  cleanedDatasetUrl: (projectId) => `${API_BASE}/product/projects/${encodeURIComponent(projectId)}/cleaned`,
  shapPlotUrl: (projectId, plotName) => `${API_BASE}/product/projects/${encodeURIComponent(projectId)}/shap/${encodeURIComponent(plotName)}`,

  // Project-specific Dynamic Inference
  predictProject: (projectId, payload) =>
    request(`/product/projects/${encodeURIComponent(projectId)}/predict`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  // Global Inference
  predict: (record) => request("/predict", { method: "POST", body: JSON.stringify(record) }),
  predictBatch: (instances) =>
    request("/predict_batch", { method: "POST", body: JSON.stringify({ instances }) }),
};

export { API_BASE };
