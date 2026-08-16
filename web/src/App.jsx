import React, { useEffect, useState } from "react";
import { API_BASE, atlasService } from "./api";
import { Header } from "./components/Header";
import { Sidebar } from "./components/Sidebar";
import { ProjectCard } from "./components/ProjectCard";
import { CreateProjectView } from "./components/CreateProjectView";
import { StageDataHealth } from "./components/StageDataHealth";
import { StageCleaning } from "./components/StageCleaning";
import { StageModelPlan } from "./components/StageModelPlan";
import { StageTraining } from "./components/StageTraining";
import { StageExplainability } from "./components/StageExplainability";
import { StageDelivery } from "./components/StageDelivery";
import { AtlasAssistant } from "./components/AtlasAssistant";
import { RunsHistory } from "./components/RunsHistory";
import { 
  Sparkles, 
  Plus, 
  Boxes, 
  Search, 
  CheckCircle2, 
  AlertCircle, 
  X,
  Bot
} from "lucide-react";

export default function App() {
  const [projects, setProjects] = useState([]);
  const [project, setProject] = useState(null);
  const [screen, setScreen] = useState("home"); // 'home' | 'new' | 'project' | 'runs'
  const [activeStage, setActiveStage] = useState("data_health");

  // Create Project Form State
  const [file, setFile] = useState(null);
  const [name, setName] = useState("");
  const [goal, setGoal] = useState("");
  const [target, setTarget] = useState("");
  const [testSize, setTestSize] = useState("0.2");
  const [outlierAction, setOutlierAction] = useState("report");
  const [anomalyAction, setAnomalyAction] = useState("report");

  // UI Status State
  const [busy, setBusy] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [assistantOpen, setAssistantOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");

  // Refresh Projects
  const refreshProjects = async (selectId) => {
    try {
      const entries = await atlasService.listProjects();
      setProjects(entries || []);
      const selected = selectId || project?.id;
      if (selected) {
        const found = entries.find((e) => e.id === selected);
        if (found) {
          setProject(found);
          if (found.stage) setActiveStage(found.stage);
        }
      }
    } catch (e) {
      setError(e.message);
    }
  };

  useEffect(() => {
    refreshProjects();
  }, []);

  // Handle Project Creation
  const handleCreateProject = async (event) => {
    event.preventDefault();
    if (!file) return setError("Choose a CSV, JSON, or Parquet dataset file.");
    if (!goal.trim()) return setError("Describe your prediction/analysis goal.");
    setBusy("create");
    setError("");
    try {
      const created = await atlasService.createProject({ file, name, goal, target });
      setProject(created);
      setActiveStage(created.stage || "data_health");
      setScreen("project");
      setMessage("Project workspace initialized successfully.");
      await refreshProjects(created.id);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy("");
    }
  };

  // Delete Project
  const handleDeleteProject = async (projectId, e) => {
    e?.stopPropagation();
    if (!window.confirm("Are you sure you want to delete this workspace project?")) return;
    try {
      await atlasService.deleteProject(projectId);
      if (project?.id === projectId) {
        setProject(null);
        setScreen("home");
      }
      await refreshProjects();
      setMessage("Workspace removed.");
    } catch (err) {
      setError(err.message);
    }
  };

  // Run Stage Action
  const handleRunAction = async (action, body = {}) => {
    if (!project) return;
    setBusy(action);
    setError("");
    try {
      const response = await atlasService.projectAction(project.id, action, body);
      setProject(response.project);
      if (response.project?.stage) {
        setActiveStage(response.project.stage);
      }
      setMessage(response.message);
      await refreshProjects(project.id);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy("");
    }
  };

  // Filter projects by search query
  const filteredProjects = projects.filter((p) => {
    const q = searchQuery.toLowerCase();
    return (
      (p.name && p.name.toLowerCase().includes(q)) ||
      (p.goal && p.goal.toLowerCase().includes(q)) ||
      (p.target && p.target.toLowerCase().includes(q)) ||
      (p.id && p.id.toLowerCase().includes(q))
    );
  });

  return (
    <div className="min-h-screen bg-[#07090e] text-slate-100 font-sans selection:bg-cyan-500 selection:text-slate-950">
      {/* Ambient Background Orbs */}
      <div className="ambient-bg">
        <div className="orb orb-1" />
        <div className="orb orb-2" />
        <div className="orb orb-3" />
      </div>

      {/* Top Header */}
      <Header
        projects={projects}
        currentProject={project}
        onSelectProject={(p) => {
          setProject(p);
          if (p.stage) setActiveStage(p.stage);
        }}
        screen={screen}
        setScreen={setScreen}
        onRefresh={() => refreshProjects()}
        busy={busy}
      />

      {/* Global Toast Notification - top-right so it doesn't overlap FAB */}
      {(message || error) && (
        <div className="fixed top-20 right-4 z-50 flex max-w-sm items-center justify-between gap-3 rounded-2xl border border-white/10 bg-slate-900/90 p-4 shadow-2xl backdrop-blur-xl">
          <div className="flex items-center gap-3">
            {error ? (
              <AlertCircle className="h-4 w-4 text-red-400 flex-shrink-0" />
            ) : (
              <CheckCircle2 className="h-4 w-4 text-emerald-400 flex-shrink-0" />
            )}
            <span className={`text-xs font-semibold ${error ? "text-red-300" : "text-emerald-300"}`}>
              {error || message}
            </span>
          </div>
          <button
            onClick={() => {
              setMessage("");
              setError("");
            }}
            className="ml-2 flex-shrink-0 text-slate-400 hover:text-white transition-colors"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      )}

      {/* Floating AI Assistant FAB (Bottom Right) */}
      <button
        onClick={() => setAssistantOpen(true)}
        className="fixed bottom-6 right-6 z-40 flex items-center gap-2.5 rounded-full bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 px-4 py-3 text-xs font-extrabold text-white shadow-2xl shadow-purple-500/40 ring-1 ring-white/20 transition-all hover:scale-105 active:scale-95"
        title="Open Atlas AI Assistant"
      >
        <Bot className="h-4 w-4 text-cyan-300" />
        <span className="hidden sm:inline">Ask Atlas AI</span>
      </button>

      {/* Main Body Routing */}
      {screen === "home" && (
        <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
          {/* Workspace Hero Header */}
          <div className="mb-8 flex flex-col justify-between gap-4 border-b border-white/10 pb-6 sm:flex-row sm:items-end">
            <div>
              <div className="inline-flex items-center gap-2 rounded-full border border-cyan-500/30 bg-cyan-500/10 px-3 py-1 text-xs font-bold text-cyan-400">
                <Sparkles className="h-3.5 w-3.5" />
                <span>Enterprise Workspace Console</span>
              </div>
              <h1 className="mt-2 text-3xl font-black text-white sm:text-4xl">
                AutoML Active Workspaces
              </h1>
              <p className="mt-1 text-xs text-slate-400">
                Select an existing machine learning project workspace or create a new experiment pipeline.
              </p>
            </div>

            <button
              onClick={() => setScreen("new")}
              className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 px-5 py-2.5 text-xs font-extrabold text-white shadow-lg shadow-cyan-500/25 hover:brightness-110 active:scale-95 transition-all"
            >
              <Plus className="h-4 w-4" />
              <span>Create New Project</span>
            </button>
          </div>

          {/* Workspace Search & Filter Bar */}
          <div className="mb-6 flex items-center justify-between gap-4">
            <div className="relative w-full max-w-md">
              <Search className="pointer-events-none absolute left-3.5 top-2.5 h-4 w-4 text-slate-500" />
              <input
                type="text"
                placeholder="Search projects by name, target, or goal..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full rounded-xl border border-white/10 bg-slate-900/90 pl-10 pr-4 py-2 text-xs text-white placeholder-slate-500 focus:border-cyan-500 focus:outline-none"
              />
            </div>
            <span className="font-mono text-xs text-slate-500">
              {filteredProjects.length} {filteredProjects.length === 1 ? "Project" : "Projects"}
            </span>
          </div>

          {/* Bento-Grid Project Cards */}
          {filteredProjects.length === 0 ? (
            <div className="flex flex-col items-center justify-center rounded-2xl border border-dashed border-white/15 bg-slate-900/40 py-16 text-center">
              <Boxes className="h-12 w-12 text-slate-600 mb-3" />
              <h3 className="text-base font-bold text-slate-300">No Active Projects Found</h3>
              <p className="mt-1 text-xs text-slate-500 max-w-md">
                Get started by creating your first autonomous machine learning workspace.
              </p>
              <button
                onClick={() => setScreen("new")}
                className="mt-4 flex items-center gap-2 rounded-xl bg-cyan-500 px-4 py-2 text-xs font-extrabold text-slate-950 hover:bg-cyan-400 transition-all"
              >
                <Plus className="h-4 w-4" />
                <span>Launch New Experiment</span>
              </button>
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
              {filteredProjects.map((p) => (
                <ProjectCard
                  key={p.id}
                  project={p}
                  onOpen={(p) => {
                    setProject(p);
                    if (p.stage) setActiveStage(p.stage);
                    setScreen("project");
                  }}
                  onDelete={handleDeleteProject}
                />
              ))}
            </div>
          )}
        </main>
      )}

      {screen === "new" && (
        <CreateProjectView
          file={file}
          setFile={setFile}
          name={name}
          setName={setName}
          goal={goal}
          setGoal={setGoal}
          target={target}
          setTarget={setTarget}
          testSize={testSize}
          setTestSize={setTestSize}
          outlierAction={outlierAction}
          setOutlierAction={setOutlierAction}
          anomalyAction={anomalyAction}
          setAnomalyAction={setAnomalyAction}
          onSubmit={handleCreateProject}
          busy={busy}
          onCancel={() => setScreen("home")}
        />
      )}

      {screen === "project" && project && (
        <div className="relative flex min-h-[calc(100vh-4rem)] flex-col lg:flex-row">
          {/* Stage Sidebar */}
          <Sidebar
            project={project}
            activeStage={activeStage}
            onSelectStage={(stageId) => setActiveStage(stageId)}
            onRunAction={handleRunAction}
            onDeleteProject={handleDeleteProject}
            busy={busy}
          />

          {/* Main Active Stage Workspace */}
          <main className="relative flex-1 overflow-auto p-4 lg:p-8">
            {activeStage === "data_health" && (
              <StageDataHealth project={project} onRunAction={handleRunAction} busy={busy} />
            )}
            {activeStage === "cleaning" && (
              <StageCleaning project={project} onRunAction={handleRunAction} busy={busy} />
            )}
            {activeStage === "model_plan" && (
              <StageModelPlan project={project} onRunAction={handleRunAction} busy={busy} />
            )}
            {activeStage === "training" && (
              <StageTraining project={project} onRunAction={handleRunAction} busy={busy} />
            )}
            {activeStage === "explainability" && (
              <StageExplainability project={project} onRunAction={handleRunAction} busy={busy} />
            )}
            {activeStage === "delivery" && (
              <StageDelivery project={project} onRunAction={handleRunAction} busy={busy} />
            )}
          </main>
        </div>
      )}

      {screen === "runs" && (
        <RunsHistory onBack={() => setScreen("home")} />
      )}

      {/* Atlas AI Drawer Assistant */}
      <AtlasAssistant
        isOpen={assistantOpen}
        onClose={() => setAssistantOpen(false)}
        project={project}
      />
    </div>
  );
}
