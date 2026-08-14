import { useEffect, useRef, useState } from "react";
import { API_BASE, atlasService } from "./api";

const STAGES = [
  ["data_health", "Data health", "Understand your data before building."],
  ["model_plan", "Model plan", "Choose the right approach for your goal."],
  ["training", "Training", "Try several models and select the strongest."],
  ["explainability", "Understand results", "See what influenced the recommendation."],
  ["delivery", "Share & launch", "Create a report or a ready-to-deploy service."],
];

function metric(value) { return typeof value === "number" ? value.toFixed(3) : "—"; }
function pct(value) { return typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "—"; }
function bytes(n) { if (!n) return "—"; if (n < 1024) return `${n} B`; if (n < 1048576) return `${(n/1024).toFixed(1)} KB`; return `${(n/1048576).toFixed(1)} MB`; }

/* ─── SVG ICONS ──────────────────────────────────────────────────────────────── */
const Icons = {
  logo: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>,
  arrowRight: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>,
  arrowLeft: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="19" y1="12" x2="5" y2="12"/><polyline points="12 19 5 12 12 5"/></svg>,
  check: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>,
  upload: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>,
  grid: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>,
  alertTriangle: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>,
  shield: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>,
  loader: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{animation:'spin-slow 1.2s linear infinite'}}><line x1="12" y1="2" x2="12" y2="6"/><line x1="12" y1="18" x2="12" y2="22"/><line x1="4.93" y1="4.93" x2="7.76" y2="7.76"/><line x1="16.24" y1="16.24" x2="19.07" y2="19.07"/><line x1="2" y1="12" x2="6" y2="12"/><line x1="18" y1="12" x2="22" y2="12"/><line x1="4.93" y1="19.07" x2="7.76" y2="16.24"/><line x1="16.24" y1="7.76" x2="19.07" y2="4.93"/></svg>,
  rocket: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z"/><path d="M12 15l-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z"/><path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0"/><path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5"/></svg>,
  download: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>,
  file: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>,
  fileCode: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><polyline points="10 12 8 14 10 16"/><polyline points="14 12 16 14 14 16"/></svg>,
  gitCommit: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="4"/><line x1="1.05" y1="12" x2="7" y2="12"/><line x1="17.01" y1="12" x2="22.96" y2="12"/></svg>,
  hash: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="4" y1="9" x2="20" y2="9"/><line x1="4" y1="15" x2="20" y2="15"/><line x1="10" y1="3" x2="8" y2="21"/><line x1="16" y1="3" x2="14" y2="21"/></svg>,
  repeat: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/></svg>,
  chevronDown: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="6 9 12 15 18 9"/></svg>,
  chevronRight: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="9 18 15 12 9 6"/></svg>,
  table: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="3" x2="9" y2="21"/></svg>,
  settings: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>,
  zap: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>,
  database: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>,
  send: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>,
  xCircle: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>,
  trendUp: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>,
  list: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg>,
};

/* ─── AMBIENT BACKGROUND ─────────────────────────────────────────────────────── */
function AmbientBg() { return <div className="ambient-bg" aria-hidden="true"><div className="orb orb-1"/><div className="orb orb-2"/><div className="orb orb-3"/></div>; }

/* ─── COLLAPSIBLE SECTION ────────────────────────────────────────────────────── */
function Collapsible({ title, icon, defaultOpen = false, children }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className={`collapsible ${open ? "open" : ""}`}>
      <button className="collapsible-toggle" onClick={() => setOpen(!open)}>
        <span className="collapsible-icon">{icon}</span>
        <span>{title}</span>
        <span className="collapsible-chevron">{open ? Icons.chevronDown : Icons.chevronRight}</span>
      </button>
      {open && <div className="collapsible-body">{children}</div>}
    </div>
  );
}

/* ─── MAIN APP ───────────────────────────────────────────────────────────────── */
export default function App() {
  const fileRef = useRef(null);
  const [projects, setProjects] = useState([]);
  const [project, setProject] = useState(null);
  const [screen, setScreen] = useState("home");
  const [file, setFile] = useState(null);
  const [name, setName] = useState("");
  const [goal, setGoal] = useState("");
  const [target, setTarget] = useState("");
  const [busy, setBusy] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [runs, setRuns] = useState([]);
  const [selectedRun, setSelectedRun] = useState(null);

  async function refreshProjects(selectId) {
    const entries = await atlasService.listProjects();
    setProjects(entries);
    const selected = selectId || project?.id;
    if (selected) setProject(entries.find((e) => e.id === selected) || null);
  }
  useEffect(() => { refreshProjects().catch((e) => setError(e.message)); }, []);

  async function createProject(event) {
    event.preventDefault();
    if (!file) return setError("Choose a data file to start your project.");
    if (!goal.trim()) return setError("Describe what you want to predict or understand.");
    setBusy("create"); setError("");
    try {
      const created = await atlasService.createProject({ file, name, goal, target });
      setProject(created); setScreen("project"); setMessage("Your project is ready. Start with the data health check.");
      await refreshProjects(created.id);
    } catch (e) { setError(e.message); } finally { setBusy(""); }
  }

  async function runAction(action, body = {}) {
    if (!project) return;
    setBusy(action); setError("");
    try {
      const response = await atlasService.projectAction(project.id, action, body);
      if (action === "ask") { setAnswer(response.answer); }
      else { setProject(response.project); setMessage(response.message); await refreshProjects(project.id); }
    } catch (e) { setError(e.message); } finally { setBusy(""); }
  }

  async function loadRuns() {
    try { const r = await atlasService.listRuns(); setRuns(r); } catch (e) { setError(e.message); }
  }
  async function loadRunDetail(runId) {
    try { const r = await atlasService.getRun(runId); setSelectedRun(r); } catch (e) { setError(e.message); }
  }

  if (screen === "new") return <NewProject file={file} fileRef={fileRef} name={name} goal={goal} target={target} busy={busy} error={error} setFile={setFile} setName={setName} setGoal={setGoal} setTarget={setTarget} onCancel={() => setScreen("home")} onSubmit={createProject} />;
  if (screen === "runs") return <RunsBrowser runs={runs} selectedRun={selectedRun} error={error} onLoad={loadRuns} onSelect={loadRunDetail} onBack={() => { setScreen("home"); setSelectedRun(null); }} />;
  if (!project || screen === "home") return <Home projects={projects} onNew={() => { setError(""); setScreen("new"); }} onSelect={(item) => { setProject(item); setScreen("project"); }} onRuns={() => { setScreen("runs"); loadRuns(); }} />;
  return <ProjectView project={project} message={message} error={error} busy={busy} question={question} answer={answer} setQuestion={setQuestion} onBack={() => setScreen("home")} onAction={runAction} />;
}

/* ─── LOGO ───────────────────────────────────────────────────────────────────── */
function Logo() { return <div className="logo"><div className="logo-icon">{Icons.logo}</div><span>atlas</span></div>; }

/* ─── HOME ───────────────────────────────────────────────────────────────────── */
function Home({ projects, onNew, onSelect, onRuns }) {
  return (
    <div className="product-shell home-shell">
      <AmbientBg />
      <header className="product-header">
        <Logo />
        <nav><a href="#product">Product</a><a href="#how">How it works</a><a onClick={onRuns} style={{cursor:'pointer'}}>Runs</a></nav>
        <button className="ghost">Sign in</button>
      </header>
      <main className="home-main">
        <section className="hero-product">
          <p className="kicker">AI-POWERED DATA SCIENCE</p>
          <h1>Turn your data into<br /><em>decisions.</em></h1>
          <p>Atlas helps your team go from a spreadsheet to a trustworthy model—without code, notebooks, or a data science degree.</p>
          <button className="primary large" onClick={onNew}>Start a new project {Icons.arrowRight}</button>
          <span className="hero-note">Bring a CSV, JSON, or Parquet file.</span>
        </section>
        <section className="how" id="how">
          <div><div className="step-num">01</div><h3>Tell Atlas your goal</h3><p>Describe the outcome you want to predict or understand.</p></div>
          <div><div className="step-num">02</div><h3>Atlas does the heavy lifting</h3><p>It checks your data, tests models, and explains its recommendation.</p></div>
          <div><div className="step-num">03</div><h3>Share with confidence</h3><p>Get a clear report and a production-ready model package.</p></div>
        </section>
        {projects.length > 0 && (
          <section className="recent">
            <div className="section-title"><div><p className="kicker">YOUR WORK</p><h2>Recent projects</h2></div><button onClick={onNew}>New project</button></div>
            <div className="project-grid">
              {projects.slice(0, 6).map((p) => (
                <button className="project-tile" key={p.id} onClick={() => onSelect(p)}>
                  <span className="tile-icon">{Icons.grid}</span>
                  <div><small>{p.stage?.replaceAll("_", " ") || "Getting started"}</small><h3>{p.name}</h3><p>{p.dataset?.name} · {p.dataset?.rows?.toLocaleString() || "—"} rows</p></div>
                  <span className="tile-arrow">{Icons.arrowRight}</span>
                </button>
              ))}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

/* ─── NEW PROJECT ────────────────────────────────────────────────────────────── */
function NewProject({ file, fileRef, name, goal, target, busy, error, setFile, setName, setGoal, setTarget, onCancel, onSubmit }) {
  return (
    <div className="product-shell setup-shell">
      <AmbientBg />
      <header className="product-header"><Logo /><button className="ghost" onClick={onCancel}>Cancel</button></header>
      <main className="setup-main">
        <div className="setup-copy">
          <p className="kicker">NEW PROJECT</p>
          <h1>What would you<br />like to <em>discover?</em></h1>
          <p>Start with a dataset and a plain-English goal. Atlas will guide you through the rest.</p>
          <div className="trust"><span className="trust-icon">{Icons.shield}</span> Your data stays in your Atlas workspace.</div>
        </div>
        <form className="setup-form" onSubmit={onSubmit}>
          <label>Project name<input value={name} onChange={(e)=>setName(e.target.value)} placeholder="e.g. Customer retention"/></label>
          <label>What do you want Atlas to help with?<textarea value={goal} onChange={(e)=>setGoal(e.target.value)} placeholder="e.g. Predict which customers are likely to leave next month"/></label>
          <label>Which column contains the outcome?<input value={target} onChange={(e)=>setTarget(e.target.value)} placeholder="Optional — Atlas can help identify it"/></label>
          <input ref={fileRef} type="file" className="hidden" accept=".csv,.json,.parquet" onChange={(e)=>setFile(e.target.files?.[0]||null)}/>
          <button type="button" className={`dropzone ${file?"chosen":""}`} onClick={()=>fileRef.current?.click()}>
            <span className="drop-icon">{file ? Icons.check : Icons.upload}</span>
            <strong>{file ? file.name : "Choose your data file"}</strong>
            <small>{file ? `${(file.size/1024/1024).toFixed(2)} MB ready to upload` : "CSV, JSON, or Parquet · up to your workspace limit"}</small>
          </button>
          {error && <p className="form-error">{error}</p>}
          <button className="primary full" disabled={busy==="create"}>{busy==="create" ? "Creating your project…" : "Create project"} <b>{Icons.arrowRight}</b></button>
        </form>
      </main>
    </div>
  );
}

/* ─── RUNS BROWSER ───────────────────────────────────────────────────────────── */
function RunsBrowser({ runs, selectedRun, error, onLoad, onSelect, onBack }) {
  useEffect(() => { onLoad(); }, []);
  return (
    <div className="product-shell home-shell">
      <AmbientBg />
      <header className="product-header"><Logo /><button className="ghost" onClick={onBack}>{Icons.arrowLeft} Back</button></header>
      <main className="home-main" style={{maxWidth: 960}}>
        <p className="kicker">POWER USER</p>
        <h1 style={{fontSize: 32, fontWeight: 800, letterSpacing: '-0.04em', marginBottom: 24}}>Runs Browser</h1>
        {error && <div className="toast error">{Icons.alertTriangle} {error}</div>}
        {!selectedRun ? (
          <div className="runs-list">
            {runs.length === 0 && <p style={{color:'var(--fg-dim)',fontSize:13}}>No CLI runs found. Run <code>atlas analyze</code> or create a project to see runs here.</p>}
            {runs.map((r) => (
              <button key={r.id} className="run-row" onClick={() => onSelect(r.id)}>
                <span className="run-id">{Icons.hash} {r.id}</span>
                <span className="run-meta">{r.dataset || "—"} · {r.artifacts?.length || 0} artifacts</span>
                <span className="tile-arrow">{Icons.arrowRight}</span>
              </button>
            ))}
          </div>
        ) : (
          <div>
            <button className="back" onClick={() => onSelect(null)} style={{marginBottom:16}}>{Icons.arrowLeft} <span>All runs</span></button>
            <h2 style={{fontSize:18,fontWeight:700,marginBottom:16}}>Run: {selectedRun.id}</h2>
            <div className="artifact-grid">
              {Object.entries(selectedRun.artifacts || {}).map(([name, val]) => (
                <div key={name} className="artifact-card">
                  <span className="artifact-icon">{name.endsWith('.json') ? Icons.fileCode : Icons.file}</span>
                  <strong>{name}</strong>
                  {val?.download && <a href={`${API_BASE}${val.download}`} target="_blank" rel="noopener noreferrer" className="artifact-dl">{Icons.download} Download</a>}
                  {typeof val === 'object' && !val?.download && <small>{Object.keys(val).length} keys</small>}
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

/* ─── PROJECT DASHBOARD ──────────────────────────────────────────────────────── */
function ProjectView({ project, message, error, busy, question, answer, setQuestion, onBack, onAction }) {
  const completed = project.completed || {};
  const currentIndex = STAGES.findIndex(([id]) => !completed[id]);
  const activeIndex = currentIndex < 0 ? STAGES.length - 1 : currentIndex;
  const stage = STAGES[activeIndex];

  return (
    <div className="product-shell project-shell">
      <aside className="product-sidebar">
        <Logo />
        <button className="back" onClick={onBack}>{Icons.arrowLeft} <span>All projects</span></button>
        <div className="side-project"><small>PROJECT</small><strong>{project.name}</strong><span>{project.dataset?.name}</span></div>
        <div className="stage-nav">
          {STAGES.map(([id, label], index) => (
            <button key={id} className={`${index === activeIndex ? "current" : ""} ${completed[id] ? "done" : ""}`} onClick={() => document.getElementById(id)?.scrollIntoView({ behavior: "smooth" })}>
              <i>{completed[id] ? Icons.check : index + 1}</i><span>{label}</span>
            </button>
          ))}
        </div>
        <div className="side-help">Need help?<br /><span>Atlas explains every step.</span></div>
      </aside>

      <main className="project-main">
        <header className="project-header">
          <div><span>Projects / {project.name}</span><h1>{stage[1]}</h1></div>
          <div className="status"><i></i> Atlas is ready</div>
        </header>

        {message && <div className="toast success">{Icons.check} {message}</div>}
        {error && <div className="toast error">{Icons.alertTriangle} {error}</div>}

        <section className="overview">
          <div>
            <p className="kicker">YOUR GOAL</p>
            <h2>{project.dataset?.goal}</h2>
            <p>{project.dataset?.name} {project.dataset?.rows ? `· ${project.dataset.rows.toLocaleString()} rows · ${project.dataset.columns} columns` : "is ready for review"}</p>
          </div>
          <div className="goal-status">
            <small>PROJECT PROGRESS</small>
            <strong>{Object.values(completed).filter(Boolean).length} of 5 steps complete</strong>
            <div><i style={{ width: `${(Object.values(completed).filter(Boolean).length / 5) * 100}%` }} /></div>
          </div>
        </section>

        {/* STEP 1: Data Health */}
        <ProductStage id="data_health" title="Know your data" eyebrow="STEP 1" done={completed.data_health} loading={busy === "data_health"} action="data_health" onAction={onAction}>
          <DataHealth data={project.data_health} dataset={project.dataset} />
        </ProductStage>

        {/* STEP 2: Plan */}
        <ProductStage id="model_plan" title="Choose the right approach" eyebrow="STEP 2" done={completed.model_plan} locked={!completed.data_health} loading={busy === "model_plan"} action="model_plan" onAction={onAction}>
          <PlanView plan={project.plan} />
        </ProductStage>

        {/* STEP 3: Training */}
        <ProductStage id="training" title="Find your best model" eyebrow="STEP 3" done={completed.training} locked={!completed.model_plan} loading={busy === "training"} action="training" onAction={onAction}>
          <Training results={project.results} />
        </ProductStage>

        {completed.training && (
          <button className="improve-strip" disabled={busy === "improve"} onClick={() => onAction("improve")}>
            {busy === "improve" ? "Atlas is improving your model…" : "Want Atlas to look for improvements?"}
            <span>Review model {Icons.arrowRight}</span>
          </button>
        )}

        {/* Critique Report (visible after review) */}
        {project.critique && <CritiqueReport critique={project.critique} />}

        {/* STEP 4: Explainability */}
        <ProductStage id="explainability" title="Understand the recommendation" eyebrow="STEP 4" done={completed.explainability} locked={!completed.training} loading={busy === "explainability"} action="explainability" onAction={onAction}>
          <Explain explanation={project.explanation} projectId={project.id} />
        </ProductStage>

        {/* STEP 5: Delivery */}
        <section id="delivery" className="delivery">
          <div>
            <p className="kicker">STEP 5</p>
            <h2>Ready to share or launch?</h2>
            <p>Turn your work into a polished report or a deployable prediction service.</p>
          </div>
          <div>
            <button className="secondary" disabled={!completed.training || busy === "report"} onClick={() => onAction("report")}>
              {project.delivery?.report_url ? "Open report" : busy === "report" ? "Preparing report…" : "Create report"}
            </button>
            {project.delivery?.report_url && <a href={atlasService.reportUrl(project.id)} target="_blank" rel="noopener noreferrer">View report</a>}
            <button className="primary" disabled={!completed.training || busy === "deployment"} onClick={() => onAction("deployment")}>
              {busy === "deployment" ? "Preparing…" : project.delivery?.deployment_ready ? "Package ready" : "Prepare for launch"} {Icons.rocket}
            </button>
          </div>
        </section>

        {/* Deployment Files */}
        {project.delivery?.deployment_files?.length > 0 && <DeploymentFiles files={project.delivery.deployment_files} projectId={project.id} />}

        {/* Reproducibility */}
        {project.reproducibility && <ReproducibilityPanel repro={project.reproducibility} busy={busy} onAction={onAction} />}

        {/* Predict Panel */}
        {completed.deployment && project.data_health?.columns?.length > 0 && <PredictPanel columns={project.data_health.columns} target={project.dataset?.target} />}

        {/* Ask Atlas */}
        <section className="ask-card">
          <div><p className="kicker">ASK ATLAS</p><h2>Questions about this project?</h2><p>Ask in plain English about your data, model, or results.</p></div>
          <div>
            <textarea value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="What makes this model a good choice?" />
            <button onClick={() => onAction("ask", { question })} disabled={!question.trim() || busy === "ask"}>{busy === "ask" ? "Thinking…" : "Ask"}</button>
          </div>
          {answer && <p className="answer">{answer}</p>}
        </section>
      </main>
    </div>
  );
}

/* ─── PRODUCT STAGE ──────────────────────────────────────────────────────────── */
function ProductStage({ id, eyebrow, title, done, locked, loading, action, onAction, children }) {
  return (
    <section id={id} className={`product-stage ${locked ? "locked" : ""}`}>
      <div className="stage-heading">
        <div>
          <p className="kicker">{eyebrow}{done && <span className="complete">{Icons.check} Completed</span>}</p>
          <h2>{title}</h2>
        </div>
        {!done && (
          <button className="primary" disabled={locked || loading} onClick={() => onAction(action)}>
            {locked ? "Complete earlier step" : loading ? <>Atlas is working… <span style={{display:'inline-flex'}}>{Icons.loader}</span></> : action === "training" ? <>Train my models {Icons.arrowRight}</> : <>Start {title.toLowerCase()} {Icons.arrowRight}</>}
          </button>
        )}
      </div>
      {done ? children : <div className="stage-placeholder">{locked ? "This step will unlock when the previous step is complete." : "Atlas will do this for you, then explain the results here."}</div>}
    </section>
  );
}

/* ─── DATA HEALTH (enhanced) ─────────────────────────────────────────────────── */
function DataHealth({ data, dataset }) {
  return (
    <div className="stage-content">
      <div className="data-stats">
        <div><small>ROWS</small><strong>{dataset.rows?.toLocaleString()}</strong></div>
        <div><small>COLUMNS</small><strong>{dataset.columns}</strong></div>
        <div><small>DATA HEALTH</small><strong>{data.severity === "WARNING" ? "Needs attention" : "Looking good"}</strong></div>
        <div><small>DUPLICATES</small><strong>{data.duplicate_rows ?? 0}</strong></div>
      </div>

      {/* Column Schema Table */}
      {data.columns?.length > 0 && (
        <Collapsible title={`Column schema (${data.columns.length} columns)`} icon={Icons.table} defaultOpen={false}>
          <div className="schema-table">
            <div className="schema-head"><span>Column</span><span>Type</span><span>Role</span></div>
            {data.columns.map((col) => (
              <div key={col.name || col.column_name} className="schema-row">
                <span>{col.name || col.column_name}</span>
                <span className="mono-val">{col.inferred_type || col.dtype || col.type || "—"}</span>
                <span className="mono-val">{col.role || "feature"}</span>
              </div>
            ))}
          </div>
        </Collapsible>
      )}

      {/* Target Distribution */}
      {data.target_distribution && (
        <Collapsible title="Target distribution" icon={Icons.trendUp} defaultOpen={false}>
          <div className="target-dist">
            {Object.entries(data.target_distribution).map(([label, count]) => (
              <div key={label} className="dist-bar-row">
                <span className="dist-label">{label}</span>
                <span className="dist-bar"><i style={{width: `${Math.min(100, (count / Math.max(...Object.values(data.target_distribution))) * 100)}%`}}/></span>
                <span className="dist-count">{count}</span>
              </div>
            ))}
          </div>
        </Collapsible>
      )}

      {data.risks?.length ? (
        <div className="recommendations">
          <h3>Atlas noticed</h3>
          {data.risks.slice(0, 5).map((risk, i) => (
            <div key={i}><span className="risk-icon">{Icons.alertTriangle}</span><p><strong>{risk.category}{risk.column ? ` in ${risk.column}` : ""}</strong>{risk.description}<span>{risk.recommendation}</span></p></div>
          ))}
        </div>
      ) : (
        <p className="all-good">{Icons.check} Atlas found no important data quality concerns.</p>
      )}
    </div>
  );
}

/* ─── PLAN (enhanced) ────────────────────────────────────────────────────────── */
function PlanView({ plan }) {
  const pre = plan.preprocessing || {};
  const fe = plan.feature_engineering || {};
  const ev = plan.evaluation || {};
  return (
    <div className="stage-content">
      <p className="narrative">{plan.reasoning}</p>
      <div className="plan-details">
        <div><small>TYPE OF PROBLEM</small><strong>{plan.task_type?.replaceAll("_", " ")}</strong></div>
        <div><small>OUTCOME</small><strong>{plan.target_column}</strong></div>
        <div><small>PRIMARY METRIC</small><strong>{ev.primary_metric?.replaceAll("_", " ") || "—"}</strong></div>
      </div>

      {/* Preprocessing Config */}
      {(pre.missing_strategy || pre.outlier_strategy || pre.scale_strategy) && (
        <Collapsible title="Preprocessing strategy" icon={Icons.settings} defaultOpen={true}>
          <div className="config-grid">
            {pre.missing_strategy && <div><small>MISSING VALUES</small><strong>{pre.missing_strategy}</strong></div>}
            {pre.outlier_strategy && <div><small>OUTLIER HANDLING</small><strong>{pre.outlier_strategy}</strong></div>}
            {pre.scale_strategy && <div><small>SCALING</small><strong>{pre.scale_strategy}</strong></div>}
            {pre.drop_columns?.length > 0 && <div><small>DROP COLUMNS (HIGH VIF)</small><strong style={{color:'var(--warning)'}}>{pre.drop_columns.join(", ")}</strong></div>}
          </div>
          {pre.notes && <p className="config-note">{pre.notes}</p>}
        </Collapsible>
      )}

      {/* Feature Engineering Config */}
      {(fe.categorical_strategy || fe.numeric_transforms?.length > 0) && (
        <Collapsible title="Feature engineering & synthesis" icon={Icons.zap} defaultOpen={true}>
          <div className="config-grid">
            {fe.numeric_transforms?.length > 0 && <div><small>NUMERIC TRANSFORMS</small><strong>{fe.numeric_transforms.join(", ")}</strong></div>}
            {fe.categorical_strategy && <div><small>CATEGORICAL ENCODING</small><strong>{fe.categorical_strategy}</strong></div>}
            {fe.datetime_features?.length > 0 && <div><small>DATETIME FEATURES</small><strong>{fe.datetime_features.join(", ")}</strong></div>}
            <div><small>INTERACTION FEATURES (TOP MI)</small><strong style={{color:'var(--accent)'}}>{fe.interaction_features ? "Yes — Top MI Pairs" : "No"}</strong></div>
            {fe.text_vectorizer && <div><small>TEXT VECTORIZER</small><strong>{fe.text_vectorizer}</strong></div>}
          </div>
        </Collapsible>
      )}

      {/* Evaluation Config */}
      {(ev.cv_strategy || ev.n_folds) && (
        <Collapsible title="Evaluation configuration & data split" icon={Icons.trendUp} defaultOpen={true}>
          <div className="config-grid">
            {ev.primary_metric && <div><small>PRIMARY METRIC</small><strong>{ev.primary_metric}</strong></div>}
            {ev.secondary_metrics?.length > 0 && <div><small>SECONDARY METRICS</small><strong>{ev.secondary_metrics.join(", ")}</strong></div>}
            {ev.cv_strategy && <div><small>CV STRATEGY</small><strong style={{color: ev.cv_strategy === 'timeseries_split' ? 'var(--warning)' : 'inherit'}}>{ev.cv_strategy} ({ev.n_folds} folds)</strong></div>}
            {ev.test_size && <div><small>TEST SPLIT</small><strong>{(ev.test_size * 100).toFixed(0)}%</strong></div>}
            <div><small>HANDLE IMBALANCE</small><strong style={{color: ev.handle_imbalance ? 'var(--warning)' : 'var(--fg-dim)'}}>{ev.handle_imbalance ? "Yes — SMOTE / class_weight" : "No"}</strong></div>
          </div>
        </Collapsible>
      )}

      <h3>Models Atlas will compare</h3>
      <div className="model-cards">
        {plan.models?.map((model) => (
          <article key={model.name}>
            <span className="priority-badge">{model.priority}</span>
            <strong>{model.name}</strong>
            {model.library && <span className="model-lib">{model.library}</span>}
            <p>{model.rationale}</p>
          </article>
        ))}
      </div>
    </div>
  );
}

/* ─── TRAINING (enhanced) ────────────────────────────────────────────────────── */
function Training({ results }) {
  const winner = results.winner;
  const failed = results.experiments?.filter((e) => e.status === "failed") || [];
  return (
    <div className="stage-content">
      {winner && (
        <div className="winner-product">
          <p className="kicker-label">ATLAS RECOMMENDS</p>
          <h3>{winner.model_name}</h3>
          <span>It balances strong performance, speed, and model size for your data.</span>
          <div className="winner-metric">
            <small>TEST {results.primary_metric?.replaceAll("_", " ")}</small>
            <strong>{metric(winner.primary_metric_test)}</strong>
          </div>
        </div>
      )}
      <div className="results-table">
        <div className="result-head">
          <span>Model</span><span>{results.primary_metric?.replaceAll("_", " ") || "Score"}</span><span>Accuracy</span><span>Duration</span><span>Status</span>
        </div>
        {results.experiments?.map((item) => (
          <div key={item.model_name} className={item.status === "failed" ? "row-failed" : ""}>
            <span>{item.model_name}{item.library && <small className="exp-lib">{item.library}</small>}</span>
            <strong>{item.status === "success" ? metric(item.metrics?.[results.primary_metric]) : "—"}</strong>
            <span>{item.status === "success" ? metric(item.metrics?.accuracy) : "—"}</span>
            <span className="mono-val">{item.duration_seconds ? `${item.duration_seconds.toFixed(1)}s` : "—"}</span>
            <span>{item.status === "success" ? <span className="status-ok">{Icons.check} OK</span> : <span className="status-fail">{Icons.xCircle} Fail</span>}</span>
          </div>
        ))}
      </div>

      {/* Rankings */}
      {results.rankings?.length > 0 && (
        <Collapsible title={`Rankings breakdown (${results.rankings.length} models)`} icon={Icons.trendUp} defaultOpen={false}>
          <div className="rankings-list">
            {results.rankings.map((r, i) => (
              <div key={r.model_name || i} className="ranking-row">
                <span className="rank-num">#{i + 1}</span>
                <strong>{r.model_name}</strong>
                {r.composite_score != null && <span className="mono-val">Score: {r.composite_score.toFixed(3)}</span>}
              </div>
            ))}
          </div>
        </Collapsible>
      )}

      {/* Failed experiments */}
      {failed.length > 0 && (
        <Collapsible title={`Failed experiments (${failed.length})`} icon={Icons.xCircle} defaultOpen={false}>
          {failed.map((f) => (
            <div key={f.model_name} className="failed-exp">
              <strong>{f.model_name}</strong>
              <p>{f.error_message || "Unknown error"}</p>
            </div>
          ))}
        </Collapsible>
      )}
    </div>
  );
}

/* ─── CRITIQUE REPORT ────────────────────────────────────────────────────────── */
function CritiqueReport({ critique }) {
  if (!critique?.diagnosis) return null;
  const diagnosisText = typeof critique.diagnosis === 'string'
    ? critique.diagnosis
    : typeof critique.diagnosis === 'object'
    ? (critique.diagnosis.overall_health || critique.diagnosis.diagnosis || JSON.stringify(critique.diagnosis))
    : String(critique.diagnosis);

  return (
    <section className="product-stage critique-section">
      <div className="stage-heading"><div><p className="kicker">AI REVIEWER<span className="complete">{Icons.check} Complete</span></p><h2>Model critique & refinement</h2></div></div>
      <div className="stage-content">
        <p className="narrative">{diagnosisText}</p>
        {critique.actions?.length > 0 && (
          <div className="critique-actions">
            <h3>Actions taken</h3>
            {critique.actions.map((a, i) => <div key={i} className="critique-action">{Icons.zap} <span>{typeof a === 'string' ? a : a.description || JSON.stringify(a)}</span></div>)}
          </div>
        )}
        {(Object.keys(critique.before_metrics || {}).length > 0 || Object.keys(critique.after_metrics || {}).length > 0) && (
          <div className="critique-delta">
            <div><small>BEFORE</small>{Object.entries(critique.before_metrics || {}).map(([k,v]) => <div key={k}><span>{k}</span><strong>{metric(v)}</strong></div>)}</div>
            <div className="delta-arrow">{Icons.arrowRight}</div>
            <div><small>AFTER</small>{Object.entries(critique.after_metrics || {}).map(([k,v]) => <div key={k}><span>{k}</span><strong className="delta-improved">{metric(v)}</strong></div>)}</div>
          </div>
        )}
        {critique.refined_model && <p className="refined-label">Refined model: <strong>{critique.refined_model}</strong></p>}
      </div>
    </section>
  );
}

/* ─── EXPLAIN (enhanced) ─────────────────────────────────────────────────────── */
function Explain({ explanation, projectId }) {
  const top = explanation.features?.[0]?.mean_abs_shap || 1;
  return (
    <div className="stage-content">
      {explanation.why_chosen && (
        <div className="why-chosen-card">
          <p className="kicker">WHY THIS MODEL</p>
          <p>{explanation.why_chosen}</p>
        </div>
      )}
      <p className="narrative">{explanation.narrative || `Atlas used ${explanation.model_name} to make this recommendation. These are the signals with the strongest influence.`}</p>
      <div className="feature-bars">
        {explanation.features?.slice(0, 8).map((f) => (
          <div key={f.feature_name}><span>{f.feature_name}</span><i><b style={{ width: `${(f.mean_abs_shap / top) * 100}%` }} /></i><strong>{f.mean_abs_shap.toFixed(3)}</strong></div>
        ))}
      </div>
      {/* SHAP plots */}
      {(explanation.has_shap_summary || explanation.has_shap_importance) && (
        <Collapsible title="SHAP visualizations" icon={Icons.trendUp} defaultOpen={false}>
          <div className="shap-plots">
            {explanation.has_shap_summary && <div><small>SHAP SUMMARY</small><img src={atlasService.shapPlotUrl(projectId, "shap_summary.png")} alt="SHAP Summary Plot" /></div>}
            {explanation.has_shap_importance && <div><small>FEATURE IMPORTANCE</small><img src={atlasService.shapPlotUrl(projectId, "shap_importance.png")} alt="SHAP Importance Plot" /></div>}
          </div>
        </Collapsible>
      )}
    </div>
  );
}

/* ─── DEPLOYMENT FILES ───────────────────────────────────────────────────────── */
function DeploymentFiles({ files, projectId }) {
  return (
    <section className="product-stage">
      <Collapsible title={`Deployment package (${files.length} files)`} icon={Icons.fileCode} defaultOpen={true}>
        <div className="deploy-files">
          {files.map((f) => (
            <a key={f.name} href={atlasService.deploymentFileUrl(projectId, f.name)} target="_blank" rel="noopener noreferrer" className="deploy-file-row">
              <span className="deploy-file-icon">{Icons.file}</span>
              <strong>{f.name}</strong>
              <span className="mono-val">{bytes(f.size)}</span>
              <span className="deploy-dl">{Icons.download}</span>
            </a>
          ))}
        </div>
      </Collapsible>
    </section>
  );
}

/* ─── REPRODUCIBILITY PANEL ──────────────────────────────────────────────────── */
function ReproducibilityPanel({ repro, busy, onAction }) {
  return (
    <section className="product-stage repro-section">
      <div className="stage-heading">
        <div><p className="kicker">REPRODUCIBILITY</p><h2>Experiment snapshot</h2></div>
        <button className="secondary" disabled={busy === "replay"} onClick={() => onAction("replay")}>
          {busy === "replay" ? <>Replaying… {Icons.loader}</> : <>{Icons.repeat} Replay experiment</>}
        </button>
      </div>
      <div className="stage-content">
        <div className="repro-grid">
          <div><small>DATASET SHA256</small><strong className="mono-val hash-val">{repro.dataset_sha256 ? repro.dataset_sha256.slice(0, 16) + "…" : "—"}</strong></div>
          <div><small>GIT COMMIT</small><strong className="mono-val">{repro.git_commit || "—"}</strong></div>
          <div><small>PYTHON VERSION</small><strong className="mono-val">{repro.python_version || "—"}</strong></div>
          <div><small>OS</small><strong className="mono-val">{repro.os_info || "—"}</strong></div>
          <div><small>RANDOM SEED</small><strong className="mono-val">{repro.random_seed ?? "—"}</strong></div>
          <div><small>SNAPSHOT CREATED</small><strong className="mono-val">{repro.created_at ? new Date(repro.created_at).toLocaleString() : "—"}</strong></div>
        </div>
        {repro.dependencies && Object.keys(repro.dependencies).length > 0 && (
          <Collapsible title={`Dependencies (${Object.keys(repro.dependencies).length})`} icon={Icons.list} defaultOpen={false}>
            <div className="deps-table">
              {Object.entries(repro.dependencies).map(([pkg, ver]) => (
                <div key={pkg}><span>{pkg}</span><span className="mono-val">{ver}</span></div>
              ))}
            </div>
          </Collapsible>
        )}
      </div>
    </section>
  );
}

/* ─── PREDICT PANEL ──────────────────────────────────────────────────────────── */
function PredictPanel({ columns, target }) {
  const [formValues, setFormValues] = useState({});
  const [prediction, setPrediction] = useState(null);
  const [predBusy, setPredBusy] = useState(false);
  const [predError, setPredError] = useState("");

  const featureCols = columns.filter((c) => {
    const name = c.name || c.column_name;
    return name && name !== target;
  });

  async function handlePredict(e) {
    e.preventDefault();
    setPredBusy(true); setPredError(""); setPrediction(null);
    try {
      const payload = {};
      featureCols.forEach((c) => {
        const name = c.name || c.column_name;
        const val = formValues[name] ?? "";
        const t = (c.inferred_type || c.dtype || c.type || "").toLowerCase();
        payload[name] = (t.includes("int") || t.includes("float") || t.includes("numeric")) ? Number(val) : val;
      });
      const result = await atlasService.predict(payload);
      setPrediction(result);
    } catch (err) { setPredError(err.message); } finally { setPredBusy(false); }
  }

  return (
    <section className="product-stage predict-section">
      <div className="stage-heading"><div><p className="kicker">INFERENCE</p><h2>Test predictions</h2></div></div>
      <div className="stage-content">
        <p className="narrative">Enter feature values to get a real-time prediction from your deployed model.</p>
        <form className="predict-form" onSubmit={handlePredict}>
          <div className="predict-fields">
            {featureCols.map((c) => {
              const name = c.name || c.column_name;
              const t = (c.inferred_type || c.dtype || c.type || "").toLowerCase();
              const isNum = t.includes("int") || t.includes("float") || t.includes("numeric");
              return (
                <label key={name}>
                  <span>{name} <small className="mono-val">{c.inferred_type || c.dtype || c.type || ""}</small></span>
                  <input type={isNum ? "number" : "text"} step="any" value={formValues[name] ?? ""} onChange={(e) => setFormValues({ ...formValues, [name]: e.target.value })} placeholder={isNum ? "0" : "value"} />
                </label>
              );
            })}
          </div>
          {predError && <p className="form-error">{predError}</p>}
          <button className="primary" type="submit" disabled={predBusy}>{predBusy ? "Predicting…" : <>{Icons.send} Predict</>}</button>
        </form>
        {prediction && (
          <div className="predict-result">
            <div><small>PREDICTION</small><strong className="pred-value">{String(prediction.prediction)}</strong></div>
            {prediction.probabilities && (
              <div><small>PROBABILITIES</small><strong className="mono-val">{prediction.probabilities.map((p) => p.toFixed(3)).join(", ")}</strong></div>
            )}
            <div><small>MODEL</small><strong className="mono-val">{prediction.model_name}</strong></div>
          </div>
        )}
      </div>
    </section>
  );
}
