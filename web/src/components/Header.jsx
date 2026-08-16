import React from "react";
import {
  Boxes,
  PlusCircle,
  History,
  RefreshCw,
  ChevronDown,
  Database,
  Cpu
} from "lucide-react";

export function Header({
  projects = [],
  currentProject,
  onSelectProject,
  screen,
  setScreen,
  onRefresh,
  busy
}) {
  const navItems = [
    { id: "home", label: "Workspaces", icon: Boxes },
    { id: "new", label: "New Experiment", icon: PlusCircle },
    { id: "runs", label: "Audit Runs", icon: History },
  ];

  return (
    <header className="sticky top-0 z-40 w-full border-b border-white/10 bg-slate-950/80 backdrop-blur-xl">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">

        {/* Brand Logo */}
        <button
          onClick={() => setScreen("home")}
          className="flex items-center gap-3 transition-transform hover:scale-[1.01] active:scale-[0.99]"
        >
          <div className="relative flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-cyan-500 via-blue-600 to-indigo-600 shadow-md shadow-cyan-500/20 ring-1 ring-white/20">
            <Cpu className="h-4 w-4 text-white" />
            <div className="absolute -bottom-0.5 -right-0.5 h-2.5 w-2.5 rounded-full bg-emerald-400 ring-2 ring-slate-950" />
          </div>
          <div className="flex items-center gap-2">
            <span className="font-mono text-lg font-black tracking-wider text-white">
              ATLAS<span className="text-cyan-400">.AI</span>
            </span>
            <span className="hidden sm:inline rounded-full bg-cyan-500/10 px-2 py-0.5 font-mono text-[10px] font-bold text-cyan-400 ring-1 ring-cyan-500/30">
              v2.0
            </span>
          </div>
        </button>

        {/* Navigation Tabs */}
        <nav className="flex items-center gap-1 rounded-xl bg-slate-900/90 p-1 ring-1 ring-white/10">
          {navItems.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setScreen(id)}
              className={`flex items-center gap-2 rounded-lg px-3 py-1.5 text-xs font-semibold transition-all ${
                screen === id
                  ? "bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-md shadow-cyan-500/20"
                  : "text-slate-400 hover:bg-white/5 hover:text-white"
              }`}
            >
              <Icon className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">{label}</span>
            </button>
          ))}
        </nav>

        {/* Right Section: Workspace Switcher & Status */}
        <div className="flex items-center gap-2 sm:gap-3">
          {/* Active Project Dropdown */}
          {projects.length > 0 && (
            <div className="group relative hidden md:block">
              <select
                value={currentProject?.id || ""}
                onChange={(e) => {
                  const found = projects.find((p) => p.id === e.target.value);
                  if (found) {
                    onSelectProject(found);
                    setScreen("project");
                  }
                }}
                className="h-8 w-44 appearance-none rounded-lg border border-white/10 bg-slate-900/90 pl-8 pr-7 text-xs font-semibold text-slate-200 shadow-sm transition-colors hover:border-cyan-500/40 focus:border-cyan-400 focus:outline-none"
              >
                <option value="" disabled>
                  Switch Workspace...
                </option>
                {projects.map((p) => (
                  <option key={p.id} value={p.id} className="bg-slate-900 text-slate-200">
                    {p.name || p.id}
                  </option>
                ))}
              </select>
              <Database className="pointer-events-none absolute left-2.5 top-2 h-3.5 w-3.5 text-cyan-400" />
              <ChevronDown className="pointer-events-none absolute right-2 top-2 h-3.5 w-3.5 text-slate-400 transition-transform group-hover:text-cyan-400" />
            </div>
          )}

          {/* API Health Status Pill */}
          <div className="hidden lg:flex items-center gap-2 rounded-full border border-emerald-500/20 bg-emerald-500/10 px-3 py-1 text-xs font-medium text-emerald-400">
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
              <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-500" />
            </span>
            <span className="font-mono text-[11px]">ONLINE</span>
          </div>

          {/* Refresh Button */}
          <button
            onClick={onRefresh}
            disabled={busy === "refresh"}
            title="Refresh Workspace"
            className="flex h-8 w-8 items-center justify-center rounded-lg border border-white/10 bg-slate-900 text-slate-400 hover:border-cyan-500/40 hover:bg-slate-800 hover:text-white active:scale-95 transition-all disabled:opacity-50"
          >
            <RefreshCw
              className={`h-3.5 w-3.5 ${busy === "refresh" ? "animate-spin text-cyan-400" : ""}`}
            />
          </button>
        </div>
      </div>
    </header>
  );
}
