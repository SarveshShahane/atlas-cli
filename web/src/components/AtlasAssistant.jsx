import React, { useState, useRef, useEffect } from "react";
import {
  Sparkles,
  Send,
  X,
  Bot,
  User,
  Code,
  Copy,
  Check,
  Zap
} from "lucide-react";
import { atlasService } from "../api";

export function AtlasAssistant({ isOpen, onClose, project }) {
  // All hooks must be at top level — before any conditional returns
  const [question, setQuestion] = useState("");
  const [generateCode, setGenerateCode] = useState(false);
  const [busy, setBusy] = useState(false);
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: `Hello! I am your **Atlas AI** Autonomous Data Science Assistant.\n\nAsk me anything about:\n• Dataset profiling & feature engineering\n• Model selection trade-offs & hyperparameters\n• SHAP feature importance & attribution\n• Custom Python data pipeline code`
    }
  ]);
  const [copiedIdx, setCopiedIdx] = useState(null);
  const messagesEndRef = useRef(null);

  // Auto-scroll to latest message
  useEffect(() => {
    if (isOpen) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, isOpen]);

  if (!isOpen) return null;

  const handleSend = async (e) => {
    e.preventDefault();
    if (!question.trim() || busy) return;

    const userMsg = question.trim();
    setQuestion("");
    setMessages((prev) => [...prev, { role: "user", content: userMsg }]);
    setBusy(true);

    try {
      if (project?.id) {
        const res = await atlasService.projectAction(project.id, "ask", {
          question: userMsg,
          generate_code: generateCode
        });
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: res.answer || res.message || "I analyzed your question."
          }
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content:
              "Please select an active **project workspace** to ask context-specific questions about your dataset and models."
          }
        ]);
      }
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `**Error:** ${err.message}` }
      ]);
    } finally {
      setBusy(false);
    }
  };

  const copyCode = (text, idx) => {
    navigator.clipboard.writeText(text);
    setCopiedIdx(idx);
    setTimeout(() => setCopiedIdx(null), 2000);
  };

  // Simple text renderer: detect ```code``` blocks
  const renderContent = (text) => {
    const parts = text.split(/(```[\s\S]*?```)/g);
    return parts.map((part, i) => {
      if (part.startsWith("```") && part.endsWith("```")) {
        const code = part.slice(3, -3).replace(/^[a-z]+\n/, "");
        return (
          <pre
            key={i}
            className="mt-2 overflow-x-auto rounded-lg bg-slate-900 p-3 text-[10px] font-mono text-cyan-300 border border-white/10"
          >
            {code}
          </pre>
        );
      }
      // Render **bold** and bullet lines
      return (
        <span key={i}>
          {part.split("\n").map((line, j) => (
            <span key={j} className="block">
              {line.split(/(\*\*[^*]+\*\*)/g).map((chunk, k) =>
                chunk.startsWith("**") && chunk.endsWith("**") ? (
                  <strong key={k} className="font-bold text-white">
                    {chunk.slice(2, -2)}
                  </strong>
                ) : (
                  chunk
                )
              )}
            </span>
          ))}
        </span>
      );
    });
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-end bg-slate-950/80 p-4 backdrop-blur-sm">
      <div className="flex h-[90vh] w-full max-w-xl flex-col rounded-2xl border border-white/10 bg-slate-900 shadow-2xl overflow-hidden">

        {/* Drawer Header */}
        <div className="flex items-center justify-between border-b border-white/10 bg-slate-950 px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 shadow-md shadow-purple-500/30">
              <Bot className="h-5 w-5 text-white" />
            </div>
            <div>
              <h3 className="text-sm font-extrabold text-white flex items-center gap-2">
                <span>Atlas AI Assistant</span>
                <span className="rounded bg-purple-500/20 px-1.5 py-0.5 text-[9px] font-bold text-purple-300 border border-purple-500/30">
                  LLM
                </span>
              </h3>
              <p className="text-[11px] text-slate-400 font-mono">
                Context:{" "}
                <span className="text-cyan-400">
                  {project?.name || "Global Workspace"}
                </span>
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="rounded-lg p-1.5 text-slate-400 hover:bg-white/10 hover:text-white transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Message Log */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`flex gap-3 text-xs leading-relaxed ${
                msg.role === "user" ? "justify-end" : "justify-start"
              }`}
            >
              {msg.role === "assistant" && (
                <div className="flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-lg bg-indigo-600/30 text-indigo-400 mt-1">
                  <Bot className="h-4 w-4" />
                </div>
              )}

              <div
                className={`max-w-[85%] rounded-2xl px-4 py-3 shadow-md ${
                  msg.role === "user"
                    ? "bg-cyan-600 text-white font-medium rounded-br-sm"
                    : "bg-slate-950 border border-white/10 text-slate-200 rounded-bl-sm"
                }`}
              >
                <div className="leading-relaxed">{renderContent(msg.content)}</div>

                {msg.role === "assistant" && (
                  <div className="mt-2 flex justify-end">
                    <button
                      onClick={() => copyCode(msg.content, idx)}
                      className="flex items-center gap-1 text-[10px] font-semibold text-slate-500 hover:text-cyan-300 transition-colors"
                    >
                      {copiedIdx === idx ? (
                        <Check className="h-3 w-3 text-emerald-400" />
                      ) : (
                        <Copy className="h-3 w-3" />
                      )}
                      <span>{copiedIdx === idx ? "Copied" : "Copy"}</span>
                    </button>
                  </div>
                )}
              </div>

              {msg.role === "user" && (
                <div className="flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-lg bg-cyan-600/30 text-cyan-300 mt-1">
                  <User className="h-4 w-4" />
                </div>
              )}
            </div>
          ))}

          {busy && (
            <div className="flex gap-3 text-xs text-slate-400">
              <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-indigo-600/30 text-indigo-400">
                <Sparkles className="h-4 w-4 animate-pulse" />
              </div>
              <div className="flex items-center gap-2 rounded-2xl rounded-bl-sm bg-slate-950 border border-white/10 px-4 py-3">
                <div className="flex gap-1">
                  <span className="h-1.5 w-1.5 rounded-full bg-indigo-400 animate-bounce" style={{ animationDelay: "0ms" }} />
                  <span className="h-1.5 w-1.5 rounded-full bg-indigo-400 animate-bounce" style={{ animationDelay: "150ms" }} />
                  <span className="h-1.5 w-1.5 rounded-full bg-indigo-400 animate-bounce" style={{ animationDelay: "300ms" }} />
                </div>
                <span className="italic text-slate-500">Atlas AI is reasoning...</span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Form Input */}
        <form
          onSubmit={handleSend}
          className="border-t border-white/10 bg-slate-950 p-4 space-y-3"
        >
          <label className="flex items-center gap-2 text-xs text-slate-400 font-mono cursor-pointer select-none w-fit">
            <input
              type="checkbox"
              checked={generateCode}
              onChange={(e) => setGenerateCode(e.target.checked)}
              className="rounded border-white/20 bg-slate-900 text-cyan-500 focus:ring-cyan-500 focus:ring-1"
            />
            <Code className="h-3.5 w-3.5 text-cyan-400" />
            <span>Request Python code snippet</span>
          </label>

          <div className="flex items-center gap-2">
            <input
              type="text"
              placeholder="Ask about features, models, SHAP, or request code..."
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              className="flex-1 rounded-xl border border-white/10 bg-slate-900 px-4 py-2.5 text-xs text-white placeholder-slate-500 focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500/50 transition-colors"
            />
            <button
              type="submit"
              disabled={busy || !question.trim()}
              className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-md shadow-cyan-500/20 hover:brightness-110 active:scale-95 disabled:opacity-40 transition-all"
            >
              <Send className="h-4 w-4" />
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
