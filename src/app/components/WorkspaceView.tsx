"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import type { Session, PipelineEvent, NotebookData } from "../types";
import { usePipeline } from "../hooks/usePipeline";
import PipelineLog from "./PipelineLog";
import NotebookViewer from "./NotebookViewer";

interface Props {
  session: Session;
  setSession: (s: Session) => void;
  apiBase: string;
  onNewPipeline: () => void;
}

export default function WorkspaceView({ session, setSession, apiBase, onNewPipeline }: Props) {
  const [events, setEvents] = useState<PipelineEvent[]>([]);
  const [activeTab, setActiveTab] = useState<"log" | "notebook">("log");
  const [notebook, setNotebook] = useState<NotebookData | null>(null);
  const [leftWidth, setLeftWidth] = useState(45);
  const resizing = useRef(false);

  const onEvent = useCallback((event: PipelineEvent) => {
    setEvents((prev) => [...prev, event]);
    if (event.type === "stage") {
      setSession({ ...session, currentStage: event.name });
    }
  }, [session, setSession]);

  const onComplete = useCallback(async (event: PipelineEvent) => {
    if (event.type === "complete") {
      setSession({ ...session, status: "complete", summary: event.summary });
      try {
        const resp = await fetch(`${apiBase}/api/notebook-json/${session.id}`);
        if (resp.ok) {
          const nb = await resp.json();
          setNotebook(nb);
          setActiveTab("notebook");
        }
      } catch (e) {
        console.error("Failed to fetch notebook:", e);
      }
    } else if (event.type === "error") {
      setSession({ ...session, status: "error" });
    }
  }, [session, setSession, apiBase]);

  const { start, cancel, isRunning } = usePipeline({ session, apiBase, onEvent, onComplete });
  
  useEffect(() => {
    if (session.status === "uploaded" && !isRunning) {
      const t = setTimeout(start, 300);
      return () => clearTimeout(t);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Resize ────────────────────────────────────────────────────────────
  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!resizing.current) return;
      setLeftWidth(Math.max(20, Math.min(70, (e.clientX / window.innerWidth) * 100)));
    };
    const onUp = () => { resizing.current = false; };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => { window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
  }, []);

  // ── Save ──────────────────────────────────────────────────────────────
  const handleSave = async () => {
    if (typeof window !== 'undefined' && window.electronAPI) {
      await window.electronAPI.saveNotebook(session.id);
    } else {
      const a = document.createElement("a");
      a.href = `${apiBase}/api/notebook/${session.id}`;
      a.download = `${session.paperName}.ipynb`;
      a.click();
    }
  };

  const handleCancelAndNew = () => {
    cancel(); // Halt the backend generation
    onNewPipeline(); // Trigger the parent component's reset logic
  };

  const lastStage = events.filter((e) => e.type === "stage").pop();
  const lastStep = events.filter((e) => e.type === "step").pop();
  const stepLabel = lastStep ? `${lastStep.step}/${lastStep.total}` : isRunning ? "Starting..." : "";

  return (
    <div className="flex flex-1 flex-col overflow-hidden animate-fade-in">
      {/* ── Toolbar ──────────────────────────────────────────────────── */}
      <div className="flex h-12 shrink-0 items-center gap-3 border-b px-4"
           style={{ background: "var(--color-surface-1)", borderColor: "var(--color-border)" }}>
        <span className="flex-1 truncate text-sm font-semibold">Jupy<span style={{ color: "var(--color-accent)" }}>Paper</span>   -   {session.paperName}</span>

        {isRunning && lastStage && (
          <span className="font-mono text-xs" style={{ color: "var(--color-accent)" }}>
            {lastStage.name} {stepLabel && `(${stepLabel})`}
          </span>
        )}

        {session.status === "complete" && (
          <button onClick={handleSave}
            className="flex cursor-pointer items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs font-medium text-white transition-colors"
            style={{ background: "var(--color-accent-dim)", borderColor: "var(--color-accent)" }}>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="7 10 12 15 17 10" />
              <line x1="12" y1="15" x2="12" y2="3" />
            </svg>
            Save .ipynb
          </button>
        )}

        <button onClick={handleCancelAndNew}
          className="flex cursor-pointer items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs font-medium transition-colors"
          style={{ background: "var(--color-surface-2)", borderColor: "var(--color-border)", color: "var(--color-muted-fg)" }}>
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" />
          </svg>
          Cancel and Start New
        </button>
      </div>

      {/* ── Split panels ─────────────────────────────────────────────── */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left: PDF */}
        <div className="flex flex-col overflow-hidden" style={{ width: `${leftWidth}%`, flexShrink: 0 }}>
          <div className="flex h-9 shrink-0 items-center gap-2 border-b px-3.5"
               style={{ background: "var(--color-surface-1)", borderColor: "var(--color-border)" }}>
            <span className="text-[11px] font-semibold uppercase tracking-widest" style={{ color: "var(--color-muted)" }}>
              PDF Preview
            </span>
          </div>
          <div className="flex-1 overflow-hidden" style={{ background: "var(--color-surface-0)" }}>
            <iframe src={`${session.pdfUrl}#toolbar=1&navpanes=0`} title="PDF"
              className="h-full w-full border-0" style={{ background: "white" }} />
          </div>
        </div>

        {/* Resize handle */}
        <div className="resize-handle w-1 shrink-0" onMouseDown={() => { resizing.current = true; }} />

        {/* Right: Pipeline / Notebook */}
        <div className="flex flex-1 flex-col overflow-hidden">
          {/* Tab bar */}
          <div className="flex shrink-0 border-b"
               style={{ background: "var(--color-surface-1)", borderColor: "var(--color-border)" }}>
            <button
              className={`cursor-pointer border-b-2 px-5 py-2 text-xs font-medium transition-colors ${activeTab === "log" ? "" : "border-transparent"}`}
              style={{
                borderColor: activeTab === "log" ? "var(--color-accent)" : "transparent",
                color: activeTab === "log" ? "var(--color-accent)" : "var(--color-muted)",
              }}
              onClick={() => setActiveTab("log")}
            >
              Pipeline Log
              {isRunning && (
                <span className="ml-1.5 inline-block h-1.5 w-1.5 rounded-full animate-dot-pulse"
                      style={{ background: "var(--color-accent)" }} />
              )}
            </button>
            <button
              className={`cursor-pointer border-b-2 px-5 py-2 text-xs font-medium transition-colors ${activeTab === "notebook" ? "" : "border-transparent"}`}
              style={{
                borderColor: activeTab === "notebook" ? "var(--color-accent)" : "transparent",
                color: activeTab === "notebook" ? "var(--color-accent)" : "var(--color-muted)",
                opacity: notebook ? 1 : 0.4,
              }}
              onClick={() => notebook && setActiveTab("notebook")}
            >
              Notebook {notebook && <span style={{ color: "var(--color-success)" }}>✓</span>}
            </button>
          </div>

          {/* Content */}
          <div className="flex-1 overflow-y-auto">
            {activeTab === "log" && <PipelineLog events={events} isRunning={isRunning} session={session} />}
            {activeTab === "notebook" && notebook && <NotebookViewer notebook={notebook} />}
          </div>
        </div>
      </div>
    </div>
  );
}