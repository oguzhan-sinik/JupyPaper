"use client";

import { useEffect, useRef } from "react";
import type { PipelineEvent, Session } from "../types";

interface Props {
  events: PipelineEvent[];
  isRunning: boolean;
  session: Session;
}

function fmtTime(ts?: number): string {
  if (!ts) return "";
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export default function PipelineLog({ events, isRunning, session }: Props) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [events.length]);

  if (events.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-5"
           style={{ color: "var(--color-muted)" }}>
        <div className="h-12 w-12 rounded-full animate-spin-ring"
             style={{ border: "2px solid var(--color-surface-4)", borderTopColor: "var(--color-accent)" }} />
        <p className="text-sm">Starting pipeline...</p>
      </div>
    );
  }

  const lastStep = [...events].reverse().find((e) => e.type === "step");
  const progressPct = lastStep?.total ? ((lastStep.step ?? 0) / lastStep.total) * 100 : 0;

  return (
    <div className="flex flex-col gap-0.5 p-4 font-mono text-xs leading-relaxed">
      {/* Progress bar */}
      {isRunning && lastStep && (
        <div className="mb-3 flex items-center gap-2.5">
          <div className="h-0.5 flex-1 overflow-hidden rounded-full" style={{ background: "var(--color-surface-3)" }}>
            <div className="progress-gradient h-full rounded-full transition-all duration-500"
                 style={{ width: `${progressPct}%` }} />
          </div>
          <span className="text-[11px]" style={{ color: "var(--color-muted)" }}>
            {lastStep.step}/{lastStep.total}
          </span>
        </div>
      )}

      {/* Complete banner */}
      {session.status === "complete" && session.summary && (
        <div className="mb-4 flex items-center gap-3 rounded-lg p-3.5 animate-complete-pop"
             style={{ background: "rgba(52,211,153,0.06)", border: "1px solid rgba(52,211,153,0.2)" }}>
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full"
               style={{ background: "rgba(52,211,153,0.15)", color: "var(--color-success)" }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <polyline points="20 6 9 17 4 12" />
            </svg>
          </div>
          <div>
            <h3 className="text-sm font-semibold" style={{ color: "var(--color-success)" }}>Pipeline Complete</h3>
            <p className="font-mono text-[11px]" style={{ color: "var(--color-muted)" }}>
              {session.summary.total_cells} cells ({session.summary.code_cells} code, {session.summary.markdown_cells} md)
              {session.summary.figures_embedded > 0 && ` · ${session.summary.figures_embedded} figs`}
              {session.summary.total_cost != null && ` · $${session.summary.total_cost.toFixed(4)}`}
            </p>
          </div>
        </div>
      )}

      {/* Log entries */}
      {events.map((evt, i) => {
        const text = getText(evt);
        if (!text) return null;
        const cls = getClass(evt);
        const icon = getIcon(evt);
        return (
          <div key={i} className={`flex gap-2 py-0.5 animate-slide-in-log ${cls}`}>
            <span className="w-14 shrink-0 pt-0.5 text-[10px]" style={{ color: "var(--color-muted)" }}>
              {fmtTime(evt.timestamp)}
            </span>
            <span>
              {icon && <span className="mr-1.5">{icon}</span>}
              {text}
            </span>
          </div>
        );
      })}

      {/* Active indicator */}
      {isRunning && (
        <div className="flex items-center gap-2 py-1" style={{ color: "var(--color-muted)" }}>
          <span className="inline-block h-1.5 w-1.5 rounded-full animate-dot-pulse"
                style={{ background: "var(--color-accent)" }} />
          Processing...
        </div>
      )}

      <div ref={endRef} />
    </div>
  );
}

function getClass(e: PipelineEvent): string {
  if (e.type === "stage") return "text-[var(--color-accent)] font-semibold text-[13px] pt-3 pb-1";
  if (e.type === "step") return "text-[var(--color-success)] font-medium";
  if (e.type === "error") return "text-[var(--color-danger)]";
  if (e.type === "cost") return "text-[var(--color-muted)] text-[11px]";
  if (e.type === "log" && (e.level === "warning" || e.message?.includes("⚠"))) return "text-[var(--color-warning)]";
  if (e.type === "log" && e.level === "error") return "text-[var(--color-danger)]";
  return "text-[var(--color-muted-fg)]";
}

function getIcon(e: PipelineEvent): string {
  switch (e.type) {
    case "stage": return "▸";
    case "step": return "✓";
    case "error": return "✗";
    case "cell_start": return "◦";
    case "cell_done": return "●";
    case "cost": return "$";
    default: return "";
  }
}

function getText(e: PipelineEvent): string {
  switch (e.type) {
    case "stage": return `${e.name}${e.description ? ` — ${e.description}` : ""}`;
    case "step": return `${e.name} ${e.detail || ""} [${e.step}/${e.total}]`;
    case "log": return e.message || "";
    case "cell_start": return `Analyzing: ${e.title} (${e.cell_type})`;
    case "cell_done": return `Done: ${e.title}`;
    case "cost": return `${e.name}: ${e.tokens} tok, ${e.elapsed?.toFixed(1)}s, $${e.cost?.toFixed(6)}`;
    case "error": return e.message || "Unknown error";
    case "progress": return `${(e as PipelineEvent & { label?: string }).title || ""}: ${e.step}/${e.total}`;
    case "complete": return "";
    default: return "";
  }
}