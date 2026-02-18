"use client";

import { type ReactNode, useState, useEffect } from "react";
import dynamic from "next/dynamic";
import type { NotebookData, NotebookCell } from "../types";

// Import JupyterNotebook with SSR disabled to avoid localStorage errors in Electron
const JupyterNotebook = dynamic(() => import("react-jupyter-notebook"), {
  ssr: false,
  loading: () => (
    <div className="mx-auto max-w-[900px] p-5 text-center" style={{ color: "var(--color-muted)" }}>
      Loading notebook viewer...
    </div>
  ),
});

interface Props {
  notebook: NotebookData;
}

export default function NotebookViewer({ notebook }: Props) {
  const [renderError, setRenderError] = useState(false);
  const [isMounted, setIsMounted] = useState(false);

  // Ensure we're client-side before rendering
  useEffect(() => {
    setIsMounted(true);
  }, []);

  if (!isMounted) {
    return (
      <div className="mx-auto max-w-[900px] p-5 text-center" style={{ color: "var(--color-muted)" }}>
        Loading notebook viewer...
      </div>
    );
  }

  // react-jupyter-notebook expects a strict internal cellType that diverges from
  // the real Jupyter nbformat spec in several ways (metadata required, source must
  // be string[], execution_count can't be null, outputs have a narrow shape).
  // We normalise what we can and cast the rest — the ErrorBoundary catches any
  // runtime issues that slip through.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const compatNotebook: any = {
    ...notebook,
    cells: notebook.cells.map((cell) => ({
      ...cell,
      execution_count: cell.execution_count ?? undefined,
      // source must be string[] per the library's type
      source: Array.isArray(cell.source) ? cell.source : [cell.source],
      // metadata must be an object, never undefined
      metadata: cell.metadata ?? {},
      // outputs must be present for code cells
      outputs: cell.outputs ?? [],
    })),
  };

  if (!renderError) {
    return (
      <ErrorBoundary onError={() => setRenderError(true)}>
        <div
          className="notebook-rjn-wrapper mx-auto max-w-[900px] py-4"
          // Scope overrides so the library's styles don't leak into the rest of the app
          style={
            {
              "--jp-content-font-size1": "13px",
              "--jp-code-font-size": "12px",
              "--jp-cell-prompt-width": "64px",
              "--jp-layout-color0": "var(--color-surface-0)",
              "--jp-layout-color1": "var(--color-surface-1)",
              "--jp-layout-color2": "var(--color-surface-2)",
              "--jp-border-color1": "var(--color-border)",
              "--jp-content-font-color1": "var(--color-foreground)",
              "--jp-content-font-color2": "var(--color-muted-fg)",
              "--jp-mirror-editor-keyword-color": "#c084fc",
              "--jp-mirror-editor-string-color": "#86efac",
              "--jp-mirror-editor-number-color": "#fbbf24",
              "--jp-mirror-editor-comment-color": "var(--color-muted)",
              "--jp-mirror-editor-def-color": "#60a5fa",
              "--jp-mirror-editor-builtin-color": "#fb923c",
              "--jp-mirror-editor-operator-color": "#94a3b8",
              "--jp-cell-editor-background": "var(--color-surface-0)",
              "--jp-cell-editor-border-color": "var(--color-border)",
              "--jp-cell-editor-active-border-color": "var(--color-accent)",
            } as React.CSSProperties
          }
        >
          <JupyterNotebook rawIpynb={compatNotebook} />
        </div>
      </ErrorBoundary>
    );
  }

  // ── Fallback renderer (used only if react-jupyter-notebook throws) ──────────
  return <FallbackViewer notebook={notebook} />;
}

// ── Error boundary ─────────────────────────────────────────────────────────────

import { Component, type ErrorInfo } from "react";

interface EBProps { children: ReactNode; onError: () => void; }
interface EBState { hasError: boolean; }

class ErrorBoundary extends Component<EBProps, EBState> {
  state: EBState = { hasError: false };
  componentDidCatch(_: Error, __: ErrorInfo) {
    this.setState({ hasError: true });
    this.props.onError();
  }
  render() {
    return this.state.hasError ? null : this.props.children;
  }
}

// ── Fallback renderer ──────────────────────────────────────────────────────────
// Preserved from original — only shown if the library fails to render.

function FallbackViewer({ notebook }: { notebook: NotebookData }) {
  return (
    <div className="mx-auto max-w-[900px] p-5">
      {notebook.cells.map((cell, i) => (
        <CellView key={i} cell={cell} index={i} />
      ))}
    </div>
  );
}

function CellView({ cell, index }: { cell: NotebookCell; index: number }) {
  const source = Array.isArray(cell.source) ? cell.source.join("") : cell.source;
  const isCode = cell.cell_type === "code";

  return (
    <div className="mb-2 overflow-hidden rounded-md" style={{ border: "1px solid var(--color-border)" }}>
      {/* Header */}
      <div
        className="flex items-center gap-2 px-3 py-1.5 text-[11px]"
        style={{ background: "var(--color-surface-2)", color: "var(--color-muted)" }}
      >
        <span
          className="rounded px-1.5 py-0.5 font-mono text-[10px] font-semibold uppercase"
          style={{
            background: isCode ? "rgba(129,140,248,0.15)" : "rgba(45,212,191,0.1)",
            color: isCode ? "var(--color-indigo)" : "var(--color-accent)",
          }}
        >
          {cell.cell_type}
        </span>
        <span className="font-mono text-[10px]" style={{ color: "var(--color-muted)" }}>
          [{index + 1}]
        </span>
      </div>

      {/* Body */}
      <div className="px-4 py-3" style={{ background: "var(--color-surface-1)" }}>
        {isCode ? (
          <pre
            className="overflow-x-auto whitespace-pre-wrap font-mono text-xs leading-relaxed"
            style={{ color: "var(--color-muted-fg)" }}
          >
            {highlightPython(source)}
          </pre>
        ) : (
          <div className="text-sm leading-relaxed" style={{ color: "var(--color-foreground)" }}>
            <MarkdownBlock text={source} />
          </div>
        )}
      </div>
    </div>
  );
}

// ── Python syntax highlighting (no deps) ──────────────────────────────────────

function highlightPython(code: string): ReactNode[] {
  return code.split("\n").map((line, i, arr) => {
    const commentIdx = findCommentStart(line);
    const main = commentIdx >= 0 ? line.slice(0, commentIdx) : line;
    const comment = commentIdx >= 0 ? line.slice(commentIdx) : "";
    const tokens = tokenize(main);
    return (
      <span key={i}>
        {tokens}
        {comment && (
          <span style={{ color: "var(--color-surface-4)", fontStyle: "italic" }}>{comment}</span>
        )}
        {i < arr.length - 1 && "\n"}
      </span>
    );
  });
}

function findCommentStart(line: string): number {
  let inStr = false,
    ch = "";
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (inStr) {
      if (c === ch && line[i - 1] !== "\\") inStr = false;
    } else {
      if (c === '"' || c === "'") {
        inStr = true;
        ch = c;
      }
      if (c === "#") return i;
    }
  }
  return -1;
}

const KW =
  /\b(def|class|return|import|from|as|if|elif|else|for|while|try|except|finally|with|yield|lambda|and|or|not|in|is|True|False|None|self|raise|pass|break|continue|assert|global|nonlocal|async|await)\b/g;
const STR = /("""[\s\S]*?"""|'''[\s\S]*?'''|f?"(?:[^"\\]|\\.)*"|f?'(?:[^'\\]|\\.)*')/g;
const NUM = /\b(\d+\.?\d*(?:e[+-]?\d+)?)\b/g;

function tokenize(text: string): ReactNode[] {
  const parts: ReactNode[] = [];
  const html = text
    .replace(STR, "⟨S⟩$1⟨/S⟩")
    .replace(KW, "⟨K⟩$1⟨/K⟩")
    .replace(NUM, "⟨N⟩$1⟨/N⟩");

  const regex = /⟨(K|S|N)⟩(.*?)⟨\/\1⟩/g;
  let last = 0;
  let m;
  while ((m = regex.exec(html)) !== null) {
    if (m.index > last) parts.push(<span key={last}>{html.slice(last, m.index)}</span>);
    const color = m[1] === "K" ? "#c084fc" : m[1] === "S" ? "#86efac" : "#fbbf24";
    parts.push(
      <span key={m.index} style={{ color }}>
        {m[2]}
      </span>
    );
    last = regex.lastIndex;
  }
  if (last < html.length) parts.push(<span key={last}>{html.slice(last)}</span>);
  return parts;
}

// ── Simple markdown renderer (no deps) ────────────────────────────────────────

function MarkdownBlock({ text }: { text: string }) {
  const lines = text.split("\n");
  const els: ReactNode[] = [];
  let inCode = false;
  const codeBuf: string[] = [];
  let k = 0;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.trim().startsWith("```")) {
      if (inCode) {
        els.push(
          <pre
            key={`c${k++}`}
            className="my-2 overflow-x-auto rounded-md p-3 font-mono text-xs"
            style={{ background: "var(--color-surface-0)" }}
          >
            <code>{codeBuf.join("\n")}</code>
          </pre>
        );
        codeBuf.length = 0;
        inCode = false;
      } else {
        inCode = true;
      }
      continue;
    }
    if (inCode) {
      codeBuf.push(line);
      continue;
    }

    if (line.startsWith("# "))
      els.push(
        <h1 key={i} className="mb-1 mt-3 text-xl font-semibold">
          {fmtInline(line.slice(2))}
        </h1>
      );
    else if (line.startsWith("## "))
      els.push(
        <h2 key={i} className="mb-1 mt-3 text-lg font-semibold">
          {fmtInline(line.slice(3))}
        </h2>
      );
    else if (line.startsWith("### "))
      els.push(
        <h3 key={i} className="mb-1 mt-2 text-base font-semibold">
          {fmtInline(line.slice(4))}
        </h3>
      );
    else if (line.startsWith("- ") || line.startsWith("* "))
      els.push(
        <p key={i} className="pl-4 py-0.5">
          • {fmtInline(line.slice(2))}
        </p>
      );
    else if (line.trim())
      els.push(
        <p key={i} className="py-0.5">
          {fmtInline(line)}
        </p>
      );
  }
  return <>{els}</>;
}

function fmtInline(text: string): ReactNode[] {
  const parts: ReactNode[] = [];
  const re = /(\*\*.*?\*\*|`[^`]+`|\*[^*]+\*)/g;
  let last = 0;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) parts.push(<span key={last}>{text.slice(last, m.index)}</span>);
    const t = m[0];
    if (t.startsWith("**"))
      parts.push(<strong key={m.index}>{t.slice(2, -2)}</strong>);
    else if (t.startsWith("`"))
      parts.push(
        <code
          key={m.index}
          className="rounded px-1 py-0.5 font-mono text-xs"
          style={{ background: "var(--color-surface-3)" }}
        >
          {t.slice(1, -1)}
        </code>
      );
    else if (t.startsWith("*"))
      parts.push(<em key={m.index}>{t.slice(1, -1)}</em>);
    last = re.lastIndex;
  }
  if (last < text.length) parts.push(<span key={last}>{text.slice(last)}</span>);
  return parts;
}