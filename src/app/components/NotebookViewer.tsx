"use client";

import { type ReactNode } from "react";
import type { NotebookData, NotebookCell } from "../types";

interface Props {
  notebook: NotebookData;
}

export default function NotebookViewer({ notebook }: Props) {
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
      <div className="flex items-center gap-2 px-3 py-1.5 text-[11px]"
           style={{ background: "var(--color-surface-2)", color: "var(--color-muted)" }}>
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
          <pre className="overflow-x-auto whitespace-pre-wrap font-mono text-xs leading-relaxed"
               style={{ color: "var(--color-muted-fg)" }}>
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

// ── Python syntax highlighting (no deps) ─────────────────────────────────────

function highlightPython(code: string): ReactNode[] {
  return code.split("\n").map((line, i, arr) => {
    const commentIdx = findCommentStart(line);
    let main = commentIdx >= 0 ? line.slice(0, commentIdx) : line;
    const comment = commentIdx >= 0 ? line.slice(commentIdx) : "";

    const tokens = tokenize(main);
    return (
      <span key={i}>
        {tokens}
        {comment && <span style={{ color: "var(--color-surface-4)", fontStyle: "italic" }}>{comment}</span>}
        {i < arr.length - 1 && "\n"}
      </span>
    );
  });
}

function findCommentStart(line: string): number {
  let inStr = false, ch = "";
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (inStr) { if (c === ch && line[i - 1] !== "\\") inStr = false; }
    else { if (c === '"' || c === "'") { inStr = true; ch = c; } if (c === "#") return i; }
  }
  return -1;
}

const KW = /\b(def|class|return|import|from|as|if|elif|else|for|while|try|except|finally|with|yield|lambda|and|or|not|in|is|True|False|None|self|raise|pass|break|continue|assert|global|nonlocal|async|await)\b/g;
const STR = /("""[\s\S]*?"""|'''[\s\S]*?'''|f?"(?:[^"\\]|\\.)*"|f?'(?:[^'\\]|\\.)*')/g;
const NUM = /\b(\d+\.?\d*(?:e[+-]?\d+)?)\b/g;

function tokenize(text: string): ReactNode[] {
  const parts: ReactNode[] = [];
  // Mark strings first
  let html = text.replace(STR, "⟨S⟩$1⟨/S⟩").replace(KW, "⟨K⟩$1⟨/K⟩").replace(NUM, "⟨N⟩$1⟨/N⟩");

  const regex = /⟨(K|S|N)⟩(.*?)⟨\/\1⟩/g;
  let last = 0;
  let m;
  while ((m = regex.exec(html)) !== null) {
    if (m.index > last) parts.push(<span key={last}>{html.slice(last, m.index)}</span>);
    const color = m[1] === "K" ? "#c084fc" : m[1] === "S" ? "#86efac" : "#fbbf24";
    parts.push(<span key={m.index} style={{ color }}>{m[2]}</span>);
    last = regex.lastIndex;
  }
  if (last < html.length) parts.push(<span key={last}>{html.slice(last)}</span>);
  return parts;
}

// ── Simple markdown renderer (no deps) ───────────────────────────────────────

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
          <pre key={`c${k++}`} className="my-2 overflow-x-auto rounded-md p-3 font-mono text-xs"
               style={{ background: "var(--color-surface-0)" }}>
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
    if (inCode) { codeBuf.push(line); continue; }

    if (line.startsWith("# ")) els.push(<h1 key={i} className="mb-1 mt-3 text-xl font-semibold">{fmtInline(line.slice(2))}</h1>);
    else if (line.startsWith("## ")) els.push(<h2 key={i} className="mb-1 mt-3 text-lg font-semibold">{fmtInline(line.slice(3))}</h2>);
    else if (line.startsWith("### ")) els.push(<h3 key={i} className="mb-1 mt-2 text-base font-semibold">{fmtInline(line.slice(4))}</h3>);
    else if (line.startsWith("- ") || line.startsWith("* ")) els.push(<p key={i} className="pl-4 py-0.5">• {fmtInline(line.slice(2))}</p>);
    else if (line.trim()) els.push(<p key={i} className="py-0.5">{fmtInline(line)}</p>);
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
    if (t.startsWith("**")) parts.push(<strong key={m.index}>{t.slice(2, -2)}</strong>);
    else if (t.startsWith("`")) parts.push(
      <code key={m.index} className="rounded px-1 py-0.5 font-mono text-xs"
            style={{ background: "var(--color-surface-3)" }}>{t.slice(1, -1)}</code>
    );
    else if (t.startsWith("*")) parts.push(<em key={m.index}>{t.slice(1, -1)}</em>);
    last = re.lastIndex;
  }
  if (last < text.length) parts.push(<span key={last}>{text.slice(last)}</span>);
  return parts;
}