"use client";

import { useState, useRef, type DragEvent, type ChangeEvent } from "react";

interface Props {
  onUpload: (file: File) => void;
  hasNvidiaKey: boolean;
  onOpenSettings: () => void;
}

export default function UploadView({ onUpload, hasNvidiaKey, onOpenSettings }: Props) {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleFile = async (file: File) => {
    if (!file.name.toLowerCase().endsWith(".pdf")) return;
    setUploading(true);
    await onUpload(file);
    setUploading(false);
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  };

  const onChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-6 p-10 animate-fade-in">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-4xl font-bold tracking-tight">
          Jupy<span style={{ color: "var(--color-accent)" }}>Paper</span>
        </h1>
        <p className="mt-2 text-base" style={{ color: "var(--color-muted)" }}>
          Drop a research paper PDF and get a Jupyter notebook
        </p>
      </div>

      {/* API key warning */}
      {!hasNvidiaKey && (
        <button
          onClick={onOpenSettings}
          className="flex cursor-pointer items-center gap-2 rounded-lg px-4 py-2.5 text-sm transition-colors"
          style={{
            color: "var(--color-warning)",
            background: "rgba(251,191,36,0.08)",
            border: "1px solid rgba(251,191,36,0.2)",
          }}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
            <line x1="12" y1="9" x2="12" y2="13" />
            <line x1="12" y1="17" x2="12.01" y2="17" />
          </svg>
          NVIDIA API key required — click here to add it in Settings
        </button>
      )}

      {/* Dropzone */}
      <div
        className={`dropzone-idle relative flex w-full max-w-lg cursor-pointer flex-col items-center gap-4 rounded-2xl p-14 ${dragging ? "dropzone-active" : ""}`}
        style={{ background: "var(--color-surface-1)" }}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        {/* Icon */}
        <div
          className="flex h-14 w-14 items-center justify-center rounded-xl transition-all"
          style={{
            background: dragging ? "rgba(45,212,191,0.15)" : "var(--color-surface-3)",
            color: "var(--color-accent)",
          }}
        >
          {uploading ? (
            <div
              className="h-6 w-6 rounded-full animate-spin-ring"
              style={{ border: "2px solid var(--color-surface-4)", borderTopColor: "var(--color-accent)" }}
            />
          ) : (
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
                 strokeLinecap="round" strokeLinejoin="round">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <polyline points="14 2 14 8 20 8" />
              <line x1="12" y1="18" x2="12" y2="12" />
              <polyline points="9 15 12 12 15 15" />
            </svg>
          )}
        </div>

        <h3 className="text-base font-medium">
          {uploading ? "Uploading..." : "Drop PDF here"}
        </h3>
        <p className="text-sm" style={{ color: "var(--color-muted)" }}>
          or click to browse — accepts .pdf research papers
        </p>

        <input
          ref={fileRef}
          type="file"
          accept=".pdf"
          onChange={onChange}
          className="absolute inset-0 cursor-pointer opacity-0"
        />
      </div>

      {/* Footer */}
      <div className="flex items-center gap-1.5 font-mono text-xs" style={{ color: "var(--color-muted)" }}>
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
          <path d="M7 11V7a5 5 0 0 1 10 0v4" />
        </svg>
        Turn Deep Learning Papers into Jupyter Notebooks: Skyrocket Your Learning
      </div>
    </div>
  );
}