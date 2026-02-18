"use client";

import { useState, useEffect, useCallback } from "react";
import type { AppView, Session, APIKeysState } from "./types";
import SplashScreen from "./components/SplashScreen";
import UploadView from "./components/UploadView";
import WorkspaceView from "./components/WorkspaceView";
import SettingsModal from "./components/SettingsModal";

const API = "http://127.0.0.1:9847";

export default function Home() {
  const [view, setView] = useState<AppView>("splash");
  const [serverReady, setServerReady] = useState(false);
  const [serverError, setServerError] = useState<string | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [keys, setKeys] = useState<APIKeysState | null>(null);

  // ── Poll Python backend health ────────────────────────────────────────
  useEffect(() => {
    if (typeof window !== 'undefined' && window.electronAPI) {
      window.electronAPI.onPythonReady(() => setServerReady(true));
    }
    const poll = setInterval(async () => {
      try {
        const r = await fetch(`${API}/api/health`);
        if (r.ok) { setServerReady(true); clearInterval(poll); }
      } catch { /* still starting */ }
    }, 600);
    return () => clearInterval(poll);
  }, []);

  // ── Splash → upload after ready ───────────────────────────────────────
  useEffect(() => {
    if (serverReady && view === "splash") {
      const t = setTimeout(() => setView("upload"), 1800);
      return () => clearTimeout(t);
    }
  }, [serverReady, view]);

  // ── Load stored keys ──────────────────────────────────────────────────
  useEffect(() => {
    if (!serverReady) return;
    fetch(`${API}/api/keys`).then(r => r.json()).then(setKeys).catch(() => {});
  }, [serverReady]);

  // ── Upload ────────────────────────────────────────────────────────────
  const handleUpload = useCallback(async (file: File) => {
    const form = new FormData();
    form.append("file", file);
    try {
      const resp = await fetch(`${API}/api/upload`, { method: "POST", body: form });
      const data = await resp.json();
      setSession({
        id: data.session_id,
        paperName: data.paper_name,
        pdfUrl: `${API}/api/pdf/${data.session_id}`,
        status: "uploaded",
        events: [],
      });
      setView("workspace");
    } catch (err) {
      console.error("Upload failed:", err);
    }
  }, []);

  return (
    <div className="flex h-screen w-full flex-col bg-[var(--color-surface-0)]">
      {/* ── Titlebar ───────────────────────────────────────────────────── */}
      <div className="drag-region relative z-50 flex h-9 shrink-0 items-center justify-end px-3"
           style={{ background: "var(--color-surface-0)" }}>
        <button
          onClick={() => setShowSettings(true)}
          className="no-drag relative z-[51] cursor-pointer rounded-md p-1.5 transition-colors"
          style={{ color: "var(--color-muted)" }}
          title="Settings"
        >
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
               strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="3" />
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
          </svg>
        </button>
      </div>

      {/* ── Views ──────────────────────────────────────────────────────── */}
      {view === "splash" && <SplashScreen error={serverError} />}
      {view === "upload" && (
        <UploadView
          onUpload={handleUpload}
          hasNvidiaKey={keys?.has_nvidia_key ?? false}
          onOpenSettings={() => setShowSettings(true)}
        />
      )}
      {view === "workspace" && session && (
        <WorkspaceView
          session={session}
          setSession={setSession}
          apiBase={API}
          onNewPipeline={() => { setSession(null); setView("upload"); }}
        />
      )}

      {/* ── Settings ───────────────────────────────────────────────────── */}
      {showSettings && (
        <SettingsModal
          apiBase={API}
          keys={keys}
          onClose={() => setShowSettings(false)}
          onSaved={setKeys}
        />
      )}
    </div>
  );
}