"use client";

import { useState, useEffect } from "react";
import type { APIKeysState } from "../types";

interface Props {
  apiBase: string;
  keys: APIKeysState | null;
  onClose: () => void;
  onSaved: (keys: APIKeysState) => void;
}

export default function SettingsModal({ apiBase, keys, onClose, onSaved }: Props) {
  const [nvidiaKey, setNvidiaKey] = useState("");
  const [githubToken, setGithubToken] = useState("");
  const [valyuKey, setValyuKey] = useState("");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    if (keys) {
      if (keys.has_nvidia_key) setNvidiaKey(keys.nvidia_api_key);
      if (keys.has_github_token) setGithubToken(keys.github_token);
      if (keys.has_valyu_key) setValyuKey(keys.valyu_api_key);
    }
  }, [keys]);

  const handleSave = async () => {
    setSaving(true);
    try {
      await fetch(`${apiBase}/api/keys`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          nvidia_api_key: nvidiaKey,
          github_token: githubToken,
          valyu_api_key: valyuKey,
        }),
      });
      const resp = await fetch(`${apiBase}/api/keys`);
      const newKeys = await resp.json();
      onSaved(newKeys);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch (e) {
      console.error("Failed to save keys:", e);
    }
    setSaving(false);
  };

  const openLink = (url: string) => {
    if (typeof window !== 'undefined' && window.electronAPI) {
      window.electronAPI.openExternal(url);
    } else {
      window.open(url, "_blank");
    }
  };

  return (
    <div className="fixed inset-0 z-[200] flex items-center justify-center animate-fade-in"
         style={{ background: "rgba(0,0,0,0.6)", backdropFilter: "blur(4px)" }}
         onClick={onClose}>
      <div className="w-full max-w-md overflow-hidden rounded-2xl animate-scale-in"
           style={{ background: "var(--color-surface-2)", border: "1px solid var(--color-border)", boxShadow: "0 8px 48px rgba(0,0,0,0.5)" }}
           onClick={(e) => e.stopPropagation()}>

        {/* Header */}
        <div className="flex items-center justify-between border-b px-5 py-4"
             style={{ borderColor: "var(--color-border)" }}>
          <h2 className="text-base font-semibold">Settings</h2>
          <button onClick={onClose} className="cursor-pointer rounded p-1 transition-colors"
                  style={{ color: "var(--color-muted)" }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        {/* Body */}
        <div className="flex flex-col gap-5 p-5">
          <KeyField
            label="NVIDIA API Key"
            required
            value={nvidiaKey}
            onChange={setNvidiaKey}
            placeholder="nvapi-..."
            hint="Get free key at"
            hintLink="build.nvidia.com"
            hintUrl="https://build.nvidia.com"
            onOpenLink={openLink}
          />
          {/* 
          <KeyField
            label="GitHub Token"
            value={githubToken}
            onChange={setGithubToken}
            placeholder="ghp_..."
            hint="Create at"
            hintLink="github.com/settings/tokens"
            hintUrl="https://github.com/settings/tokens"
            onOpenLink={openLink}
          />
          */}
          
          <KeyField
            label="Valyu API Key for Web Search"
            value={valyuKey}
            onChange={setValyuKey}
            placeholder="val-..."
            hint="Get free key at"
            hintLink="platform.valyu.ai"
            hintUrl="https://platform.valyu.ai"
            onOpenLink={openLink}
          />
          {saved && (
            <p className="text-center text-xs" style={{ color: "var(--color-success)" }}>
              ✓ Keys saved successfully
            </p>
          )}
        </div>

        {/* Footer */}
        <div className="flex justify-end gap-2.5 border-t px-5 py-3.5"
             style={{ borderColor: "var(--color-border)" }}>
          <button onClick={onClose}
            className="cursor-pointer rounded-md border px-5 py-2 text-sm font-medium transition-colors"
            style={{ background: "var(--color-surface-3)", borderColor: "var(--color-border)", color: "var(--color-muted-fg)" }}>
            Cancel
          </button>
          <button onClick={handleSave} disabled={saving}
            className="cursor-pointer rounded-md border px-5 py-2 text-sm font-medium text-white transition-colors"
            style={{ background: "var(--color-accent-dim)", borderColor: "var(--color-accent)" }}>
            {saving ? "Saving..." : "Save Keys"}
          </button>
        </div>
      </div>
    </div>
  );
}

function KeyField({
  label, required, value, onChange, placeholder, hint, hintLink, hintUrl, onOpenLink,
}: {
  label: string;
  required?: boolean;
  value: string;
  onChange: (v: string) => void;
  placeholder: string;
  hint: string;
  hintLink: string;
  hintUrl: string;
  onOpenLink: (url: string) => void;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <label className="flex items-center gap-1.5 text-xs font-semibold"
             style={{ color: "var(--color-muted-fg)" }}>
        {label}
        {required
          ? <span className="text-[10px]" style={{ color: "var(--color-danger)" }}>required</span>
          : <span className="text-[10px] font-normal" style={{ color: "var(--color-muted)" }}>optional</span>}
      </label>
      <input
        type="password"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        spellCheck={false}
        className="rounded-md border px-3 py-2 font-mono text-sm outline-none transition-colors focus:border-[var(--color-accent)]"
        style={{
          background: "var(--color-surface-0)",
          borderColor: "var(--color-border)",
          color: "var(--color-foreground)",
        }}
      />
      <span className="text-[11px]" style={{ color: "var(--color-muted)" }}>
        {hint}{" "}
        <button className="cursor-pointer underline" style={{ color: "var(--color-accent)" }}
                onClick={() => onOpenLink(hintUrl)}>
          {hintLink}
        </button>
      </span>
    </div>
  );
}