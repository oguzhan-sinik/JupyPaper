// ── Electron IPC Bridge ──────────────────────────────────────────────────────

export interface ElectronAPI {
    sendNotification: (text: string) => void;
    getAppVersion: () => string;
    getPythonPort: () => Promise<number>;
    openFileDialog: () => Promise<string | null>;
    openExternal: (url: string) => Promise<void>;
    saveNotebook: (sessionId: string) => Promise<string | null>;
    onPythonReady: (callback: (port: number) => void) => void;
  }
  
  declare global {
    interface Window {
      electronAPI?: ElectronAPI;
    }
  }
  
  // ── Pipeline Events (streamed via WebSocket from Python backend) ─────────────
  
  export type PipelineEventType =
    | "stage"
    | "step"
    | "log"
    | "progress"
    | "cell_start"
    | "cell_done"
    | "cost"
    | "error"
    | "complete";
  
  export interface PipelineEvent {
    type: PipelineEventType;
    timestamp?: number;
    name?: string;
    description?: string;
    detail?: string;
    step?: number;
    total?: number;
    message?: string;
    level?: "info" | "warning" | "error" | "success";
    cell_id?: number;
    title?: string;
    cell_type?: string;
    tokens?: number;
    elapsed?: number;
    cost?: number;
    notebook_path?: string;
    summary?: PipelineSummary;
  }
  
  export interface PipelineSummary {
    notebook_path: string;
    total_cells: number;
    code_cells: number;
    markdown_cells: number;
    figures_embedded: number;
    repo_url?: string;
    repo_source?: string;
    data_strategy?: string;
    cost?: string;
    total_cost?: number;
  }
  
  // ── App State ────────────────────────────────────────────────────────────────
  
  export type AppView = "splash" | "upload" | "workspace";
  
  export interface Session {
    id: string;
    paperName: string;
    pdfUrl: string;
    status: "uploaded" | "running" | "complete" | "error";
    events: PipelineEvent[];
    currentStage?: string;
    summary?: PipelineSummary;
  }
  
  // ── API Keys ─────────────────────────────────────────────────────────────────
  
  export interface APIKeysState {
    nvidia_api_key: string;
    github_token: string;
    valyu_api_key: string;
    has_nvidia_key: boolean;
    has_github_token: boolean;
    has_valyu_key: boolean;
  }
  
  // ── Notebook ─────────────────────────────────────────────────────────────────
  
  export interface NotebookCell {
    cell_type: "code" | "markdown" | "raw";
    source: string[] | string;
    metadata?: Record<string, unknown>;
    outputs?: NotebookOutput[];
    execution_count?: number | null;
  }
  
  export interface NotebookOutput {
    output_type: string;
    text?: string[];
    data?: Record<string, string[]>;
    name?: string;
  }
  
  export interface NotebookData {
    nbformat: number;
    nbformat_minor: number;
    metadata: Record<string, unknown>;
    cells: NotebookCell[];
  }