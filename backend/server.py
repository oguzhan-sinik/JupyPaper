"""
JupyPaper — FastAPI Bridge Server for Electron App.

Wraps the full pipeline and streams real-time events via WebSocket.
Electron spawns this server on startup and connects to it.

Endpoints:
  POST /api/upload         — Upload PDF, returns session_id
  WS   /ws/{session_id}    — Stream pipeline events in real-time
  GET  /api/notebook/{id}  — Download final .ipynb
  GET  /api/notebook-json/{id} — Notebook as JSON for inline rendering
  GET  /api/pdf/{id}       — Serve uploaded PDF for preview
  GET  /api/keys           — Get stored API keys (masked)
  POST /api/keys           — Save API keys
  GET  /api/health         — Health check
"""

import asyncio
import json
import os
import re
import sys
import uuid
import threading
import time
import traceback
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# ── Persistent app data ──────────────────────────────────────────────────────

APP_DATA_DIR = Path.home() / ".jupypaper"
KEYS_FILE = APP_DATA_DIR / "api_keys.json"
SESSIONS_DIR = APP_DATA_DIR / "sessions"
APP_DATA_DIR.mkdir(exist_ok=True)
SESSIONS_DIR.mkdir(exist_ok=True)

# Backend modules live alongside this file
BACKEND_DIR = Path(__file__).parent
sys.path.insert(0, str(BACKEND_DIR))

app = FastAPI(title="JupyPaper Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: dict[str, dict] = {}
ws_connections: dict[str, WebSocket] = {}


# ── API Key Management ────────────────────────────────────────────────────────

class APIKeys(BaseModel):
    nvidia_api_key: str = ""
    github_token: str = ""
    valyu_api_key: str = ""


def load_keys() -> dict:
    if KEYS_FILE.exists():
        try:
            return json.loads(KEYS_FILE.read_text())
        except Exception:
            pass
    return {}


def save_keys(keys: dict):
    KEYS_FILE.write_text(json.dumps(keys, indent=2))


def mask_key(key: str) -> str:
    if not key or len(key) < 8:
        return "***" if key else ""
    return key[:4] + "•" * (len(key) - 8) + key[-4:]


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "2.0"}


@app.get("/api/keys")
async def get_keys():
    keys = load_keys()
    return {
        "nvidia_api_key": mask_key(keys.get("nvidia_api_key", "")),
        "github_token": mask_key(keys.get("github_token", "")),
        "valyu_api_key": mask_key(keys.get("valyu_api_key", "")),
        "has_nvidia_key": bool(keys.get("nvidia_api_key")),
        "has_github_token": bool(keys.get("github_token")),
        "has_valyu_key": bool(keys.get("valyu_api_key")),
    }


@app.post("/api/keys")
async def set_keys(keys: APIKeys):
    existing = load_keys()
    if keys.nvidia_api_key and "•" not in keys.nvidia_api_key:
        existing["nvidia_api_key"] = keys.nvidia_api_key
    if keys.github_token and "•" not in keys.github_token:
        existing["github_token"] = keys.github_token
    if keys.valyu_api_key and "•" not in keys.valyu_api_key:
        existing["valyu_api_key"] = keys.valyu_api_key
    save_keys(existing)
    return {"status": "saved"}


# ── PDF Upload ────────────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())[:8]
    session_dir = SESSIONS_DIR / session_id
    session_dir.mkdir(exist_ok=True)

    pdf_path = session_dir / file.filename
    with open(pdf_path, "wb") as f:
        content = await file.read()
        f.write(content)

    paper_name = Path(file.filename).stem
    sessions[session_id] = {
        "id": session_id,
        "pdf_path": str(pdf_path),
        "paper_name": paper_name,
        "output_dir": str(session_dir / "output" / paper_name),
        "status": "uploaded",
    }
    return {"session_id": session_id, "paper_name": paper_name}


# ── Event Emitter (pipeline → WebSocket) ─────────────────────────────────────

class EventEmitter:
    def __init__(self, session_id: str, loop: asyncio.AbstractEventLoop):
        self.session_id = session_id
        self.loop = loop

    def emit(self, event: dict):
        event["timestamp"] = time.time()
        ws = ws_connections.get(self.session_id)
        if ws:
            try:
                asyncio.run_coroutine_threadsafe(ws.send_json(event), self.loop)
            except Exception:
                pass

    def stage(self, name: str, description: str = ""):
        self.emit({"type": "stage", "name": name, "description": description})

    def step(self, name: str, detail: str = "", step: int = 0, total: int = 0):
        self.emit({"type": "step", "name": name, "detail": detail, "step": step, "total": total})

    def log(self, message: str, level: str = "info"):
        self.emit({"type": "log", "message": message, "level": level})

    def cell_start(self, cell_id: int, title: str, cell_type: str):
        self.emit({"type": "cell_start", "cell_id": cell_id, "title": title, "cell_type": cell_type})

    def error(self, message: str):
        self.emit({"type": "error", "message": message})

    def complete(self, notebook_path: str, summary: dict):
        self.emit({"type": "complete", "notebook_path": notebook_path, "summary": summary})


# ── Pipeline execution thread ─────────────────────────────────────────────────

def run_pipeline_thread(session_id: str, emitter: EventEmitter):
    session = sessions.get(session_id)
    if not session:
        emitter.error("Session not found")
        return

    try:
        keys = load_keys()
        if keys.get("nvidia_api_key"):
            os.environ["NVIDIA_API_KEY"] = keys["nvidia_api_key"]
        if keys.get("github_token"):
            os.environ["GITHUB_TOKEN"] = keys["github_token"]
        if keys.get("valyu_api_key"):
            os.environ["VALYU_API_KEY"] = keys["valyu_api_key"]

        api_key = os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            emitter.error("NVIDIA API key not set. Please add it in Settings.")
            return

        from utils import get_nim_client, CostTracker, estimate_tokens, load_config
        from pdf_processor import process_pdf
        from orchestrator import Orchestrator
        from planning import run_planning
        from analysis import run_analysis
        from code_generation import run_code_generation
        from notebook_assembler import (
            assemble_notebook, validate_notebook, debug_notebook, save_notebook,
        )

        pdf_path = session["pdf_path"]
        paper_name = session["paper_name"]
        output_dir = session["output_dir"]
        for d in [output_dir, f"{output_dir}/analysis", f"{output_dir}/cells"]:
            os.makedirs(d, exist_ok=True)

        cfg = load_config(str(BACKEND_DIR / "config.yaml"))
        m = cfg.get("models", {})
        orch_model = m.get("orchestrator", "nvidia/llama-3.3-nemotron-super-49b-v1.5")
        sub_model = m.get("sub_agent", "nvidia/nvidia-nemotron-nano-9b-v2")
        vis_model = m.get("vision", "nvidia/nemotron-nano-12b-v2-vl")
        base_url = m.get("base_url", "https://integrate.api.nvidia.com/v1")

        client = get_nim_client(api_key, base_url)
        cost_tracker = CostTracker()

        # ── Capture ALL pipeline output → WebSocket ───────────────────
        # Monkeypatch rich console.print BEFORE any pipeline stage
        import utils as _u
        _orig_console_print = _u.console.print
        
        # Also capture bare print() and sys.stdout writes
        _orig_stdout = sys.stdout
        _orig_stderr = sys.stderr

        class _StreamCapture:
            """Tee stdout/stderr to both original stream and WebSocket emitter."""
            def __init__(self, orig, em, level="info"):
                self.orig = orig
                self.em = em
                self.level = level
                self.buf = ""
            def write(self, s):
                self.orig.write(s)
                self.buf += s
                while "\n" in self.buf:
                    line, self.buf = self.buf.split("\n", 1)
                    line = line.strip()
                    if line:
                        # Strip ANSI escape codes and Rich markup
                        clean = re.sub(r'\x1b\[[0-9;]*m', '', line)
                        clean = re.sub(r'\[/?[^\]]*\]', '', clean).strip()
                        if clean and not clean.startswith("─") and clean != "|":
                            self.em.log(clean, self.level)
            def flush(self):
                self.orig.flush()
                if self.buf.strip():
                    clean = re.sub(r'\x1b\[[0-9;]*m', '', self.buf)
                    clean = re.sub(r'\[/?[^\]]*\]', '', clean).strip()
                    if clean:
                        self.em.log(clean, self.level)
                    self.buf = ""

        sys.stdout = _StreamCapture(_orig_stdout, emitter, "info")
        sys.stderr = _StreamCapture(_orig_stderr, emitter, "warning")

        # ── Stage 1: PDF Processing ───────────────────────────────────
        emitter.stage("PDF Processing", "Extracting text and figures from PDF")
        emitter.log(f"Processing {os.path.basename(pdf_path)}...")

        pdf_result = process_pdf(
            pdf_path=pdf_path,
            vision_client=client,
            vision_model=vis_model,
            jpeg_quality=cfg.get("pdf", {}).get("vision_jpeg_quality", 75),
            extra_remove=set(cfg.get("remove_sections", [])),
        )
        paper_md = pdf_result.markdown
        figures = pdf_result.figures

        with open(f"{output_dir}/paper.md", "w", encoding="utf-8") as f:
            f.write(paper_md)

        tok = estimate_tokens(paper_md)
        emitter.log(f"Paper: {len(paper_md):,} chars (~{tok:,} tokens)")
        emitter.log(f"Extracted {len(figures)} figure(s)")
        emitter.step("PDF Processing", "Complete", step=1, total=5)

        # ── Initialize Orchestrator ───────────────────────────────────
        emitter.log(f"NIM Orchestrator: {orch_model.split('/')[-1]}")
        emitter.log(f"NIM Sub-agent: {sub_model.split('/')[-1]}")
        orchestrator = Orchestrator(
            client=client,
            orchestrator_model=orch_model,
            sub_agent_model=sub_model,
            cost_tracker=cost_tracker,
            api_key=api_key,
            base_url=base_url,
        )

        # ── Stage 2: Planning ─────────────────────────────────────────
        emitter.stage("Planning", "Analyzing paper, searching repos, extracting config")

        planning_result = run_planning(orchestrator, paper_md, output_dir)
        emitter.step("Planning", "Complete", step=2, total=5)

        repo_url = planning_result.get("repo_info", {}).get("repo_url")
        if repo_url:
            emitter.log(f"Found repository: {repo_url}")
        else:
            emitter.log("No repository found — generating from paper only")

        # ── Stage 3: Analysis ─────────────────────────────────────────
        emitter.stage("Analysis", "Detailed logic for each notebook cell")
        structure = planning_result.get("notebook_structure", {})
        cells = structure.get("cells", [])
        for i, cell in enumerate(cells):
            emitter.cell_start(cell.get("cell_id", i), cell.get("title", f"Cell {i+1}"), cell.get("cell_type", "code"))

        cell_analyses = run_analysis(orchestrator, paper_md, planning_result, output_dir)
        emitter.step("Analysis", f"Analyzed {len(cell_analyses)} cells", step=3, total=5)

        # ── Stage 4: Code Generation ──────────────────────────────────
        emitter.stage("Code Generation", "Generating code with state manifest")
        generated_cells = run_code_generation(orchestrator, paper_md, planning_result, cell_analyses, output_dir)
        emitter.step("Code Generation", f"Generated {len(generated_cells)} cells", step=4, total=5)

        # ── Stage 5: Assembly & Debug ─────────────────────────────────
        emitter.stage("Assembly", "Import dedup, reference fixing, figure embedding")
        nb_cfg = cfg.get("notebook", {})
        nb = assemble_notebook(
            generated_cells, planning_result,
            kernel=nb_cfg.get("kernel", "python3"),
            language=nb_cfg.get("language", "python"),
            figures=figures,
        )

        emitter.log("Validating notebook...")
        issues = validate_notebook(nb, orchestrator)
        retries = cfg.get("pipeline", {}).get("max_retries", 2)
        attempt = 0
        while issues and attempt < retries:
            emitter.log(f"Debug pass {attempt + 1}: fixing {len(issues)} issue(s)")
            nb = debug_notebook(nb, issues, orchestrator, planning_result)
            issues = validate_notebook(nb)
            attempt += 1

        notebook_path = f"{output_dir}/{paper_name}.ipynb"
        save_notebook(nb, notebook_path)
        cost_tracker.save(f"{output_dir}/cost_summary.json")
        emitter.step("Assembly", "Complete", step=5, total=5)

        # ── Done ──────────────────────────────────────────────────────
        cn = sum(1 for c in nb.cells if c.cell_type == "code")
        mn = sum(1 for c in nb.cells if c.cell_type == "markdown")
        summary = {
            "notebook_path": notebook_path,
            "total_cells": len(nb.cells),
            "code_cells": cn,
            "markdown_cells": mn,
            "figures_embedded": len(figures),
            "repo_url": planning_result.get("repo_info", {}).get("repo_url"),
            "repo_source": planning_result.get("repo_info", {}).get("source"),
            "data_strategy": planning_result.get("config", {}).get("data_strategy"),
            "cost": cost_tracker.summary(),
            "total_cost": cost_tracker.total_cost,
        }

        sessions[session_id]["status"] = "complete"
        sessions[session_id]["notebook_path"] = notebook_path
        sessions[session_id]["summary"] = summary

        # Restore all captured streams
        _u.console.print = _orig_console_print
        sys.stdout = _orig_stdout
        sys.stderr = _orig_stderr

        emitter.complete(notebook_path, summary)

    except Exception as e:
        # Restore streams on error too
        try:
            import utils as _u2
            if '_orig_console_print' in dir():
                _u2.console.print = _orig_console_print
            if '_orig_stdout' in dir():
                sys.stdout = _orig_stdout
                sys.stderr = _orig_stderr
        except Exception:
            pass
        emitter.error(f"Pipeline failed: {str(e)}\n{traceback.format_exc()}")
        sessions[session_id]["status"] = "error"


# ── WebSocket pipeline streaming ──────────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    ws_connections[session_id] = websocket
    try:
        data = await websocket.receive_json()
        if data.get("action") == "start":
            loop = asyncio.get_event_loop()
            emitter = EventEmitter(session_id, loop)
            sessions[session_id]["status"] = "running"
            thread = threading.Thread(target=run_pipeline_thread, args=(session_id, emitter), daemon=True)
            thread.start()

            while True:
                try:
                    msg = await asyncio.wait_for(websocket.receive_json(), timeout=1.0)
                    if msg.get("action") == "cancel":
                        break
                except asyncio.TimeoutError:
                    status = sessions.get(session_id, {}).get("status", "")
                    if status in ("complete", "error"):
                        await asyncio.sleep(0.5)
                        break
    except WebSocketDisconnect:
        pass
    finally:
        ws_connections.pop(session_id, None)


# ── File serving ──────────────────────────────────────────────────────────────

@app.get("/api/notebook/{session_id}")
async def get_notebook(session_id: str):
    session = sessions.get(session_id)
    if not session or not session.get("notebook_path"):
        return JSONResponse({"error": "Not found"}, status_code=404)
    path = session["notebook_path"]
    if not os.path.exists(path):
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(path, media_type="application/json", filename=os.path.basename(path))


@app.get("/api/notebook-json/{session_id}")
async def get_notebook_json(session_id: str):
    session = sessions.get(session_id)
    if not session or not session.get("notebook_path"):
        return JSONResponse({"error": "Not found"}, status_code=404)
    path = session["notebook_path"]
    if not os.path.exists(path):
        return JSONResponse({"error": "File not found"}, status_code=404)
    with open(path, "r") as f:
        nb = json.load(f)
    return JSONResponse(nb)


@app.get("/api/pdf/{session_id}")
async def get_pdf(session_id: str):
    session = sessions.get(session_id)
    if not session:
        return JSONResponse({"error": "Not found"}, status_code=404)
    pdf_path = session["pdf_path"]
    if not os.path.exists(pdf_path):
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(pdf_path, media_type="application/pdf")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9847)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()
    print(f"JupyPaper server starting on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")