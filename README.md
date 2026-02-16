# JupyPaper

### Transform research papers into executable Jupyter notebooks — powered by NVIDIA NIM

<p align="center">
  <img src="https://img.shields.io/badge/NVIDIA-NIM-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="NVIDIA NIM" />
  <img src="https://img.shields.io/badge/Electron-Next.js-000000?style=for-the-badge&logo=electron&logoColor=white" alt="Electron" />
  <img src="https://img.shields.io/badge/Python-FastAPI-009688?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/TypeScript-React-3178C6?style=for-the-badge&logo=typescript&logoColor=white" alt="TypeScript" />
</p>

---

> **🏆 Built for the [NVIDIA GTC 2026 Golden Ticket Contest](https://developer.nvidia.com/gtc-golden-ticket-contest)**
>
> This project was developed as a submission to NVIDIA's GTC Golden Ticket Contest, showcasing the power of NVIDIA NIM microservices for agentic AI workflows in scientific research and education.

---

## What is JupyPaper?

JupyPaper is a desktop application that takes a research paper PDF as input and produces a fully executable, educational Jupyter notebook as output. It reads the paper, understands the methodology, discovers existing code repositories, generates implementation code, and assembles everything into a clean notebook with explanatory markdown cells — all powered by NVIDIA NIM inference endpoints.

**The entire pipeline runs locally through NVIDIA NIM API calls — your papers never leave your machine.**

### What It Does

Upload any deep learning research paper (e.g., *Attention Is All You Need*, *BERT*, *ResNet*) and JupyPaper will:

1. Extract text and figures from the PDF using OCR and vision models
2. Analyze the paper's architecture, datasets, hyperparameters, and training procedures
3. Search for official code repositories via Papers With Code, GitHub, and Valyu
4. Plan a 20–30 cell notebook structure with dependency ordering
5. Generate each cell with a running state manifest that prevents disconnected code
6. Deduplicate imports, fix cross-cell references, embed figures, and validate with AST
7. Deliver a production-quality `.ipynb` file ready for Jupyter/Colab

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                       Electron Desktop App                           │
│                                                                      │
│  ┌─────────────────┐   ┌──────────────────────────────────────────┐  │
│  │                 │   │              Right Panel                 │  │
│  │   PDF Preview   │   │  ┌─────────────────────────────────────┐ │  │
│  │   (left panel)  │   │  │ Pipeline Log (real-time streaming) │ │  │
│  │                 │   │  │                                     │ │  │
│  │   iframe with   │   │  │ ▸ PDF Processing — Extracting...   │ │  │
│  │   full paper    │   │  │ ✓ Found 6 figure(s)                │ │  │
│  │   rendering     │   │  │ ▸ Planning — Searching repos...    │ │  │
│  │                 │   │  │ ✓ Found: github.com/...            │ │  │
│  │                 │   │  │ ▸ Code Generation — Cell 3/24...   │ │  │
│  │                 │   │  │                                     │ │  │
│  │                 │   │  ├─────────────────────────────────────┤ │  │
│  │                 │   │  │ Notebook Viewer (on complete)      │ │  │
│  │                 │   │  │ Inline .ipynb renderer with syntax │ │  │
│  │                 │   │  │ highlighting + markdown rendering  │ │  │
│  └─────────────────┘   │  └─────────────────────────────────────┘ │  │
│           ↕            └──────────────────────────────────────────┘  │
│                    WebSocket (real-time events)                       │
│           ↕                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐│
│  │                 Python FastAPI Server (port 9847)                ││
│  │  Spawned automatically by Electron — no terminal needed         ││
│  │                                                                  ││
│  │  PDF Processor → Planner → Analyzer → Code Generator → Assembler││
│  └──────────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Deep Dive

The pipeline consists of five stages, each orchestrated through NVIDIA NIM endpoints with structured Pydantic output schemas.

### Stage 1: PDF Processing

Converts the research paper PDF into clean markdown with extracted figures.

| Component | Technology |
|-----------|-----------|
| PDF → Markdown | `pymupdf4llm` with embedded OCR |
| Figure extraction | Base64 image parsing, JPEG/PNG optimization |
| Figure description | **NVIDIA Nemotron-Nano-12B-v2-VL** (vision-language model via NIM) |
| Boilerplate removal | Regex-based section filtering (references, acknowledgements, authors) |

The vision model describes each figure in the paper (charts, diagrams, architecture illustrations) so the LLM can reference them during code generation. Figures are also preserved as base64 for embedding directly into the output notebook.

### Stage 2: Planning

Analyzes the paper and plans the notebook structure. This is the most complex stage with four sub-steps:

**2a. Paper Analysis** — Extracts title, methodology (500+ words), datasets, metrics, hyperparameters, training details, and model architecture using **Nemotron Super 49B** with Pydantic structured output (`PaperAnalysis` schema via NIM `guided_json`).

**2b. Repository Discovery** — A strict 4-tier strategy to find the paper's official code, designed to avoid false matches (e.g., returning `ossu/computer-science` for "Attention Is All You Need"):

1. LLM-extracted GitHub URL from paper text
2. Regex extraction of `github.com` URLs in the paper
3. Papers With Code API — exact arXiv ID match, then title search
4. Valyu academic search API as fallback

When a repo is found, priority source files are scraped (`model.py`, `train.py`, `config.py`, `attention.py`, etc.) and injected into generation prompts.

**2c. Config Extraction** — Pulls all hyperparameters, dimensions, learning rates into a structured `ConfigExtraction` schema. When datasets are unavailable, plans synthetic data generation via NVIDIA NeMo Data Designer.

**2d. Notebook Architecture** — Designs 20–30 cell structure with dependency ordering (`NotebookStructure` schema): title → setup → config → data → model components → training → evaluation → visualization. Every code cell is preceded by a markdown cell explaining what follows.

### Stage 3: Analysis

Produces detailed implementation logic for every notebook cell using **Nemotron Nano 9B** (efficient per-cell analysis). Each analysis includes exact algorithms, equations, API calls, tensor shapes, and cross-cell variable references — detailed enough that the code generator never needs to re-read the paper.

### Stage 4: Code Generation

Generates each cell with a **State Manifest** — the key innovation that prevents disconnected code:

```
┌─────────────────────────────────────────────┐
│              State Manifest                  │
│                                              │
│  After Cell 3:                               │
│    imports: {torch, nn, F, math}             │
│    classes: {MultiHeadAttention}             │
│    functions: {scaled_dot_product_attention}  │
│    variables: {d_model=512, n_heads=8}       │
│                                              │
│  → Injected into Cell 4's generation prompt  │
│  → Cell 4 references existing variables      │
│  → No duplicate imports, no undefined vars   │
└─────────────────────────────────────────────┘
```

After each code cell is generated, it is parsed with Python's `ast` module to extract all defined imports, functions, classes, and variables. This manifest is passed to every subsequent cell's prompt, ensuring the entire notebook forms a coherent, connected program.

**Dual-model strategy:** **Nemotron Super 49B** handles code cells (complex reasoning), while **Nemotron Nano 9B** handles markdown cells (simpler, faster).

### Stage 5: Assembly & Debug

Post-processes the generated cells into a valid `.ipynb` file:

- **Import deduplication** — Consolidates all import statements into a single setup cell, removing duplicates across the entire notebook
- **Broken reference removal** — Fixes hallucinated `from cell_6 import X` patterns that LLMs sometimes generate
- **Figure embedding** — Attaches paper figures directly into markdown cells via nbformat image attachments
- **AST validation** — Static analysis catches undefined variables, syntax errors, and type mismatches
- **LLM-powered debug** — Remaining issues are fixed by **Nemotron Super 49B** in iterative debug passes (configurable retries)

---

## NVIDIA Technologies Used

### NIM Models

| Model | Role | Parameters |
|-------|------|------------|
| `nvidia/llama-3.3-nemotron-super-49b-v1` | Orchestrator — paper analysis, notebook design, code generation, debugging | 49B |
| `nvidia/nvidia-nemotron-nano-9b-v2` | Sub-agent — per-cell analysis, markdown generation, config extraction | 9B |
| `nvidia/nemotron-nano-12b-v2-vl` | Vision — PDF figure description, diagram understanding | 12B |

All models are accessed through **NVIDIA NIM** (NVIDIA Inference Microservices) via the OpenAI-compatible API at `integrate.api.nvidia.com`.

### Structured Output with NIM `guided_json`

All LLM calls that require structured data use **Pydantic models** with NVIDIA NIM's `guided_json` parameter. This guarantees valid JSON responses from the model, eliminating the regex-based extraction failures that plague naive approaches.

```python
response = client.chat.completions.create(
    model="nvidia/llama-3.3-nemotron-super-49b-v1",
    messages=[...],
    extra_body={
        "nvext": {
            "guided_json": PaperAnalysis.model_json_schema()
        }
    }
)
result = PaperAnalysis.model_validate_json(response.choices[0].message.content)
```

Pydantic schemas: `PaperAnalysis`, `ConfigExtraction`, `NotebookStructure`, `DatasetInfo`, `TrainingDetails`, `ModelArchitecture`, `NotebookCell`.

### Additional NVIDIA Stack

- **NVIDIA NeMo Agent Toolkit** — ReAct agent orchestration with registered tool functions (`workflow.yaml`), automatic retries, and structured tool responses
- **LangChain ChatNVIDIA** — LangChain integration for NIM endpoints with `.with_structured_output()` support
- **NVIDIA NeMo Data Designer** — Synthetic dataset generation microservice for when paper datasets are unavailable
- **Arize Phoenix** — Optional observability and tracing for all NIM calls, configured in `workflow.yaml`

---

## Recursive Language Model (RLM) Context

For long papers that exceed context limits, the pipeline uses a **Recursive Language Model** approach (`rlm.py`):

- The paper is parsed into named sections (Abstract, Methodology, Experiments, etc.)
- Instead of stuffing the full paper into every prompt, sections are queried on-demand
- Long sections are recursively chunked and aggregated
- Stage-aware context curation provides only relevant sections per pipeline stage (AOrchestra pattern)

This prevents context rot and ensures the model focuses on the most relevant information for each task.

---

## Tech Stack

### Desktop Application

| Layer | Technology |
|-------|-----------|
| Shell | Electron 28 |
| Frontend | Next.js 15 (App Router), React 19, TypeScript |
| Styling | Tailwind CSS v4 (`@theme inline`), Geist fonts |
| Bridge | FastAPI WebSocket server (Python, auto-spawned) |
| IPC | Electron `contextBridge` + `ipcMain` handlers |

### Backend Pipeline

| Component | Technology |
|-----------|-----------|
| PDF parsing | `pymupdf4llm` with OCR |
| LLM inference | NVIDIA NIM via OpenAI SDK |
| Structured output | Pydantic v2 + NIM `guided_json` |
| Agent framework | NeMo Agent Toolkit + LangChain ChatNVIDIA |
| Repo discovery | Papers With Code API, GitHub REST API, Valyu API |
| Notebook format | `nbformat` v4 |
| Observability | Arize Phoenix (optional) |
| Logging | Rich console → WebSocket real-time streaming |

---

## Quick Start

### Prerequisites

- **Node.js** 18+ — [nodejs.org](https://nodejs.org)
- **Python** 3.10+ — [python.org](https://python.org)
- **NVIDIA API Key** (free) — [build.nvidia.com](https://build.nvidia.com)

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/paper2notebook.git
cd paper2notebook

# Install Node.js dependencies
npm install

# Install Python dependencies
pip install -r backend/requirements.txt
```

### 2. Run the app

```bash
npm run electron:dev
```

This launches:
1. **Next.js** dev server on port 3000 (hot reload)
2. **Electron** window loading from the dev server
3. **Python FastAPI** server on port 9847 (auto-spawned by Electron)

### 3. Add your API key

Click the **⚙** gear icon in the titlebar → enter your NVIDIA API key → **Save**.

Keys are stored locally at `~/.paper2notebook/api_keys.json` and never transmitted anywhere except to NVIDIA's NIM endpoints.

### 4. Upload a paper

Drag and drop any research paper PDF. The pipeline log streams every step in real-time — PDF processing, figure extraction, repo discovery, per-cell generation, and assembly.

---

## API Keys

| Key | Required | Purpose | Get it at |
|-----|----------|---------|-----------|
| NVIDIA API Key | **Yes** | NIM model inference (Nemotron Super, Nano, VL) | [build.nvidia.com](https://build.nvidia.com) |
| GitHub Token | No | Higher GitHub API rate limits for repo discovery | [github.com/settings/tokens](https://github.com/settings/tokens) |
| Valyu API Key | No | Academic paper search (fallback repo discovery) | [platform.valyu.ai](https://platform.valyu.ai) |

---

## Project Structure

```
paper2notebook/
├── electron/
│   ├── main.js                  # Electron main process — spawns Python server,
│   │                            #   window creation, IPC handlers
│   └── preload.ts               # Context bridge for secure renderer ↔ main IPC
│
├── src/app/
│   ├── layout.tsx               # Root layout (Geist fonts, metadata)
│   ├── page.tsx                 # App state machine: splash → upload → workspace
│   ├── globals.css              # Tailwind v4 dark theme (teal accents, animations)
│   ├── types.ts                 # TypeScript interfaces (events, sessions, notebook)
│   ├── components/
│   │   ├── SplashScreen.tsx     # Animated loading screen
│   │   ├── UploadView.tsx       # PDF drag-and-drop upload
│   │   ├── WorkspaceView.tsx    # Split panel (PDF + pipeline log + notebook)
│   │   ├── PipelineLog.tsx      # Real-time scrolling log with progress bar
│   │   ├── NotebookViewer.tsx   # Inline .ipynb renderer with syntax highlighting
│   │   └── SettingsModal.tsx    # API key management
│   └── hooks/
│       └── usePipeline.ts       # WebSocket connection hook
│
├── backend/
│   ├── server.py                # FastAPI bridge — WebSocket streaming, REST endpoints
│   ├── config.yaml              # Pipeline configuration (models, retries, sections)
│   ├── workflow.yaml            # NeMo Agent Toolkit workflow definition
│   ├── schemas.py               # Pydantic schemas for NIM guided_json
│   ├── orchestrator.py          # Agentic orchestrator (NIM + LangChain ChatNVIDIA)
│   ├── pdf_processor.py         # PDF → markdown + figure extraction + vision model
│   ├── planning.py              # Paper analysis, repo discovery, notebook architecture
│   ├── analysis.py              # Per-cell implementation analysis
│   ├── code_generation.py       # State manifest code generation
│   ├── notebook_assembler.py    # Import dedup, figure embedding, AST validation
│   ├── web_tools.py             # Papers With Code, GitHub API, Valyu integration
│   ├── rlm.py                   # Recursive Language Model context management
│   ├── synthetic_data.py        # NeMo Data Designer synthetic dataset generation
│   ├── utils.py                 # NIM client, cost tracking, Phoenix observability
│   ├── main.py                  # CLI entry point (standalone usage)
│   └── requirements.txt         # Python dependencies
│
├── package.json
├── next.config.ts
├── tsconfig.json
├── electron-builder.yml
└── README.md
```

---

## Configuration

Pipeline behavior is controlled by `backend/config.yaml`:

```yaml
models:
  orchestrator: "nvidia/llama-3.3-nemotron-super-49b-v1"   # Complex reasoning
  sub_agent: "nvidia/nvidia-nemotron-nano-9b-v2"            # Fast per-cell analysis
  vision: "nvidia/nemotron-nano-12b-v2-vl"                  # Figure understanding

pipeline:
  max_retries: 2          # Debug iterations for notebook validation

notebook:
  max_cells: 50           # Upper bound on generated cells
  include_explanations: true
  include_setup_cell: true
```

---

## Build for Distribution

```bash
npm run electron:build
# Output: ./release/
#   macOS  → .dmg + .zip
#   Windows → .exe (NSIS installer)
#   Linux  → .AppImage + .deb
```

---

## Cost Tracking

Every NIM API call is tracked with token counts, latency, and estimated cost. The cost summary is saved to `cost_summary.json` in the session output directory and displayed in the Pipeline Log upon completion.

---

## CLI Usage (Without Electron)

The pipeline can also be run directly from the command line:

```bash
cd backend
python main.py --pdf /path/to/paper.pdf --output ./output
```

---

## Acknowledgements

Built with NVIDIA's AI infrastructure:

- [NVIDIA NIM](https://build.nvidia.com) — Inference microservices for Nemotron models
- [NVIDIA NeMo Agent Toolkit](https://docs.nvidia.com/nemo/agent-toolkit/) — Agentic AI framework
- [NVIDIA NeMo Data Designer](https://docs.nvidia.com/nemo/microservices/latest/design-synthetic-data-from-scratch-or-seeds/) — Synthetic data generation
- [LangChain ChatNVIDIA](https://python.langchain.com/docs/integrations/chat/nvidia_ai_endpoints/) — LangChain + NIM integration

