"""
Stage 1: Planning — Analyze paper, discover repos, plan notebook.

Key improvements:
  - Updated web_tools integration (Papers With Code API, GitHub API, Valyu)
  - Richer notebook architecture: 20-30 cells with markdown between every code section
  - Explicit dependency ordering for PaperCoder-style generation
  - Synthetic data planning via NVIDIA NeMo Data Designer when datasets unavailable
  - Better analysis prompts extracting more paper details

Steps:
  6a. Search for existing code repos (PwC → GitHub API → Valyu)
  6b. Analyze paper: methodology, datasets, metrics, hyperparameters
  6c. Extract configs — if no dataset, plan synthetic data via NeMo
  7.  Architect notebook with dependency-ordered cells
"""

import json
from orchestrator import Orchestrator
from web_tools import search_for_repo
from schemas import PaperAnalysis, ConfigExtraction, NotebookStructure
from utils import save_artifact, log_stage, console


ANALYZER_SYSTEM = """You are an expert AI research analyst. Analyze this paper thoroughly.
Extract every detail needed to reproduce the implementation.

Be detailed in the methodology field (500+ words): include architecture, data flow,
key innovations, and how components interact.

For code_repo_mentioned: look carefully for any GitHub URL, code link, or "code available at" 
statement in the paper text. If found, include the EXACT URL. If not found, set to null.

Use null for anything not stated. Do NOT fabricate values."""


ARCHITECT_SYSTEM = """You are a Jupyter notebook architect creating an educational, reproducible notebook.
Design a notebook that teaches the reader about this paper while building a working implementation.

CRITICAL RULES:
- notebook_title MUST be the actual paper title — NOT "Title and Description"
- Every code section MUST be preceded by a markdown cell explaining what follows
- Target 20-30 cells (roughly 50/50 markdown and code)
- cells MUST specify dependencies (which prior cell_ids they need)
- Generate cells in logical execution order

REQUIRED STRUCTURE:
1. [markdown] Title + Abstract + Paper link
2. [code] Setup: pip installs + ALL imports (one consolidated cell)
3. [markdown] Configuration explanation
4. [code] Configuration: all hyperparameters as a config dict
5. [markdown] Dataset explanation
6. [code] Dataset: load real data OR create synthetic data with comments
7. [markdown] Architecture overview with key equations
8-N. [code+markdown pairs] Model components (one class/function per cell)
N+1. [markdown] Training explanation
N+2. [code] Training loop
N+3. [markdown] Evaluation explanation
N+4. [code] Evaluation + metrics
N+5. [code] Visualization + plots
N+6. [markdown] Summary + results discussion"""


CONFIG_SYSTEM = """Extract ALL configuration values and hyperparameters from the paper.
Be thorough — include every number, dimension, rate, and setting mentioned.
Only use values explicitly from the paper. Use null for anything not stated. Include units."""


def run_planning(orchestrator: Orchestrator, paper_md: str, output_dir: str) -> dict:
    """Run full planning stage. Returns {analysis, notebook_structure, config, repo_info}."""
    log_stage("Stage 1: Planning", "Analyze paper → search repos → extract config → design notebook")

    # Truncate for context windows
    full = paper_md[:28000]
    short = paper_md[:16000]

    # ── 1. Paper Analysis (Pydantic structured output) ────────────────────────
    console.print("\n[bold]Step 1/4: Paper Analysis[/bold]")
    analysis_obj = orchestrator.call_nim_structured(
        system=ANALYZER_SYSTEM, context=full, schema=PaperAnalysis,
        name="PaperAnalyzer", model=orchestrator.orchestrator_model, max_tokens=4096,
    )
    if analysis_obj is not None:
        analysis = analysis_obj.model_dump(mode="json")
    else:
        # Total failure — use minimal dict so pipeline continues
        console.print("    [yellow]Using minimal fallback analysis[/yellow]")
        analysis = {"title": "Unknown Paper", "problem": "", "methodology": "",
                     "key_components": [], "datasets": [], "metrics": [],
                     "hyperparameters": {}, "baselines": [], "dependencies": [],
                     "training_details": {}, "model_architecture": {},
                     "code_repo_mentioned": None}
    save_artifact(analysis, f"{output_dir}/planning_analysis.json")

    # ── 2. Repo Discovery (exact URL from paper only) ─────────────────────────
    console.print("\n[bold]Step 2/4: Repository Discovery[/bold]")
    title = analysis.get("title", "")
    mentioned = analysis.get("code_repo_mentioned")

    repo_info = search_for_repo(
        paper_title=title,
        paper_text=full,  # pass full paper text for URL extraction
        mentioned_url=mentioned,
    )
    save_artifact(repo_info, f"{output_dir}/planning_repo_info.json")

    if repo_info.get("repo_url"):
        console.print(f"  [green]✓ Found repo via {repo_info.get('source')}: {repo_info['repo_url']}[/green]")
    else:
        console.print("  [dim]No repo found — generating from paper only[/dim]")

    # ── 3. Config Extraction (Pydantic structured output) ─────────────────────
    console.print("\n[bold]Step 3/4: Configuration Extraction[/bold]")
    config_obj = orchestrator.call_nim_structured(
        system=CONFIG_SYSTEM, context=short, schema=ConfigExtraction,
        name="ConfigExtractor", model=orchestrator.sub_agent_model, max_tokens=2048,
    )
    if config_obj is not None:
        config = config_obj.model_dump(mode="json")
    else:
        config = {"training": {}, "model": {}, "data": {}, "evaluation": {}, "other": {}}

    # Determine data strategy
    datasets = analysis.get("datasets", [])
    has_public_dataset = bool(datasets and any(
        (d.get("url_or_source") and d.get("url_or_source") != "not provided"
         and d.get("is_public", False))
        if isinstance(d, dict) else False
        for d in datasets
    ))
    config["has_dataset"] = has_public_dataset
    config["data_strategy"] = "real" if has_public_dataset else "synthetic"

    if not has_public_dataset:
        console.print("  [yellow]No public dataset found — will plan synthetic data generation[/yellow]")
        config["synthetic_data_plan"] = _plan_synthetic_data(analysis, config)

    save_artifact(config, f"{output_dir}/planning_config.json")

    # ── 4. Notebook Architecture ──────────────────────────────────────────────
    console.print("\n[bold]Step 4/4: Notebook Architecture[/bold]")
    arch_ctx = [
        f"## Paper Analysis\n{json.dumps(analysis, indent=2, default=str)[:6000]}",
        f"\n## Configuration\n{json.dumps(config, indent=2, default=str)[:3000]}",
    ]

    if repo_info.get("repo_files"):
        arch_ctx.append(
            f"\n## Existing Code Available ({repo_info.get('repo_url', '')})\n"
            f"Files found: {', '.join(repo_info['repo_files'].keys())}\n"
            "IMPORTANT: Use this code as reference. The notebook should closely follow "
            "the repo's architecture when possible."
        )
        # Show key file snippets
        for k, v in list(repo_info["repo_files"].items())[:4]:
            if not k.startswith("_"):
                arch_ctx.append(f"### {k}\n```\n{v[:1500]}\n```")

    if not has_public_dataset:
        arch_ctx.append(
            "\n## DATA STRATEGY: SYNTHETIC\n"
            "No public dataset URL found. The notebook MUST include:\n"
            "- A markdown cell explaining the synthetic data approach\n"
            "- A code cell creating realistic synthetic data matching paper's specs\n"
            "- Comments explaining what real data would look like\n"
            f"- Data specs: {json.dumps(config.get('synthetic_data_plan', {}), indent=2, default=str)[:1000]}"
        )

    structure_obj = orchestrator.call_nim_structured(
        system=ARCHITECT_SYSTEM, context="\n".join(arch_ctx), schema=NotebookStructure,
        name="NotebookArchitect", model=orchestrator.orchestrator_model, max_tokens=4096,
    )
    if structure_obj is not None:
        structure = structure_obj.model_dump(mode="json")
    else:
        structure = _default_structure(analysis, has_public_dataset, repo_info)

    # Fix generic titles
    real_title = analysis.get("title", "")
    if real_title:
        nt = structure.get("notebook_title", "").lower()
        if nt in ["title and description", "## title and description", ""]:
            structure["notebook_title"] = real_title

    # Ensure generation_order exists
    if "generation_order" not in structure:
        structure["generation_order"] = [c["cell_id"] for c in structure.get("cells", [])]

    save_artifact(structure, f"{output_dir}/planning_structure.json")

    return {
        "analysis": analysis,
        "notebook_structure": structure,
        "config": config,
        "repo_info": repo_info,
    }


def _plan_synthetic_data(analysis: dict, config: dict) -> dict:
    """
    Plan synthetic data generation based on paper's dataset description.

    Returns a spec that can be used to either:
      - Generate data inline in the notebook (simple approach)
      - Call NVIDIA NeMo Data Designer API (advanced approach)
    """
    datasets = analysis.get("datasets", [])
    data_config = config.get("data", {})
    model_config = config.get("model", {})

    plan = {
        "approach": "inline_synthetic",  # or "nemo_data_designer"
        "description": "Synthetic data matching paper specifications",
        "datasets_described": datasets,
        "input_format": data_config.get("input_format"),
        "sequence_length": data_config.get("sequence_length") or model_config.get("max_seq_length"),
        "vocab_size": data_config.get("vocab_size") or model_config.get("vocab_size"),
        "train_samples": 1000,  # default for demo
        "val_samples": 200,
        "columns": [],
    }

    # Infer data structure from paper
    arch_type = model_config.get("architecture", "").lower()
    if any(k in arch_type for k in ["transformer", "seq2seq", "translation", "language"]):
        plan["data_type"] = "sequence_pair"
        plan["columns"] = [
            {"name": "source", "type": "token_sequence", "max_length": plan["sequence_length"] or 512},
            {"name": "target", "type": "token_sequence", "max_length": plan["sequence_length"] or 512},
        ]
    elif any(k in arch_type for k in ["classification", "bert", "encoder"]):
        plan["data_type"] = "text_classification"
        plan["columns"] = [
            {"name": "text", "type": "text", "max_length": plan["sequence_length"] or 512},
            {"name": "label", "type": "categorical", "num_classes": 2},
        ]
    elif any(k in arch_type for k in ["generation", "gpt", "decoder", "lm"]):
        plan["data_type"] = "language_modeling"
        plan["columns"] = [
            {"name": "text", "type": "text", "max_length": plan["sequence_length"] or 1024},
        ]
    else:
        plan["data_type"] = "generic"
        plan["columns"] = [
            {"name": "input", "type": "tensor", "shape": "infer_from_model"},
            {"name": "target", "type": "tensor", "shape": "infer_from_model"},
        ]

    return plan


def _default_structure(analysis: dict, has_dataset: bool, repo_info: dict) -> dict:
    """Generate a rich default notebook structure when NIM fails to produce JSON."""
    t = analysis.get("title", "Paper Reproduction")
    methodology = analysis.get("methodology", "")
    key_components = analysis.get("key_components", [])

    cells = [
        # Introduction
        {"cell_id": 1, "cell_type": "markdown", "title": t,
         "description": "Title, abstract, paper link, and notebook overview",
         "dependencies": [], "estimated_complexity": "low"},

        # Setup
        {"cell_id": 2, "cell_type": "code", "title": "Setup & Imports",
         "description": "Install dependencies and import all required libraries",
         "dependencies": [], "estimated_complexity": "low"},

        # Configuration
        {"cell_id": 3, "cell_type": "markdown", "title": "Configuration & Hyperparameters",
         "description": "Explain the hyperparameters and their values from the paper",
         "dependencies": [], "estimated_complexity": "low"},
        {"cell_id": 4, "cell_type": "code", "title": "Configuration",
         "description": "Define all hyperparameters as a config dictionary matching the paper",
         "dependencies": [2], "estimated_complexity": "low"},

        # Data
        {"cell_id": 5, "cell_type": "markdown", "title": "Dataset Preparation",
         "description": "Explain the dataset, format, and preprocessing pipeline",
         "dependencies": [], "estimated_complexity": "low"},
        {"cell_id": 6, "cell_type": "code", "title": "Dataset & DataLoader",
         "description": "Load real data or create synthetic data matching paper specifications",
         "dependencies": [2, 4], "estimated_complexity": "medium"},

        # Model Architecture
        {"cell_id": 7, "cell_type": "markdown", "title": "Model Architecture",
         "description": "Detailed explanation of the model architecture with equations",
         "dependencies": [], "estimated_complexity": "low"},
    ]

    # Add component cells based on what we know about the architecture
    cid = 8
    if key_components:
        for comp in key_components[:4]:
            cells.append({
                "cell_id": cid, "cell_type": "code",
                "title": f"{comp} Implementation",
                "description": f"Implement the {comp} component as described in the paper",
                "dependencies": [2, 4] + ([cid - 1] if cid > 8 else []),
                "estimated_complexity": "high",
            })
            cid += 1
    else:
        cells.append({
            "cell_id": cid, "cell_type": "code", "title": "Model Definition",
            "description": "Define the complete model architecture",
            "dependencies": [2, 4], "estimated_complexity": "high",
        })
        cid += 1

    # Training
    cells.extend([
        {"cell_id": cid, "cell_type": "markdown", "title": "Training Procedure",
         "description": "Explain optimizer, learning rate schedule, loss function, and training tricks",
         "dependencies": [], "estimated_complexity": "low"},
        {"cell_id": cid + 1, "cell_type": "code", "title": "Training Loop",
         "description": "Complete training loop with loss computation, backprop, and logging",
         "dependencies": [2, 4, 6, cid - 1], "estimated_complexity": "high"},

        # Evaluation
        {"cell_id": cid + 2, "cell_type": "markdown", "title": "Evaluation",
         "description": "Explain evaluation metrics and procedure",
         "dependencies": [], "estimated_complexity": "low"},
        {"cell_id": cid + 3, "cell_type": "code", "title": "Evaluation & Metrics",
         "description": "Evaluate the model and compute metrics from the paper",
         "dependencies": [cid + 1], "estimated_complexity": "medium"},

        # Results
        {"cell_id": cid + 4, "cell_type": "code", "title": "Results Visualization",
         "description": "Plot training curves, attention maps, or other relevant visualizations",
         "dependencies": [cid + 1, cid + 3], "estimated_complexity": "medium"},

        # Summary
        {"cell_id": cid + 5, "cell_type": "markdown", "title": "Summary & Discussion",
         "description": "Summarize results, compare with paper, discuss limitations",
         "dependencies": [], "estimated_complexity": "low"},
    ])

    return {
        "notebook_title": t,
        "notebook_description": f"Educational Jupyter notebook reproducing: {t}",
        "cells": cells,
        "estimated_packages": analysis.get("dependencies", ["torch", "numpy", "matplotlib", "tqdm"]),
        "generation_order": [c["cell_id"] for c in cells],
    }