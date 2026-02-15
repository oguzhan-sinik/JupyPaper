"""
Paper2Notebook — Main entry point.

Full 10-step pipeline using NVIDIA's AI ecosystem:
  - NVIDIA NIM for all LLM inference (Nemotron models)
  - NVIDIA NeMo Agent Toolkit for agentic orchestration & observability
  - NVIDIA LangChain integration (ChatNVIDIA) for model interface
  - Valyu Search / GitHub API / Papers With Code for repo discovery
  - NVIDIA NeMo Data Designer for synthetic dataset generation

Steps:
  1-5: PDF → Markdown → Clean → Vision descriptions → Compress + Figure extraction
  6:   Planning (repo search, paper analysis, config extraction)
  7:   Notebook architecture design
  8:   Per-cell analysis (full paper context)
  9:   Code generation (with state manifest)
  10:  Assembly (import dedup, ref fixing, figure embedding) + Debugging
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from utils import (
    load_config, get_nim_client, CostTracker, log_stage,
    estimate_tokens, init_phoenix, console,
)
from pdf_processor import process_pdf, PDFResult
from orchestrator import Orchestrator
from planning import run_planning
from analysis import run_analysis
from code_generation import run_code_generation
from notebook_assembler import (
    assemble_notebook, validate_notebook, debug_notebook, save_notebook,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Paper2Notebook: Papers → Notebooks (NVIDIA NIM + NeMo)")
    p.add_argument("--pdf_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./output")
    p.add_argument("--paper_name", type=str, default="")
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--model", type=str, default="", help="Override orchestrator model")
    p.add_argument("--sub_model", type=str, default="", help="Override sub-agent model")
    p.add_argument("--skip_vision", action="store_true")
    p.add_argument("--skip_debug", action="store_true")
    p.add_argument("--skip_repo_search", action="store_true")
    p.add_argument("--skip_figures", action="store_true", help="Skip figure embedding")
    p.add_argument("--markdown_path", type=str, default="", help="Pre-computed markdown")
    p.add_argument("--phoenix", action="store_true", help="Enable Arize Phoenix observability")
    p.add_argument("--phoenix_endpoint", type=str, default="http://localhost:6006/v1/traces")
    p.add_argument("--debug", action="store_true", help="Verbose output")
    return p.parse_args()


def main():
    args = parse_args()
    load_dotenv()

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    m = cfg.get("models", {})
    orch_model = args.model or m.get("orchestrator", "nvidia/llama-3.3-nemotron-super-49b-v1.5")
    sub_model = args.sub_model or m.get("sub_agent", "nvidia/nvidia-nemotron-nano-9b-v2")
    vis_model = m.get("vision", "nvidia/nemotron-nano-12b-v2-vl")
    base_url = m.get("base_url", "https://integrate.api.nvidia.com/v1")

    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        console.print("[red]Error: Set NVIDIA_API_KEY (get from build.nvidia.com)[/red]")
        sys.exit(1)

    paper_name = args.paper_name or os.path.splitext(os.path.basename(args.pdf_path))[0]
    output_dir = os.path.join(args.output_dir, paper_name)
    for d in [output_dir, f"{output_dir}/analysis", f"{output_dir}/cells"]:
        os.makedirs(d, exist_ok=True)

    client = get_nim_client(api_key, base_url)
    cost_tracker = CostTracker()

    # ── Arize Phoenix Observability ───────────────────────────────────────────
    phoenix_active = False
    if args.phoenix:
        phoenix_active = init_phoenix(
            endpoint=args.phoenix_endpoint, project="paper2notebook")

    console.print(Panel(
        f"[bold cyan]Paper2Notebook[/bold cyan]\n\n"
        f"Paper: {args.pdf_path}\n"
        f"[bold]NVIDIA Stack:[/bold]\n"
        f"  NIM Orchestrator: {orch_model}\n"
        f"  NIM Sub-agent:    {sub_model}\n"
        f"  NIM Vision:       {vis_model}\n"
        f"  NeMo Agent Toolkit: Orchestration + Observability\n"
        f"  LangChain-NVIDIA:   ChatNVIDIA interface\n"
        f"  Arize Phoenix:      {'✓ Active → ' + args.phoenix_endpoint if phoenix_active else '✗ Disabled (use --phoenix)'}\n"
        f"[bold]Discovery:[/bold] Papers With Code API → GitHub API → Valyu Search\n"
        f"Output: {output_dir}",
        title="Configuration", border_style="cyan",
    ))

    # ═════════════════════════════════════════════════════════════════════════
    # Steps 1-5: PDF → Clean Markdown + Figure Extraction
    # ═════════════════════════════════════════════════════════════════════════
    figures = []

    if args.markdown_path:
        log_stage("Steps 1-5: Load Markdown", f"From {args.markdown_path}")
        with open(args.markdown_path, "r") as f:
            paper_md = f.read()
    else:
        log_stage("Steps 1-5: PDF Processing",
                  "PDF → MD → strip sections → vision (NVIDIA NIM) → compress + figure extraction")
        vis_client = None if args.skip_vision else client
        pdf_result = process_pdf(
            pdf_path=args.pdf_path,
            vision_client=vis_client,
            vision_model=vis_model,
            jpeg_quality=cfg.get("pdf", {}).get("vision_jpeg_quality", 75),
            extra_remove=set(cfg.get("remove_sections", [])),
        )
        paper_md = pdf_result.markdown
        if not args.skip_figures:
            figures = pdf_result.figures
            console.print(f"  Extracted {len(figures)} figure(s) for notebook embedding")

    md_path = f"{output_dir}/paper.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(paper_md)
    tok = estimate_tokens(paper_md)
    console.print(f"  Paper: {len(paper_md):,} chars (~{tok:,} tokens)")

    # ═════════════════════════════════════════════════════════════════════════
    # Initialize NVIDIA NeMo Orchestrator
    # ═════════════════════════════════════════════════════════════════════════
    orchestrator = Orchestrator(
        client=client,
        orchestrator_model=orch_model,
        sub_agent_model=sub_model,
        cost_tracker=cost_tracker,
        api_key=api_key,
        base_url=base_url,
    )

    # ═════════════════════════════════════════════════════════════════════════
    # Steps 6-7: Planning (PwC + GitHub + Valyu + NVIDIA NIM)
    # ═════════════════════════════════════════════════════════════════════════
    planning_result = run_planning(orchestrator, paper_md, output_dir)

    # ═════════════════════════════════════════════════════════════════════════
    # Step 8: Analysis (NVIDIA NIM Nemotron Nano)
    # ═════════════════════════════════════════════════════════════════════════
    cell_analyses = run_analysis(orchestrator, paper_md, planning_result, output_dir)

    # ═════════════════════════════════════════════════════════════════════════
    # Step 9: Code Generation (NVIDIA NIM Nemotron Super + State Manifest)
    # ═════════════════════════════════════════════════════════════════════════
    generated_cells = run_code_generation(
        orchestrator, paper_md, planning_result, cell_analyses, output_dir)

    # ═════════════════════════════════════════════════════════════════════════
    # Step 10: Assembly (Import Dedup + Ref Fix + Figure Embed) + Debug
    # ═════════════════════════════════════════════════════════════════════════
    nb_cfg = cfg.get("notebook", {})
    nb = assemble_notebook(
        generated_cells,
        planning_result,
        kernel=nb_cfg.get("kernel", "python3"),
        language=nb_cfg.get("language", "python"),
        figures=figures,
    )

    if not args.skip_debug:
        issues = validate_notebook(nb, orchestrator)
        retries = cfg.get("pipeline", {}).get("max_retries", 2)
        attempt = 0
        while issues and attempt < retries:
            nb = debug_notebook(nb, issues, orchestrator, planning_result)
            issues = validate_notebook(nb)
            attempt += 1

    notebook_path = f"{output_dir}/{paper_name}.ipynb"
    save_notebook(nb, notebook_path)
    cost_tracker.save(f"{output_dir}/cost_summary.json")

    # ── Summary ───────────────────────────────────────────────────────────────
    cn = sum(1 for c in nb.cells if c.cell_type == "code")
    mn = sum(1 for c in nb.cells if c.cell_type == "markdown")
    repo = planning_result.get("repo_info", {}).get("repo_url", "None found")
    repo_source = planning_result.get("repo_info", {}).get("source", "N/A")
    data_strategy = planning_result.get("config", {}).get("data_strategy", "unknown")
    n_figs = len(figures)

    console.print(Panel(
        f"[bold green]Pipeline Complete[/bold green]\n\n"
        f"Notebook: {notebook_path}\n"
        f"Cells: {len(nb.cells)} ({cn} code, {mn} markdown)\n"
        f"Figures embedded: {n_figs}\n"
        f"Repo: {repo} (via {repo_source})\n"
        f"Data strategy: {data_strategy}\n\n"
        f"[bold]NVIDIA NIM Cost Summary:[/bold]\n"
        f"{cost_tracker.summary()}",
        title="Results", border_style="green",
    ))


if __name__ == "__main__":
    main()