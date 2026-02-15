"""
Stage 2: Analysis — Detailed logic for each notebook cell.

Uses NVIDIA NIM (Nemotron Nano) for efficient per-cell analysis.
Provides FULL paper text so every detail is available.
"""

import json
from orchestrator import Orchestrator
from utils import save_artifact, log_stage, console


ANALYSIS_SYSTEM = """You are an expert researcher and Python developer using NVIDIA NIM.
Produce a detailed implementation analysis for this Jupyter notebook cell.

Your analysis must be precise enough for a code generator to write the cell
WITHOUT re-reading the paper. Include:
1. Exact algorithms, equations, or procedures from the paper
2. Required imports and library API calls (be specific: torch.nn.Module, etc.)
3. Variable names, data structures, tensor shapes
4. How to connect to previous cells (what variables/objects to reuse)
5. Edge cases and error handling
6. If dataset cell: how to load or create synthetic version

Be specific. Quote exact values. No vagueness."""


def run_analysis(
    orchestrator: Orchestrator,
    paper_md: str,
    planning_result: dict,
    output_dir: str,
) -> dict[int, str]:
    """Produce per-cell analysis. Returns {cell_id: analysis_text}."""
    log_stage("Stage 2: Analysis", "Detailed logic for each notebook cell")

    structure = planning_result["notebook_structure"]
    analysis = planning_result["analysis"]
    config = planning_result["config"]
    repo_info = planning_result.get("repo_info", {})
    cells = structure.get("cells", [])

    full_paper = paper_md[:24000]
    cell_analyses: dict[int, str] = {}

    for i, cell in enumerate(cells):
        cid = cell.get("cell_id", i + 1)
        ctype = cell.get("cell_type", "code")
        title = cell.get("title", f"Cell {cid}")

        console.print(f"\n  [bold]Cell {cid}/{len(cells)}: {title}[/bold]")

        if ctype == "markdown":
            cell_analyses[cid] = (
                f"Markdown cell: {title}\n"
                f"Description: {cell.get('description', '')}\n"
                f"Paper title: {analysis.get('title', '')}")
            console.print("    [dim]Markdown — minimal analysis[/dim]")
            continue

        ctx = _build_context(cell, analysis, config, repo_info,
                             full_paper, cell_analyses, structure)

        result = orchestrator.call_nim(
            system=ANALYSIS_SYSTEM, context=ctx,
            name=f"Analyzer-Cell{cid}",
            model=orchestrator.sub_agent_model, max_tokens=3072,
        )
        cell_analyses[cid] = result["content"]
        save_artifact(result["content"], f"{output_dir}/analysis/cell_{cid}_analysis.txt")

    return cell_analyses


def _build_context(cell, analysis, config, repo_info, paper, prev, structure):
    parts = [
        f"## Cell to Analyze\nID: {cell.get('cell_id')}\n"
        f"Title: {cell.get('title')}\nType: {cell.get('cell_type')}\n"
        f"Description: {cell.get('description')}\nDepends on: {cell.get('dependencies', [])}",
        f"\n## Full Paper\n{paper}",
        f"\n## Config\n{json.dumps(config, indent=2, default=str)[:3000]}",
    ]

    repo_files = repo_info.get("repo_files", {})
    if repo_files:
        parts.append("\n## Existing Repo Code")
        for fn, code in list(repo_files.items())[:3]:
            if not fn.startswith("_"):
                parts.append(f"### {fn}\n```\n{code[:1500]}\n```")

    deps = cell.get("dependencies", [])
    recent = {k: v for k, v in prev.items()
              if k in deps or k in list(prev.keys())[-2:]}
    if recent:
        parts.append("\n## Previous Analyses")
        for pid, pa in recent.items():
            parts.append(f"### Cell {pid}\n{pa[:600]}")

    parts.append("\n## Notebook Structure")
    for c in structure.get("cells", []):
        m = "→" if c.get("cell_id") == cell.get("cell_id") else " "
        parts.append(f"  {m} Cell {c.get('cell_id')}: [{c.get('cell_type')}] {c.get('title')}")

    return "\n".join(parts)