"""
Stage 3: Code Generation — Write each notebook cell.

Key improvements over v1:
  - STATE MANIFEST: tracks all defined variables, functions, classes, and
    imported modules after each cell — passed to every subsequent generation
  - DEPENDENCY-AWARE: cells are generated in dependency order with full
    visibility into prior cells' code and state
  - RICHER PROMPTS: explicit instructions to NOT re-import, to reference
    prior variables by name, and to write connected, runnable code
  - REPO CONTEXT: when repo code is available, relevant files are injected
    into the prompt for each cell based on keyword matching

Uses NVIDIA NIM:
  - Nemotron Super (orchestrator) for code cells (complex reasoning)
  - Nemotron Nano (sub-agent) for markdown cells (simpler)
"""

import json
import ast
import re
from typing import Optional

from orchestrator import Orchestrator
from utils import save_artifact, extract_code, strip_think, log_stage, console


# ── State Manifest ────────────────────────────────────────────────────────────

class StateManifest:
    """
    Tracks the running state of the notebook as cells are generated.

    After each code cell, we parse it with AST to extract:
      - All import statements (module → alias)
      - All defined functions and classes
      - All top-level variable assignments
      - All data structures (DataLoaders, datasets, etc.)

    This manifest is injected into every subsequent cell's prompt so the
    LLM knows exactly what's available without needing to re-import or
    re-define anything.
    """

    def __init__(self):
        self.imports: dict[str, str] = {}     # module → "import X" or "from X import Y"
        self.functions: list[str] = []         # function names
        self.classes: list[str] = []           # class names
        self.variables: dict[str, str] = {}    # name → brief type hint
        self.all_code: dict[int, str] = {}     # cell_id → full source

    def update_from_code(self, cell_id: int, code: str):
        """Parse a code cell and update the manifest."""
        self.all_code[cell_id] = code
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name
                    self.imports[name] = f"import {alias.name}" + (
                        f" as {alias.asname}" if alias.asname else "")

            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                for alias in node.names:
                    name = alias.asname or alias.name
                    self.imports[name] = f"from {mod} import {alias.name}" + (
                        f" as {alias.asname}" if alias.asname else "")

            elif isinstance(node, ast.FunctionDef):
                args = [a.arg for a in node.args.args[:5]]
                self.functions.append(f"{node.name}({', '.join(args)})")

            elif isinstance(node, ast.ClassDef):
                bases = [getattr(b, "id", getattr(b, "attr", "?")) for b in node.bases[:3]]
                self.classes.append(f"{node.name}({', '.join(bases)})" if bases else node.name)

            elif isinstance(node, ast.Assign) and isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        # Try to infer type from value
                        type_hint = _infer_type(node.value)
                        self.variables[target.id] = type_hint

    def format_for_prompt(self, max_chars: int = 3000) -> str:
        """Format the manifest as a prompt-friendly string."""
        parts = ["## Available State (from previous cells — DO NOT re-import or redefine)"]

        if self.imports:
            parts.append("\n### Already Imported Modules")
            for name, stmt in sorted(self.imports.items()):
                parts.append(f"  - {stmt}")

        if self.classes:
            parts.append("\n### Defined Classes")
            for cls in self.classes:
                parts.append(f"  - {cls}")

        if self.functions:
            parts.append("\n### Defined Functions")
            for fn in self.functions:
                parts.append(f"  - {fn}")

        if self.variables:
            parts.append("\n### Available Variables")
            for name, type_hint in sorted(self.variables.items()):
                parts.append(f"  - {name}: {type_hint}")

        result = "\n".join(parts)
        return result[:max_chars]

    def get_recent_code(self, n: int = 2, max_chars: int = 3000) -> str:
        """Get the most recent N cells' code for context."""
        recent = sorted(self.all_code.keys())[-n:]
        parts = []
        for cid in recent:
            code = self.all_code[cid]
            parts.append(f"### Cell {cid}\n```python\n{code[:max_chars // n]}\n```")
        return "\n\n".join(parts)


def _infer_type(node) -> str:
    """Infer a simple type hint from an AST value node."""
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        elif isinstance(func, ast.Attribute):
            return f"{getattr(func.value, 'id', '?')}.{func.attr}"
    elif isinstance(node, ast.Dict):
        return "dict"
    elif isinstance(node, ast.List):
        return "list"
    elif isinstance(node, ast.Constant):
        return type(node.value).__name__
    return "unknown"


# ── System Prompts ────────────────────────────────────────────────────────────

CODE_SYSTEM = """You are an expert Python developer writing a Jupyter notebook cell.
This cell is part of a multi-cell notebook — previous cells have ALREADY defined
imports, classes, functions, and variables that you MUST reuse.

CRITICAL RULES:
1. Do NOT add ANY import statements — all imports are in the setup cell already
2. Do NOT redefine classes/functions from previous cells — use them directly
3. Reference variables from previous cells by their exact names
4. Write COMPLETE, RUNNABLE code — NO TODOs, NO placeholders, NO "implement this"
5. Add clear inline comments explaining key steps and paper references
6. Include proper error handling where appropriate
7. Use `device` variable (already defined) for GPU/CPU placement
8. Output ONLY Python code inside a single ```python block
9. Do NOT include <think> tags — ONLY code
10. If creating a model, assign it to a clearly named variable (e.g., `model = ...`)
11. If creating data, assign to clear names (e.g., `train_loader`, `dataset`)
12. Print informative output (shapes, sizes, sample data) so the user sees results"""

MD_SYSTEM = """You are writing an educational markdown cell for a Jupyter notebook
that teaches readers about a deep learning paper.

CRITICAL RULES:
1. Output ONLY raw markdown — no code fences, no ```markdown wrapper
2. Do NOT include <think> tags
3. Start directly with a markdown header (# or ##)
4. Write SUBSTANTIAL content: 8-20 lines of rich explanation
5. Use LaTeX equations where relevant: $inline$ or $$block$$
6. Reference specific methods, architectures, equations, and values FROM THE PAPER
7. Explain WHY each component matters, not just what it is
8. For architecture cells: describe data flow, dimensions, and key design choices
9. For results cells: explain metrics, baselines, and significance
10. Write as an expert teacher — assume the reader knows basic ML but not this paper
11. NEVER output just a title — always include substantive explanatory text"""


# ── Code Generation ──────────────────────────────────────────────────────────

def run_code_generation(
    orchestrator: Orchestrator,
    paper_md: str,
    planning_result: dict,
    cell_analyses: dict[int, str],
    output_dir: str,
) -> list[dict]:
    """Generate code/markdown for each cell with state tracking."""
    log_stage("Stage 3: Code Generation", "Writing notebook cells via NVIDIA NIM")

    structure = planning_result["notebook_structure"]
    config = planning_result["config"]
    analysis = planning_result["analysis"]
    repo_info = planning_result.get("repo_info", {})
    cells = structure.get("cells", [])

    manifest = StateManifest()
    generated: list[dict] = []

    for i, cell in enumerate(cells):
        cid = cell.get("cell_id", i + 1)
        ctype = cell.get("cell_type", "code")
        title = cell.get("title", f"Cell {cid}")

        console.print(f"\n  [bold]Cell {cid}/{len(cells)}: {title}[/bold]")

        if ctype == "markdown":
            src = _gen_md(orchestrator, cell, analysis, paper_md, manifest)
        else:
            src = _gen_code(
                orchestrator, cell, planning_result,
                cell_analyses.get(cid, ""), manifest, paper_md,
            )
            # Update state manifest with generated code
            manifest.update_from_code(cid, src)

        generated.append({
            "cell_id": cid,
            "cell_type": ctype,
            "title": title,
            "source": src,
        })
        ext = "py" if ctype == "code" else "md"
        save_artifact(src, f"{output_dir}/cells/cell_{cid}_{ctype}.{ext}")

    return generated


# ── Markdown Generation ───────────────────────────────────────────────────────

def _gen_md(orch, cell, analysis, paper_md, manifest: StateManifest):
    """Generate a rich, educational markdown cell."""
    title = cell.get("title", "")
    desc = cell.get("description", "")
    cid = cell.get("cell_id", 1)
    paper_title = analysis.get("title", "Unknown Paper")
    methodology = str(analysis.get("methodology", ""))[:2000]
    problem = str(analysis.get("problem", ""))[:800]
    key_components = analysis.get("key_components", [])
    equations = analysis.get("key_equations", [])
    datasets = analysis.get("datasets", [])
    metrics = analysis.get("metrics", [])
    baselines = analysis.get("baselines", [])
    dependencies = analysis.get("dependencies", [])

    # Build rich, cell-type-specific instructions
    if cid == 1 or "title" in title.lower():
        specific = (
            f"Write the TITLE AND INTRODUCTION cell for this notebook.\n\n"
            f"Start with: # {paper_title}\n\n"
            f"Then include ALL of the following (8-15 lines total):\n"
            f"- A 3-4 sentence abstract summarizing the paper's contribution\n"
            f"- The problem being solved: {problem[:400]}\n"
            f"- Key innovation in 2-3 sentences\n"
            f"- What this notebook covers (brief outline)\n"
            f"- Key components: {', '.join(key_components[:6]) if key_components else 'see methodology'}\n"
            f"- Dependencies used: {', '.join(dependencies[:6]) if dependencies else 'PyTorch, NumPy'}\n"
            f"- Placeholder link: [📄 Paper Link](URL)\n\n"
            f"This MUST start with '# {paper_title}' — NOT 'Title and Description'"
        )
    elif any(k in title.lower() for k in ["architecture", "model", "method", "approach"]):
        specific = (
            f"Write a DETAILED architecture/method explanation cell: {title}\n\n"
            f"Start with: ## {title}\n\n"
            f"Include ALL of the following (10-20 lines):\n"
            f"- High-level description of the architecture/method\n"
            f"- Methodology: {methodology[:1000]}\n"
            f"- Key components and their roles: {', '.join(key_components[:5]) if key_components else 'see paper'}\n"
            f"- Relevant equations in LaTeX: {'; '.join(equations[:4]) if equations else 'derive from methodology'}\n"
            f"- Data flow: describe input → processing → output\n"
            f"- Key dimensions/sizes if mentioned in the paper\n"
            f"- Design choices and why they matter\n"
            f"- How this connects to what the next code cell implements"
        )
    elif any(k in title.lower() for k in ["data", "dataset"]):
        specific = (
            f"Write a data/dataset explanation cell: {title}\n\n"
            f"Start with: ## {title}\n\n"
            f"Include (8-15 lines):\n"
            f"- What dataset(s) the paper uses: {json.dumps(datasets[:3], default=str) if datasets else 'described below'}\n"
            f"- Data format, sizes, and preprocessing steps\n"
            f"- If synthetic data: explain what it represents and why\n"
            f"- How the data connects to the model (input shapes, vocab, etc.)"
        )
    elif any(k in title.lower() for k in ["train", "optim", "loss"]):
        specific = (
            f"Write a training explanation cell: {title}\n\n"
            f"Start with: ## {title}\n\n"
            f"Include (8-15 lines):\n"
            f"- Training procedure from the paper\n"
            f"- Optimizer, learning rate schedule, loss function\n"
            f"- Key hyperparameters and their paper-specified values\n"
            f"- Training tricks mentioned (warmup, dropout, label smoothing, etc.)\n"
            f"- Expected convergence behavior"
        )
    elif any(k in title.lower() for k in ["result", "summary", "conclusion", "eval"]):
        specific = (
            f"Write a results/evaluation summary cell: {title}\n\n"
            f"Start with: ## {title}\n\n"
            f"Include (8-15 lines):\n"
            f"- Metrics used: {', '.join(metrics[:5]) if metrics else 'see paper'}\n"
            f"- Baselines compared against: {', '.join(baselines[:5]) if baselines else 'N/A'}\n"
            f"- Key results and findings from the paper\n"
            f"- What this notebook demonstrated\n"
            f"- Limitations and potential improvements\n"
            f"- References to specific tables/figures from the paper"
        )
    else:
        specific = (
            f"Write a detailed explanation cell: {title}\n\n"
            f"Start with: ## {title}\n\n"
            f"Description: {desc}\n"
            f"Write 8-15 lines explaining this section:\n"
            f"- What concept/component this covers\n"
            f"- How it relates to the overall architecture\n"
            f"- Key details from the paper\n"
            f"- What the following code cell implements"
        )

    ctx = (
        f"{specific}\n\n"
        f"## Paper Context\n"
        f"Title: {paper_title}\n"
        f"Problem: {problem[:400]}\n"
        f"Method: {methodology[:800]}\n"
        f"Equations: {'; '.join(equations[:5]) if equations else 'N/A'}\n\n"
        f"Paper excerpt:\n{paper_md[:2000]}"
    )

    result = orch.call_nim(
        system=MD_SYSTEM, context=ctx,
        name=f"MDWriter-{cid}",
        model=orch.sub_agent_model, max_tokens=1500, temperature=0.3,
    )

    content = strip_think(result["content"])

    # Fallback: if model returned empty or too short
    if not content.strip() or len(content.strip()) < 30:
        if cid == 1:
            content = (
                f"# {paper_title}\n\n"
                f"{problem[:300] if problem else desc}\n\n"
                f"**Key components**: {', '.join(key_components[:5]) if key_components else 'See methodology below'}\n\n"
                f"This notebook provides a step-by-step implementation of the paper's core contributions."
            )
        else:
            content = f"## {title}\n\n{desc}\n\n{methodology[:300] if methodology else ''}"

    return content


# ── Code Generation ───────────────────────────────────────────────────────────

def _gen_code(orch, cell, planning, analysis_text, manifest: StateManifest, paper_md):
    """Generate a code cell with full state awareness."""
    cid = cell.get("cell_id", 0)
    deps = cell.get("dependencies", [])
    config = planning["config"]
    repo_info = planning.get("repo_info", {})
    analysis = planning.get("analysis", {})

    parts = [
        f"## Cell to Write\n"
        f"ID: {cid}\n"
        f"Title: {cell.get('title')}\n"
        f"Description: {cell.get('description')}\n"
        f"Complexity: {cell.get('estimated_complexity', 'medium')}",
    ]

    # STATE MANIFEST — the key improvement
    if manifest.imports or manifest.variables or manifest.classes:
        parts.append(manifest.format_for_prompt(max_chars=3000))

    # Detailed analysis for this cell
    if analysis_text:
        parts.append(f"\n## Detailed Analysis\n{analysis_text}")

    # Config/hyperparameters
    parts.append(f"\n## Config\n```json\n{json.dumps(config, indent=2, default=str)[:2000]}\n```")

    # Previous cells' code (dependency cells first, then recent)
    if manifest.all_code:
        parts.append("\n## Previous Cell Code (for reference — reuse these variables)")
        # Show dependency cells
        for d in deps:
            if d in manifest.all_code:
                parts.append(f"### Cell {d} (dependency)\n```python\n{manifest.all_code[d][:2000]}\n```")
        # Show most recent cells not in deps
        recent = sorted(manifest.all_code.keys())[-2:]
        for r in recent:
            if r not in deps and r in manifest.all_code:
                parts.append(f"### Cell {r} (recent)\n```python\n{manifest.all_code[r][:1500]}\n```")

    # Repo code context — match by cell title keywords
    repo_files = repo_info.get("repo_files", {})
    if repo_files:
        title_l = cell.get("title", "").lower()
        matched_files = _match_repo_files(title_l, repo_files)
        if matched_files:
            parts.append("\n## Reference Implementation (from existing repo)")
            for fn, code in matched_files[:2]:
                parts.append(f"### {fn}\n```python\n{code[:3000]}\n```")

    # Paper methodology for model/training cells
    title_l = cell.get("title", "").lower()
    if any(k in title_l for k in ["model", "architecture", "network", "train",
                                    "loss", "attention", "encoder", "decoder"]):
        parts.append(f"\n## Paper Methodology\n{paper_md[:4000]}")

    result = orch.call_nim(
        system=CODE_SYSTEM, context="\n\n".join(parts),
        name=f"CodeWriter-{cid}",
        model=orch.orchestrator_model, max_tokens=4096, temperature=0.2,
    )
    return extract_code(result["content"])


def _match_repo_files(title_lower: str, repo_files: dict) -> list[tuple[str, str]]:
    """Match repo files to a cell based on title keywords."""
    keyword_map = {
        "model": ["model", "models", "network", "net", "architecture", "module"],
        "train": ["train", "trainer", "run", "main"],
        "data": ["data", "dataset", "dataloader"],
        "config": ["config", "configs", "hparams"],
        "loss": ["loss", "losses", "criterion"],
        "eval": ["eval", "evaluate", "metrics", "test"],
        "attention": ["attention", "transformer", "encoder", "decoder", "layer"],
        "util": ["utils", "helper", "helpers"],
    }

    relevant_keywords = set()
    for key, aliases in keyword_map.items():
        if key in title_lower:
            relevant_keywords.update(aliases)

    if not relevant_keywords:
        return []

    matches = []
    for fn, code in repo_files.items():
        if fn.startswith("_"):
            continue
        fn_lower = fn.lower().replace(".py", "").replace("/", " ")
        if any(kw in fn_lower for kw in relevant_keywords):
            matches.append((fn, code))

    return matches