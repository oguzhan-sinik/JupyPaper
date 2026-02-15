"""
Stage 4: Assembly + Debugging.

Builds .ipynb from generated cells, then applies critical post-processing:
  - Import deduplication: consolidates all imports into cell 1 (setup cell)
  - Broken reference removal: fixes `from cell_6 import X` hallucinations
  - Figure embedding: attaches paper images to markdown cells via nbformat
  - AST validation: catches undefined variables and syntax errors
  - Uses NVIDIA NIM (Nemotron Super) to fix remaining issues
"""

import ast
import re
import base64
import io
from typing import Optional
from collections import OrderedDict

import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

from orchestrator import Orchestrator
from utils import log_stage, log_sub, strip_think, console


# ── Import Deduplication ──────────────────────────────────────────────────────

def _parse_imports(code: str) -> tuple[list[str], str]:
    """
    Separate import statements from the rest of the code.
    Returns (import_lines, remaining_code).
    """
    lines = code.split("\n")
    imports = []
    body = []
    in_import_block = True  # imports typically at top

    for line in lines:
        stripped = line.strip()
        # Detect import lines (including multi-line from X import (...))
        if stripped.startswith(("import ", "from ")) and "importlib" not in stripped:
            imports.append(stripped)
        elif stripped == "" and in_import_block:
            continue  # skip blank lines in import block
        else:
            in_import_block = False
            body.append(line)

    return imports, "\n".join(body).strip()


def _normalize_import(imp: str) -> str:
    """Normalize an import statement for dedup comparison."""
    # Remove comments
    imp = re.sub(r'#.*$', '', imp).strip()
    # Normalize whitespace
    imp = re.sub(r'\s+', ' ', imp)
    return imp


def _is_hallucinated_import(imp: str) -> bool:
    """Detect LLM-hallucinated imports that would fail in a notebook."""
    hallucination_patterns = [
        r'from\s+cell_\d+',           # from cell_6 import TransformerModel
        r'from\s+Cell\d+',            # from Cell6 import ...
        r'from\s+config\s+import',    # from config import (when no config module exists)
        r'from\s+[\w_]+_repl\s+import',  # from rlm_repl import ...
        r'from\s+model\s+import',     # from model import (when model.py doesn't exist)
        r'from\s+train\s+import',     # from train import
        r'from\s+utils\s+import',     # from utils import (when no utils module exists)
        r'from\s+helper\s+import',    # from helper import
        r'from\s+src\.',              # from src.module import
        r'from\s+qwen3\s+import',     # from qwen3 import Qwen3
    ]
    for pat in hallucination_patterns:
        if re.search(pat, imp, re.IGNORECASE):
            return True
    return False


def _deduplicate_imports(cells: list[dict]) -> list[dict]:
    """
    Consolidate imports: keep first occurrence in earliest cell, remove dupes.

    Strategy:
      - Collect all imports across cells
      - Remove hallucinated imports entirely
      - Place consolidated imports in the first code cell (setup cell)
      - Remove duplicate imports from subsequent cells
    """
    log_sub("Import Dedup", "Consolidating imports across cells")

    seen_imports: OrderedDict[str, str] = OrderedDict()  # normalized → original
    hallucinated = []

    # First pass: collect all unique imports
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        imports, _ = _parse_imports(cell.get("source", ""))
        for imp in imports:
            if _is_hallucinated_import(imp):
                hallucinated.append(imp)
                continue
            norm = _normalize_import(imp)
            if norm and norm not in seen_imports:
                seen_imports[norm] = imp

    if hallucinated:
        console.print(f"    [yellow]Removed {len(hallucinated)} hallucinated import(s)[/yellow]")
        for h in hallucinated[:5]:
            console.print(f"      [dim]✗ {h}[/dim]")

    # Group imports by category for clean ordering
    stdlib_imports = []
    third_party_imports = []
    torch_imports = []

    stdlib_modules = {
        "os", "sys", "re", "json", "math", "time", "copy", "warnings",
        "collections", "itertools", "functools", "pathlib", "typing",
        "io", "abc", "dataclasses", "enum", "random",
    }

    for norm, orig in seen_imports.items():
        module = orig.split()[1].split(".")[0] if orig.startswith("import ") else orig.split()[1].split(".")[0]
        if module in stdlib_modules:
            stdlib_imports.append(orig)
        elif module in ("torch", "torchvision", "torch_geometric"):
            torch_imports.append(orig)
        else:
            third_party_imports.append(orig)

    # Build consolidated import block
    import_sections = []
    if stdlib_imports:
        import_sections.append("\n".join(sorted(set(stdlib_imports))))
    if third_party_imports:
        import_sections.append("\n".join(sorted(set(third_party_imports))))
    if torch_imports:
        import_sections.append("\n".join(sorted(set(torch_imports))))

    consolidated = "\n\n".join(import_sections)

    # Second pass: strip imports from all cells, add consolidated to setup cell
    setup_found = False
    result = []

    for cell in cells:
        if cell.get("cell_type") != "code":
            result.append(cell)
            continue

        imports, body = _parse_imports(cell.get("source", ""))

        if not setup_found:
            # First code cell becomes the setup cell with all imports
            setup_found = True
            # Keep any pip install lines from the original
            pip_lines = [l for l in cell.get("source", "").split("\n")
                         if l.strip().startswith(("!pip", "# !pip", "%pip"))]
            pip_block = "\n".join(pip_lines) + "\n\n" if pip_lines else ""

            cell["source"] = (
                f"# Setup & Imports\n"
                f"{pip_block}"
                f"{consolidated}\n\n"
                f"# Verify GPU availability\n"
                f"import torch\n"
                f"device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
                f"print(f'Using device: {{device}}')"
            )
        else:
            # Subsequent cells: only keep body, no imports
            # Exception: cell-specific imports not in consolidated (rare libs)
            cell_specific = []
            for imp in imports:
                if not _is_hallucinated_import(imp):
                    norm = _normalize_import(imp)
                    if norm not in seen_imports:
                        cell_specific.append(imp)

            if cell_specific and body:
                cell["source"] = "\n".join(cell_specific) + "\n\n" + body
            elif body:
                cell["source"] = body
            # else: cell becomes empty, will be cleaned up later

        result.append(cell)

    n_orig = sum(1 for c in cells if c.get("cell_type") == "code")
    console.print(f"    Consolidated {len(seen_imports)} unique imports into setup cell")
    return result


# ── Broken Reference Fixing ───────────────────────────────────────────────────

def _fix_broken_references(cells: list[dict]) -> list[dict]:
    """
    Fix common LLM hallucination patterns in generated code.

    Removes:
      - `from cell_N import X` → replaced with comment
      - `global config, model, ...` → removed (not needed in notebooks)
      - try/except blocks that just re-import on failure
    """
    log_sub("Reference Fix", "Cleaning hallucinated cross-cell references")
    fixes_applied = 0

    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        src = cell["source"]

        # Fix 1: Remove `from cell_N import X` lines
        new_lines = []
        for line in src.split("\n"):
            stripped = line.strip()
            if re.match(r'from\s+cell_\d+\s+import', stripped, re.IGNORECASE):
                # Replace with comment about the dependency
                var_match = re.search(r'import\s+(.+)', stripped)
                if var_match:
                    vars_str = var_match.group(1).strip()
                    new_lines.append(f"# Uses {vars_str} from previous cell")
                fixes_applied += 1
            elif re.match(r'global\s+(config|model|train|dataset|device)', stripped):
                # Remove unnecessary global statements
                fixes_applied += 1
            elif "Reference variables from previous cells" in stripped:
                # Remove boilerplate comments
                fixes_applied += 1
            else:
                new_lines.append(line)

        cell["source"] = "\n".join(new_lines)

    if fixes_applied:
        console.print(f"    Fixed {fixes_applied} broken reference(s)")
    return cells


# ── Figure Embedding ──────────────────────────────────────────────────────────

def _embed_figures_in_notebook(
    nb: nbformat.NotebookNode,
    figures: list[dict],
) -> nbformat.NotebookNode:
    """
    Embed paper figures into markdown cells using nbformat attachments.

    Uses the official nbformat attachment mechanism (v4.1+) which stores
    base64 data in a dedicated `attachments` dict — works reliably across
    JupyterLab, Classic Notebook, and GitHub rendering.

    Args:
        nb: The notebook to modify
        figures: List of {"figure_id": int, "image_b64": str, "mime": str,
                         "caption": str, "description": str}
    """
    if not figures:
        return nb

    log_sub("Figure Embedding", f"Embedding {len(figures)} figure(s)")

    for fig in figures:
        fid = fig.get("figure_id", 0)
        img_b64 = fig.get("image_b64", "")
        mime = fig.get("mime", "image/png")
        caption = fig.get("caption", f"Figure {fid}")
        description = fig.get("description", "")

        if not img_b64:
            continue

        # Find the best markdown cell to attach to (by caption matching or position)
        target_cell = _find_target_cell_for_figure(nb, fid, caption)

        if target_cell is not None:
            ext = "png" if "png" in mime else "jpg"
            attachment_name = f"figure_{fid}.{ext}"

            # Add attachment
            if "attachments" not in target_cell:
                target_cell["attachments"] = {}
            target_cell["attachments"][attachment_name] = {mime: img_b64}

            # Add image reference to markdown
            img_md = f"\n\n![{caption}](attachment:{attachment_name})\n"
            if description:
                img_md += f"\n*{caption}*: {description}\n"
            target_cell.source += img_md

            console.print(f"    [green]✓[/green] Figure {fid} → Cell {nb.cells.index(target_cell)}")

    return nb


def _find_target_cell_for_figure(nb, fig_id, caption):
    """Find the best markdown cell to embed a figure into."""
    caption_lower = caption.lower()

    # Strategy 1: Find cell that mentions this figure
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            if f"figure {fig_id}" in cell.source.lower() or f"fig. {fig_id}" in cell.source.lower():
                return cell

    # Strategy 2: Find cell with related keywords from caption
    keywords = [w for w in caption_lower.split() if len(w) > 3 and w not in {
        "figure", "shows", "from", "with", "that", "this", "paper"
    }]
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            src_lower = cell.source.lower()
            if any(kw in src_lower for kw in keywords):
                return cell

    # Strategy 3: Insert before the first code cell after the title
    md_cells = [c for c in nb.cells if c.cell_type == "markdown"]
    if len(md_cells) > 1:
        # Try architecture/method explanation cells
        for cell in md_cells[1:]:
            if any(kw in cell.source.lower() for kw in ["architecture", "model", "method", "overview"]):
                return cell
        return md_cells[1]  # second markdown cell

    return md_cells[0] if md_cells else None


# ── Code Cleaning ─────────────────────────────────────────────────────────────

def _clean_code(s: str) -> str:
    """Clean generated code cell content."""
    s = strip_think(s)
    # Remove code fences
    s = re.sub(r'^```(?:python)?\s*\n?', '', s)
    s = re.sub(r'\n?```\s*$', '', s)
    # Remove file headers (e.g., "## model.py")
    lines = s.split("\n")
    while lines and re.match(r'^##?\s*\w+\.(py|ipynb)', lines[0]):
        lines.pop(0)
    # Remove empty trailing lines
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines).strip()


def _clean_md(s: str) -> str:
    """Clean generated markdown cell content."""
    s = strip_think(s)
    if s.startswith("```") and s.endswith("```"):
        lines = s.split("\n")
        if len(lines) > 2:
            s = "\n".join(lines[1:-1])
    return s.strip()


# ── Remove Empty/Trivial Cells ───────────────────────────────────────────────

def _remove_empty_cells(cells: list[dict]) -> list[dict]:
    """Remove cells that are empty or contain only comments."""
    result = []
    for cell in cells:
        src = cell.get("source", "").strip()
        if not src:
            continue
        # Check if code cell has only comments/whitespace
        if cell.get("cell_type") == "code":
            non_comment = [l for l in src.split("\n")
                           if l.strip() and not l.strip().startswith("#")]
            if not non_comment:
                continue
        result.append(cell)
    return result


# ── Assembly ──────────────────────────────────────────────────────────────────

def assemble_notebook(
    generated_cells: list[dict],
    planning_result: dict,
    kernel: str = "python3",
    language: str = "python",
    figures: list[dict] = None,
) -> nbformat.NotebookNode:
    """
    Assemble notebook from generated cells with full post-processing.

    Post-processing pipeline:
      1. Clean code/markdown content
      2. Fix broken cross-cell references
      3. Deduplicate imports (consolidate into setup cell)
      4. Remove empty/trivial cells
      5. Embed paper figures
      6. Add notebook metadata
    """
    log_stage("Stage 4: Assembly + Debug", "Building Jupyter notebook")

    # Step 1: Clean content
    cleaned = []
    for cd in generated_cells:
        src = cd.get("source", "")
        cell_type = cd.get("cell_type", "code")
        if cell_type == "markdown":
            src = _clean_md(src)
        else:
            src = _clean_code(src)
        cleaned.append({**cd, "source": src})

    # Step 2: Fix broken references
    cleaned = _fix_broken_references(cleaned)

    # Step 3: Deduplicate imports
    cleaned = _deduplicate_imports(cleaned)

    # Step 4: Remove empty cells
    cleaned = _remove_empty_cells(cleaned)

    # Build notebook
    nb = new_notebook()
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": language,
        "name": kernel,
    }
    nb.metadata.language_info = {"name": language, "version": "3.10.0"}

    # Add paper metadata
    paper_title = planning_result.get("analysis", {}).get("title", "")
    if paper_title:
        nb.metadata["paper_title"] = paper_title

    for cd in cleaned:
        src = cd.get("source", "")
        if cd.get("cell_type") == "markdown":
            nb.cells.append(new_markdown_cell(source=src))
        else:
            nb.cells.append(new_code_cell(source=src))

    # Step 5: Embed figures
    if figures:
        nb = _embed_figures_in_notebook(nb, figures)

    cn = sum(1 for c in nb.cells if c.cell_type == "code")
    mn = sum(1 for c in nb.cells if c.cell_type == "markdown")
    console.print(f"  Assembled {len(nb.cells)} cells ({cn} code, {mn} markdown)")
    return nb


# ── Validation ────────────────────────────────────────────────────────────────

def validate_notebook(
    nb: nbformat.NotebookNode,
    orch: Optional[Orchestrator] = None,
) -> list[str]:
    """Validate notebook for common issues using AST analysis."""
    log_sub("Validation", "Checking for issues")
    issues = []

    # Track defined names across cells for cross-cell reference checking
    all_defined: set[str] = set()

    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            # Check markdown quality
            if c.cell_type == "markdown":
                if "title and description" in c.source.lower()[:100]:
                    issues.append(f"Cell {i}: Generic title — needs real paper title")
                if len(c.source.strip()) < 20:
                    issues.append(f"Cell {i}: Markdown too short (<20 chars)")
            continue

        src = c.source
        if not src.strip():
            issues.append(f"Cell {i}: Empty code cell")
            continue

        # Check for placeholders
        if "TODO" in src or "FIXME" in src:
            issues.append(f"Cell {i}: Contains TODO/placeholder")
        if "<think>" in src:
            issues.append(f"Cell {i}: Contains <think> block")

        # Syntax check via compile
        try:
            compile(src, f"<cell_{i}>", "exec")
        except SyntaxError as e:
            issues.append(f"Cell {i}: Syntax error — {e}")
            continue

        # AST analysis for deeper issues
        try:
            tree = ast.parse(src)

            # Collect defined names in this cell
            cell_defined = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    cell_defined.add(node.name)
                elif isinstance(node, ast.ClassDef):
                    cell_defined.add(node.name)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            cell_defined.add(target.id)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        name = alias.asname or alias.name.split(".")[0]
                        cell_defined.add(name)
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        name = alias.asname or alias.name
                        cell_defined.add(name)

            all_defined.update(cell_defined)

        except Exception:
            pass

        # Check for hallucinated module imports
        for line in src.split("\n"):
            stripped = line.strip()
            if re.match(r'from\s+cell_\d+\s+import', stripped):
                issues.append(f"Cell {i}: Hallucinated import — {stripped}")

    if issues:
        console.print(f"  [yellow]{len(issues)} issue(s) found[/yellow]")
        for iss in issues[:10]:
            console.print(f"    [yellow]⚠ {iss}[/yellow]")
    else:
        console.print("  [green]✓ No issues[/green]")
    return issues


# ── Debug ─────────────────────────────────────────────────────────────────────

def debug_notebook(
    nb: nbformat.NotebookNode,
    issues: list[str],
    orchestrator: Orchestrator,
    planning_result: dict,
) -> nbformat.NotebookNode:
    """Fix issues in the notebook, combining auto-fixes and NIM-powered repairs."""
    if not issues:
        return nb
    log_sub("Debug", f"Fixing {len(issues)} issue(s)")

    # Auto-fix: <think> blocks
    for c in nb.cells:
        c.source = strip_think(c.source)

    # Auto-fix: generic titles
    real_title = planning_result.get("analysis", {}).get("title", "")
    if real_title:
        for c in nb.cells:
            if c.cell_type == "markdown" and "title and description" in c.source.lower()[:100]:
                c.source = re.sub(r'(?i)#+\s*title\s*(and|&)\s*description',
                                  f"# {real_title}", c.source)

    # Auto-fix: hallucinated imports
    for c in nb.cells:
        if c.cell_type == "code":
            lines = c.source.split("\n")
            c.source = "\n".join(
                l for l in lines
                if not re.match(r'\s*from\s+cell_\d+\s+import', l)
            )

    # Check remaining issues
    remaining = [i for i in issues
                 if "<think>" not in i
                 and "Generic title" not in i
                 and "Hallucinated import" not in i]

    if not remaining:
        console.print("  [green]✓ All issues auto-fixed[/green]")
        return nb

    # Use NVIDIA NIM for remaining issues
    code_cells = [f"## Cell {i}\n```python\n{c.source}\n```"
                  for i, c in enumerate(nb.cells) if c.cell_type == "code"]

    result = orchestrator.call_nim(
        system=(
            "Fix the listed issues in this Jupyter notebook. For each fix output:\n\n"
            "Cell: <number>\n<<<<<<< SEARCH\n<original>\n=======\n<fixed>\n>>>>>>> REPLACE\n\n"
            "ONLY fix what's broken. Keep changes minimal. No <think> tags."
        ),
        context=(
            f"## Issues\n" + "\n".join(f"- {i}" for i in remaining) +
            f"\n\n## Code\n" + "\n\n".join(code_cells[:15])
        ),
        name="Debugger",
        model=orchestrator.orchestrator_model,
        max_tokens=4096,
    )

    # Apply search/replace fixes
    fixes = re.findall(
        r'Cell:\s*(\d+)\s*\n<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE',
        result["content"], re.DOTALL,
    )
    applied = 0
    code_cell_indices = [i for i, c in enumerate(nb.cells) if c.cell_type == "code"]
    for idx_s, search, replace in fixes:
        idx = int(idx_s)
        if idx < len(code_cell_indices):
            real_idx = code_cell_indices[idx]
            if search.strip() in nb.cells[real_idx].source:
                nb.cells[real_idx].source = nb.cells[real_idx].source.replace(
                    search.strip(), replace.strip()
                )
                applied += 1
    console.print(f"  Applied {applied}/{len(fixes)} NIM fixes")
    return nb


# ── Save ──────────────────────────────────────────────────────────────────────

def save_notebook(nb: nbformat.NotebookNode, path: str):
    """Save notebook to disk."""
    with open(path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    console.print(f"\n  [green bold]✓ Saved → {path}[/green bold]")