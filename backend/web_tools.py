"""
GitHub repository discovery and code scraping.

SIMPLIFIED STRATEGY (fixes svoice/ossu bugs):
  1. If paper text contains an exact GitHub URL → use it
  2. If paper mentions an arxiv ID → check Papers With Code for that paper
  3. Papers With Code title search (exact match only)
  4. NO broad GitHub search — it returned completely unrelated repos

The old approach searched GitHub for the paper title and returned whatever
had the most stars. This matched:
  - ossu/computer-science (201K stars, a course curriculum)
  - facebookresearch/svoice (1.3K stars, a speech separation model)
Neither is related to "Attention Is All You Need".

Now we only use repos that are DIRECTLY linked to the paper.
"""

import os
import re
import json
import time
from typing import Optional

import requests
from utils import console, log_sub


# ── GitHub REST API ───────────────────────────────────────────────────────────

GITHUB_API = "https://api.github.com"
GITHUB_RAW = "https://raw.githubusercontent.com"

PRIORITY_FILES = [
    "model.py", "models.py", "main.py", "train.py", "trainer.py",
    "config.py", "configs.py", "run.py", "run_train.py",
    "network.py", "net.py", "architecture.py", "module.py", "modules.py",
    "layers.py", "attention.py", "transformer.py", "encoder.py", "decoder.py",
    "dataset.py", "data.py", "dataloader.py", "data_utils.py",
    "loss.py", "losses.py", "metrics.py", "evaluate.py", "eval.py",
    "utils.py", "helper.py", "helpers.py",
    "requirements.txt", "setup.py", "pyproject.toml",
]


def _github_headers() -> dict:
    headers = {"Accept": "application/vnd.github.v3+json", "User-Agent": "Paper2Notebook"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


# ── URL Extraction from Paper Text ────────────────────────────────────────────

def _extract_github_urls(text: str) -> list[str]:
    """
    Extract GitHub repository URLs from paper text.

    Looks for patterns like:
      - https://github.com/owner/repo
      - github.com/owner/repo
      - "code available at github.com/..."
      - footnotes with GitHub links
    """
    urls = []
    for m in re.finditer(
        r'(?:https?://)?github\.com/([\w.-]+)/([\w.-]+)',
        text, re.IGNORECASE
    ):
        owner, repo = m.group(1), m.group(2)
        # Skip GitHub meta-pages
        if owner.lower() in ("topics", "features", "about", "settings",
                              "marketplace", "explore", "trending", "collections"):
            continue
        # Skip file extensions that got caught
        if "." in repo and repo.rsplit(".", 1)[1] in ("pdf", "html", "md", "txt"):
            continue
        clean_url = f"https://github.com/{owner}/{repo}"
        if clean_url not in urls:
            urls.append(clean_url)
    return urls


# ── Papers With Code (exact match) ───────────────────────────────────────────

def _search_paperswithcode(title: str) -> Optional[str]:
    """
    Search Papers With Code for implementation repos.
    Uses exact title match to avoid returning wrong papers.
    """
    try:
        resp = requests.get(
            "https://paperswithcode.com/api/v1/papers/",
            params={"q": title[:100], "items_per_page": 5},
            timeout=10,
        )
        if not resp.ok:
            console.print(f"    [dim]Papers With Code HTTP {resp.status_code}[/dim]")
            return None

        content_type = resp.headers.get("content-type", "")
        if "json" not in content_type:
            console.print(f"    [dim]Papers With Code returned non-JSON[/dim]")
            return None

        papers = resp.json().get("results", [])
        for paper in papers:
            # Only accept if title is a close match
            pwc_title = (paper.get("title") or "").lower().strip()
            query_title = title.lower().strip()
            if not _titles_match(query_title, pwc_title):
                continue

            paper_id = paper.get("id")
            if not paper_id:
                continue

            repo_resp = requests.get(
                f"https://paperswithcode.com/api/v1/papers/{paper_id}/repositories/",
                timeout=10,
            )
            if not repo_resp.ok or "json" not in repo_resp.headers.get("content-type", ""):
                continue

            repos = repo_resp.json().get("results", [])
            for r in sorted(repos, key=lambda x: (
                x.get("is_official", False), x.get("stars", 0)
            ), reverse=True):
                url = r.get("url", "")
                if "github.com" in url:
                    console.print(f"    [green]✓[/green] Papers With Code: {url}")
                    return url.rstrip("/")

    except json.JSONDecodeError as e:
        console.print(f"    [dim]Papers With Code JSON error: {e}[/dim]")
    except Exception as e:
        console.print(f"    [dim]Papers With Code error: {e}[/dim]")
    return None


def _titles_match(a: str, b: str) -> bool:
    """Check if two paper titles are substantially the same."""
    a = re.sub(r'[^\w\s]', '', a).lower().split()
    b = re.sub(r'[^\w\s]', '', b).lower().split()
    if not a or not b:
        return False
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    longer_set = set(longer)
    matches = sum(1 for w in shorter if w in longer_set)
    return matches / len(shorter) >= 0.7


# ── Valyu Search (targeted, academic) ────────────────────────────────────────

def _valyu_search_paper_code(title: str) -> Optional[str]:
    """
    Search Valyu's academic sources for the paper, extract code links from content.
    Only used as a last resort before giving up.
    """
    api_key = os.getenv("VALYU_API_KEY")
    if not api_key:
        return None

    try:
        from valyu import Valyu
        valyu = Valyu(api_key)

        response = valyu.search(
            title,
            search_type="proprietary",
            max_num_results=3,
            included_sources=["valyu/valyu-arxiv"],
        )

        if response.results:
            for result in response.results:
                content = getattr(result, "content", "") or ""
                result_title = getattr(result, "title", "") or ""

                if not _titles_match(title, result_title):
                    continue

                urls = _extract_github_urls(content)
                if urls:
                    console.print(f"    [green]✓[/green] Valyu arxiv → {urls[0]}")
                    return urls[0]

    except ImportError:
        pass
    except Exception as e:
        console.print(f"    [dim]Valyu search error: {e}[/dim]")

    return None


# ── GitHub Repo Scraping ──────────────────────────────────────────────────────

def _get_default_branch(owner: str, repo: str) -> str:
    try:
        resp = requests.get(f"{GITHUB_API}/repos/{owner}/{repo}",
                            headers=_github_headers(), timeout=10)
        if resp.ok:
            return resp.json().get("default_branch", "main")
    except Exception:
        pass
    return "main"


def _get_repo_tree(owner: str, repo: str, branch: str = "main") -> list[dict]:
    try:
        resp = requests.get(
            f"{GITHUB_API}/repos/{owner}/{repo}/git/trees/{branch}",
            params={"recursive": "1"},
            headers=_github_headers(), timeout=15,
        )
        if resp.ok:
            return resp.json().get("tree", [])
        if branch == "main":
            return _get_repo_tree(owner, repo, "master")
    except Exception:
        pass
    return []


def _fetch_raw_file(owner: str, repo: str, branch: str, filepath: str) -> Optional[str]:
    url = f"{GITHUB_RAW}/{owner}/{repo}/{branch}/{filepath}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.ok and len(resp.text) > 20:
            return resp.text
    except Exception:
        pass
    return None


def _select_best_files(tree: list[dict], max_files: int = 8) -> list[str]:
    py_files = [f["path"] for f in tree
                if f["type"] == "blob" and f["path"].endswith(".py")
                and f.get("size", 0) < 200_000]

    selected = []

    for pf in PRIORITY_FILES:
        for fp in py_files:
            if fp.rsplit("/", 1)[-1] == pf and fp not in selected:
                selected.append(fp)
                if len(selected) >= max_files:
                    return selected

    for fp in py_files:
        if fp not in selected and any(d in fp for d in ["model", "train", "config", "net", "layer"]):
            selected.append(fp)
            if len(selected) >= max_files:
                return selected

    for fp in py_files:
        if fp not in selected and "/" not in fp:
            selected.append(fp)
            if len(selected) >= max_files:
                return selected

    return selected[:max_files]


def scrape_github_repo(repo_url: str) -> dict[str, str]:
    """Scrape a GitHub repository for key implementation files."""
    log_sub("Repo Scraping", f"Fetching code from {repo_url}")
    files: dict[str, str] = {}

    match = re.match(r'https?://github\.com/([\w.-]+)/([\w.-]+)', repo_url)
    if not match:
        console.print(f"    [yellow]⚠ Cannot parse repo URL: {repo_url}[/yellow]")
        return files

    owner, repo = match.group(1), match.group(2)
    branch = _get_default_branch(owner, repo)
    console.print(f"    Branch: {branch}")

    tree = _get_repo_tree(owner, repo, branch)
    if not tree:
        console.print("    [yellow]⚠ Could not fetch repo tree[/yellow]")
        return files

    has_python = any(f["path"].endswith(".py") and f["type"] == "blob" for f in tree)
    if not has_python:
        console.print("    [yellow]⚠ No Python files — skipping[/yellow]")
        return files

    targets = _select_best_files(tree, max_files=8)
    console.print(f"    Selected {len(targets)} target file(s)")

    for rv in ["README.md", "readme.md", "README.rst"]:
        if any(f["path"].lower() == rv.lower() and f["type"] == "blob" for f in tree):
            targets.insert(0, rv)
            break

    for filepath in targets:
        if len(files) >= 10:
            break
        content = _fetch_raw_file(owner, repo, branch, filepath)
        if content:
            files[filepath] = content[:8000]
            console.print(f"    [green]✓[/green] {filepath} ({len(content):,} chars)")

    console.print(f"    Retrieved {len(files)} file(s) total")
    return files


# ── Main Entry Point ──────────────────────────────────────────────────────────

def search_for_repo(paper_title: str, paper_text: str = "",
                    mentioned_url: str = None) -> dict:
    """
    Find a paper's code repository.

    SIMPLIFIED STRATEGY — only uses direct links, never broad search:
      0. Paper's LLM-extracted URL (code_repo_mentioned field)
      1. Regex-extracted GitHub URLs from full paper text
      2. Papers With Code (exact title match only)
      3. Valyu academic search (look for code links in arxiv content)
      4. Give up — generate from scratch (better than using wrong repo)

    Returns: {"repo_url": str|None, "repo_files": dict, "source": str|None}
    """
    log_sub("Repo Discovery", "Searching for existing implementation")

    # ── Strategy 0: Explicit URL from LLM analysis ────────────────────────
    if mentioned_url and "github.com" in mentioned_url:
        console.print(f"    Paper mentions repo: {mentioned_url}")
        repo_files = scrape_github_repo(mentioned_url)
        if repo_files:
            return {"repo_url": mentioned_url, "repo_files": repo_files,
                    "source": "paper_mentioned"}
        else:
            console.print(f"    [yellow]⚠ Mentioned URL had no Python files[/yellow]")

    # ── Strategy 1: Extract GitHub URLs from full paper text ──────────────
    if paper_text:
        urls = _extract_github_urls(paper_text)
        for url in urls:
            console.print(f"    Found URL in paper text: {url}")
            repo_files = scrape_github_repo(url)
            if repo_files:
                return {"repo_url": url, "repo_files": repo_files,
                        "source": "paper_text"}

    # ── Strategy 2: Papers With Code (exact title match) ──────────────────
    console.print("    Trying Papers With Code (exact title match)...")
    pwc_url = _search_paperswithcode(paper_title)
    if pwc_url:
        repo_files = scrape_github_repo(pwc_url)
        if repo_files:
            return {"repo_url": pwc_url, "repo_files": repo_files,
                    "source": "paperswithcode"}

    # ── Strategy 3: Valyu academic search ─────────────────────────────────
    console.print("    Trying Valyu academic search...")
    valyu_url = _valyu_search_paper_code(paper_title)
    if valyu_url:
        repo_files = scrape_github_repo(valyu_url)
        if repo_files:
            return {"repo_url": valyu_url, "repo_files": repo_files,
                    "source": "valyu_arxiv"}

    # ── No repo found ─────────────────────────────────────────────────────
    console.print("    [dim]No repo found — will generate from scratch[/dim]")
    return {"repo_url": None, "repo_files": {}, "source": None}