"""
Shared utilities for Paper2Notebook pipeline.

All LLM calls go through NVIDIA NIM endpoints.
Cost tracking, JSON/code extraction, artifact I/O, logging.
"""

import json
import os
import re
import time
import yaml
from typing import Any, Optional

from openai import OpenAI
from rich.console import Console
from rich.panel import Panel

console = Console()


# ── Configuration ─────────────────────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def get_nim_client(api_key: str = "", base_url: str = "") -> OpenAI:
    """Create NVIDIA NIM client (OpenAI-compatible)."""
    api_key = api_key or os.getenv("NVIDIA_API_KEY", "")
    base_url = base_url or os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


# ── Arize Phoenix Observability ────────────────────────────────────────────────

_PHOENIX_TRACER = None

def init_phoenix(endpoint: str = "http://localhost:6006/v1/traces",
                 project: str = "paper2notebook") -> bool:
    """
    Initialize Arize Phoenix tracing for observability.

    Requires: pip install arize-phoenix openinference-instrumentation-openai arize-phoenix-otel
    Launch Phoenix server first: python -m phoenix.server.main serve

    All NVIDIA NIM calls will be auto-traced in the Phoenix UI.
    """
    global _PHOENIX_TRACER
    try:
        from phoenix.otel import register
        from openinference.instrumentation.openai import OpenAIInstrumentor

        tracer_provider = register(
            project_name=project,
            endpoint=endpoint,
            auto_instrument=True,
        )
        # Auto-instrument OpenAI calls (which NVIDIA NIM uses)
        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

        _PHOENIX_TRACER = tracer_provider
        console.print(f"  [green]✓[/green] Arize Phoenix tracing → {endpoint} (project: {project})")
        return True
    except ImportError:
        console.print("  [yellow]⚠ Phoenix not installed. Run: pip install arize-phoenix "
                      "openinference-instrumentation-openai arize-phoenix-otel[/yellow]")
        return False
    except Exception as e:
        console.print(f"  [yellow]⚠ Phoenix init failed: {e}[/yellow]")
        return False


# ── Think Block Stripping ─────────────────────────────────────────────────────

def strip_think(text: str) -> str:
    """
    Aggressively strip ALL <think>...</think> reasoning blocks from LLM output.

    Handles:
      - Normal: <think>reasoning</think>actual output
      - Nested: <think>...<think>...</think>...</think>actual output
      - Unclosed: <think>reasoning that never closes
      - Multiple: <think>a</think>text<think>b</think>more text
      - With newlines and whitespace variations
    """
    if not text or "<think>" not in text.lower():
        return text

    import re

    # 1. Remove all <think>...</think> blocks (greedy, handles nested)
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # 2. Handle unclosed <think> — remove everything from <think> to end
    cleaned = re.sub(r'<think>.*$', '', cleaned, flags=re.DOTALL | re.IGNORECASE)

    # 3. Handle orphan </think> (closing tag without opener)
    cleaned = re.sub(r'^.*?</think>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)

    return cleaned.strip()


# ── LLM Call via NVIDIA NIM ───────────────────────────────────────────────────

def llm_call(
    client: OpenAI,
    messages: list[dict],
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    response_format: dict | None = None,
) -> dict:
    """Unified LLM call via NVIDIA NIM. Returns {content, usage, model, elapsed}."""
    t0 = time.time()
    try:
        kwargs = dict(
            model=model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )
        if response_format:
            kwargs["response_format"] = response_format

        completion = client.chat.completions.create(**kwargs)
        content = strip_think(completion.choices[0].message.content or "")
        usage = {
            "prompt_tokens": completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens,
            "total_tokens": completion.usage.total_tokens,
        }
    except Exception as e:
        content = f"[LLM Error: {e}]"
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    return {"content": content, "usage": usage, "model": model, "elapsed": time.time() - t0}


def llm_call_structured(
    client: OpenAI,
    messages: list[dict],
    model: str,
    json_schema: dict,
    temperature: float = 0.2,
    max_tokens: int = 4096,
) -> dict:
    """
    LLM call with NVIDIA NIM guided_json for structured output.

    Tries in order:
      1. extra_body={"guided_json": schema}  (NIM constrained decoding)
      2. extra_body={"nvext": {"guided_json": schema}}  (alternative NIM path)
      3. response_format={"type": "json_object"}  (basic JSON mode fallback)

    Returns {content, usage, model, elapsed} — same shape as llm_call.
    """
    t0 = time.time()

    # Strategy 1: guided_json (top-level extra_body — works with newer NIM)
    try:
        completion = client.chat.completions.create(
            model=model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
            extra_body={"guided_json": json_schema},
        )
        content = strip_think(completion.choices[0].message.content or "")
        usage = {
            "prompt_tokens": completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens,
            "total_tokens": completion.usage.total_tokens,
        }
        return {"content": content, "usage": usage, "model": model, "elapsed": time.time() - t0}
    except Exception as e1:
        console.print(f"    [dim]guided_json not supported: {str(e1)[:80]}[/dim]")

    # Strategy 2: nvext guided_json (older NIM container format)
    try:
        completion = client.chat.completions.create(
            model=model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
            extra_body={"nvext": {"guided_json": json_schema}},
        )
        content = strip_think(completion.choices[0].message.content or "")
        usage = {
            "prompt_tokens": completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens,
            "total_tokens": completion.usage.total_tokens,
        }
        return {"content": content, "usage": usage, "model": model, "elapsed": time.time() - t0}
    except Exception as e2:
        console.print(f"    [dim]nvext guided_json not supported: {str(e2)[:80]}[/dim]")

    # Strategy 3: json_object mode (universally supported)
    return llm_call(
        client, messages, model,
        temperature=temperature, max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )


# ── JSON / Code Extraction ────────────────────────────────────────────────────

def extract_json(text: str) -> Optional[dict]:
    """Extract JSON from LLM response (handles fences, wrappers, thinking)."""
    text = strip_think(text)
    for pat, flags in [
        (r'\[CONTENT\](.*?)\[/CONTENT\]', re.DOTALL),
        (r'```json\s*\n(.*?)\n```', re.DOTALL),
        (r'```\s*\n(.*?)\n```', re.DOTALL),
    ]:
        m = re.search(pat, text, flags)
        if m:
            text = m.group(1).strip()
            break
    text = re.sub(r',\s*([}\]])', r'\1', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for p in [r'\{[\s\S]*\}', r'\[[\s\S]*\]']:
            m = re.search(p, text)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    continue
    return None


def extract_code(content: str) -> str:
    """Extract code from markdown code blocks."""
    content = strip_think(content)
    m = re.search(r'```(?:python)?\s*\n(.*?)```', content, re.DOTALL)
    return m.group(1).strip() if m else content.strip()


# ── Cost Tracking (NVIDIA NIM pricing) ────────────────────────────────────────

NIM_PRICING = {
    "nvidia/llama-3.3-nemotron-super-49b-v1.5": {"input": 0.80, "output": 2.40},
    "nvidia/nvidia-nemotron-nano-9b-v2": {"input": 0.10, "output": 0.30},
    "nvidia/nemotron-nano-12b-v2-vl": {"input": 0.15, "output": 0.45},
    "default": {"input": 1.00, "output": 3.00},
}


class CostTracker:
    def __init__(self):
        self.total_cost = 0.0
        self.stage_costs: dict[str, float] = {}
        self.call_count = 0

    def record(self, stage: str, usage: dict, model: str):
        p = NIM_PRICING.get(model, NIM_PRICING["default"])
        cost = (usage.get("prompt_tokens", 0) / 1e6) * p["input"] + \
               (usage.get("completion_tokens", 0) / 1e6) * p["output"]
        self.total_cost += cost
        self.stage_costs[stage] = self.stage_costs.get(stage, 0.0) + cost
        self.call_count += 1

    def summary(self) -> str:
        lines = [f"Total: ${self.total_cost:.6f}  ({self.call_count} NIM calls)"]
        for s, c in self.stage_costs.items():
            lines.append(f"  {s}: ${c:.6f}")
        return "\n".join(lines)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"total_cost": self.total_cost, "stage_costs": self.stage_costs,
                        "call_count": self.call_count}, f, indent=2)


# ── Helpers ───────────────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)

def save_artifact(content: Any, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(content, (dict, list)):
            json.dump(content, f, indent=2, ensure_ascii=False)
        else:
            f.write(str(content))

def load_artifact(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        t = f.read()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return t

def log_stage(stage: str, msg: str = ""):
    console.print(Panel(
        f"[bold]{stage}[/bold]\n{msg}" if msg else f"[bold]{stage}[/bold]",
        border_style="cyan"))

def log_sub(name: str, detail: str = ""):
    console.print(f"  [green]▸[/green] {name}" + (f" — {detail}" if detail else ""))

def log_cost(result: dict, stage: str, tracker: CostTracker):
    tracker.record(stage, result["usage"], result["model"])
    t = result["usage"]["total_tokens"]
    p = NIM_PRICING.get(result["model"], NIM_PRICING["default"])
    cost = (result["usage"].get("prompt_tokens", 0) / 1e6) * p["input"] + \
           (result["usage"].get("completion_tokens", 0) / 1e6) * p["output"]
    console.print(f"    [dim]({t:,} tok, {result['elapsed']:.1f}s, ${cost:.6f})[/dim]")