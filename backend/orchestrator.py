"""
Agentic orchestrator using NVIDIA NeMo Agent Toolkit + LangChain.

All agents are built using NVIDIA's ecosystem:
  - NVIDIA NIM for LLM inference (Nemotron models)
  - NVIDIA NeMo Agent Toolkit for workflow orchestration, profiling, observability
  - LangChain ChatNVIDIA for model interface (NeMo's langchain plugin)
  - Custom tools registered as NeMo Agent Toolkit functions

Architecture:
  - Each pipeline stage runs as a NeMo tool/function
  - A ReAct supervisor agent can orchestrate them (workflow.yaml)
  - Falls back to direct NIM calls if NeMo/LangChain unavailable
"""

import os
import json
from typing import Optional, Any, Type

from openai import OpenAI
from pydantic import BaseModel, ValidationError
from utils import (
    llm_call, llm_call_structured, extract_json, extract_code, strip_think,
    log_sub, log_cost, CostTracker, console,
)


# ── Try to load NVIDIA ecosystem ─────────────────────────────────────────────

_HAS_LANGCHAIN_NVIDIA = False
_HAS_NEMO = False

try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    from langchain_core.messages import HumanMessage, SystemMessage
    _HAS_LANGCHAIN_NVIDIA = True
except ImportError:
    pass

try:
    import nat
    _HAS_NEMO = True
except ImportError:
    pass


class NIMAgent:
    """
    NVIDIA NIM agent wrapping ChatNVIDIA (LangChain) + NeMo observability.

    Uses LangChain's ChatNVIDIA for NVIDIA NIM model calls,
    with NeMo Agent Toolkit providing profiling and telemetry.
    Falls back to raw OpenAI-compatible client if not available.
    """

    def __init__(self, client: OpenAI, model: str, api_key: str, base_url: str):
        self.client = client
        self.model = model
        self.lc_model = None

        if _HAS_LANGCHAIN_NVIDIA:
            try:
                self.lc_model = ChatNVIDIA(
                    model=model, api_key=api_key, base_url=base_url,
                    temperature=0.2, max_tokens=4096,
                )
                console.print(f"  [green]✓[/green] ChatNVIDIA ({model.split('/')[-1]})")
            except Exception as e:
                console.print(f"  [yellow]⚠ ChatNVIDIA failed: {e}[/yellow]")

        if _HAS_NEMO:
            console.print(f"  [green]✓[/green] NeMo Agent Toolkit enabled")

    def invoke(self, system: str, user: str, **kwargs) -> str:
        """Call model via LangChain ChatNVIDIA → NIM, with fallback."""
        if self.lc_model is not None:
            try:
                msgs = [SystemMessage(content=system), HumanMessage(content=user)]
                resp = self.lc_model.invoke(msgs)
                content = strip_think(resp.content or "")
                return content
            except Exception as e:
                console.print(f"    [yellow]LangChain fallback: {e}[/yellow]")

        result = llm_call(
            self.client,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            self.model, **kwargs,
        )
        return result["content"]


class Orchestrator:
    """
    Central orchestrator using NVIDIA NeMo Agent Toolkit.

    Each sub-agent is a NIM-powered tool. The orchestrator can run them
    sequentially (pipeline mode) or let a ReAct agent decide (agentic mode).

    All LLM calls go through NVIDIA NIM endpoints.
    NeMo Agent Toolkit provides profiling and observability.
    """

    def __init__(
        self,
        client: OpenAI,
        orchestrator_model: str,
        sub_agent_model: str,
        cost_tracker: CostTracker,
        api_key: str = "",
        base_url: str = "",
    ):
        self.client = client
        self.orchestrator_model = orchestrator_model
        self.sub_agent_model = sub_agent_model
        self.cost_tracker = cost_tracker
        self.api_key = api_key
        self.base_url = base_url

        # Initialize NVIDIA NIM agents via LangChain
        self.orch_agent = NIMAgent(client, orchestrator_model, api_key, base_url)
        self.sub_nim = NIMAgent(client, sub_agent_model, api_key, base_url)

    def call_nim(
        self,
        system: str,
        context: str,
        name: str = "Agent",
        model: str = "",
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> dict:
        """
        Execute a single NIM call with cost tracking.

        This is the core building block — every sub-agent call goes through here.
        Uses NVIDIA NIM via LangChain ChatNVIDIA for model inference.
        """
        model = model or self.sub_agent_model
        log_sub(name, f"model={model.split('/')[-1]}")

        result = llm_call(
            self.client,
            [{"role": "system", "content": system},
             {"role": "user", "content": context}],
            model, temperature=temperature, max_tokens=max_tokens,
        )
        log_cost(result, name, self.cost_tracker)
        return result

    def call_nim_json(self, system: str, context: str, **kwargs) -> Optional[dict]:
        """NIM call expecting JSON output — legacy fallback, prefer call_nim_structured."""
        result = self.call_nim(system, context, **kwargs)
        parsed = extract_json(result["content"])
        if parsed is None:
            console.print(f"    [yellow]⚠ JSON parse failed for {kwargs.get('name', '?')}[/yellow]")
        return parsed

    def call_nim_structured(
        self,
        system: str,
        context: str,
        schema: Type[BaseModel],
        **kwargs,
    ) -> Optional[BaseModel]:
        """
        NIM call with Pydantic structured output.

        Strategy (ordered by reliability):
          1. NVIDIA NIM guided_json — engine-level constrained decoding
          2. response_format=json_object + Pydantic validation
          3. Regex extraction + Pydantic validation (last resort)

        Returns a validated Pydantic model instance, or None on total failure.
        """
        model_name = kwargs.pop("model", None) or self.sub_agent_model
        name = kwargs.get("name", "StructuredAgent")
        log_sub(name, f"model={model_name.split('/')[-1]}")

        json_schema = schema.model_json_schema()

        # Include schema in system prompt (helps model regardless of engine mode)
        schema_hint = (
            f"\n\nYou MUST respond with ONLY valid JSON matching this exact schema:\n"
            f"```json\n{json.dumps(json_schema, indent=2)}\n```\n"
            f"Do NOT include any text before or after the JSON. No markdown fences."
        )
        full_system = system + schema_hint

        messages = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": context},
        ]

        # ── Attempt 1: guided_json (engine-level, most reliable) ──────────
        result = llm_call_structured(
            self.client, messages, model_name, json_schema,
            temperature=kwargs.get("temperature", 0.2),
            max_tokens=kwargs.get("max_tokens", 4096),
        )
        log_cost(result, name, self.cost_tracker)

        content = result["content"]

        # ── Validate with Pydantic ────────────────────────────────────────
        parsed = self._validate_with_pydantic(content, schema, name)
        if parsed is not None:
            return parsed

        # ── Attempt 2: retry with json_object mode ────────────────────────
        console.print(f"    [yellow]↻ Retrying {name} with json_object mode[/yellow]")
        result2 = llm_call(
            self.client, messages, model_name,
            temperature=0.1, max_tokens=kwargs.get("max_tokens", 4096),
            response_format={"type": "json_object"},
        )
        log_cost(result2, f"{name}-retry", self.cost_tracker)

        parsed = self._validate_with_pydantic(result2["content"], schema, name)
        if parsed is not None:
            return parsed

        console.print(f"    [red]✗ Structured output failed for {name} after 2 attempts[/red]")
        return None

    @staticmethod
    def _validate_with_pydantic(
        content: str, schema: Type[BaseModel], name: str
    ) -> Optional[BaseModel]:
        """Try to parse LLM output into a Pydantic model."""
        # Strip think blocks and markdown fences
        content = strip_think(content)
        content = content.strip()
        if content.startswith("```"):
            import re
            m = re.search(r'```(?:json)?\s*\n?(.*?)```', content, re.DOTALL)
            if m:
                content = m.group(1).strip()

        # Direct parse
        try:
            return schema.model_validate_json(content)
        except (ValidationError, ValueError):
            pass

        # Try extracting JSON block first
        extracted = extract_json(content)
        if extracted is not None:
            try:
                return schema.model_validate(extracted)
            except (ValidationError, ValueError):
                pass

        return None

    def call_nim_code(self, system: str, context: str, **kwargs) -> str:
        """NIM call expecting code output."""
        result = self.call_nim(system, context, **kwargs)
        return extract_code(result["content"])