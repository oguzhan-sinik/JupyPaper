"""
RLM (Recursive Language Model) context for Paper2Notebook.

Stores the paper markdown as a queryable variable (not in the prompt).
Provides section parsing, targeted queries, and recursive sub-calls
for long sections. Also provides stage-aware context curation to
prevent context rot (AOrchestra pattern).
"""

import re
from typing import Optional

from openai import OpenAI
from utils import llm_call, console, CostTracker, log_cost


class RLMContext:
    """
    Manages the paper as a queryable context variable.

    Key features:
      - Parse markdown into named sections
      - Provide compact overviews (not full text)
      - Recursive chunk-and-aggregate for long sections
      - Stage-aware context curation
    """

    def __init__(
        self,
        full_text: str,
        client: OpenAI,
        model: str,
        cost_tracker: CostTracker,
        chunk_size: int = 8000,
        max_depth: int = 1,
    ):
        self.full_text = full_text
        self.client = client
        self.model = model
        self.cost_tracker = cost_tracker
        self.chunk_size = chunk_size
        self.max_depth = max_depth
        self.sections = self._parse_sections()

    def _parse_sections(self) -> dict[str, str]:
        sections: dict[str, str] = {}
        current_title = "preamble"
        current_lines: list[str] = []
        for line in self.full_text.split("\n"):
            hdr = re.match(r'^(#{1,3})\s+(.+)', line)
            if hdr:
                if current_lines:
                    sections[current_title] = "\n".join(current_lines).strip()
                current_title = hdr.group(2).strip().lower()
                current_lines = [line]
            else:
                current_lines.append(line)
        if current_lines:
            sections[current_title] = "\n".join(current_lines).strip()
        return sections

    def get_section_names(self) -> list[str]:
        return list(self.sections.keys())

    def get_section(self, name: str) -> Optional[str]:
        name_l = name.lower().strip()
        if name_l in self.sections:
            return self.sections[name_l]
        for key, content in self.sections.items():
            if name_l in key or key in name_l:
                return content
        return None

    def get_overview(self, max_chars: int = 4000) -> str:
        lines = [f"Paper sections ({len(self.sections)} total):"]
        chars = 0
        for name, content in self.sections.items():
            preview = content[:300].replace("\n", " ").strip()
            entry = f"\n## {name}\n{preview}…"
            if chars + len(entry) > max_chars:
                lines.append(f"\n[…{len(self.sections) - len(lines) + 1} more sections truncated]")
                break
            lines.append(entry)
            chars += len(entry)
        return "\n".join(lines)

    def get_full_text_truncated(self, max_chars: int = 32000) -> str:
        """Return the full paper text, truncated if needed."""
        if len(self.full_text) <= max_chars:
            return self.full_text
        half = max_chars // 2
        return (self.full_text[:half] +
                f"\n\n[…TRUNCATED {len(self.full_text) - max_chars} chars…]\n\n" +
                self.full_text[-half:])

    def query_section(self, section_name: str, question: str) -> str:
        """RLM sub-query: ask a question about a specific section."""
        content = self.get_section(section_name)
        if content is None:
            return f"[Section '{section_name}' not found]"
        if len(content) < self.chunk_size * 4:
            return self._direct_query(content, question)
        return self._recursive_query(content, question)

    def _direct_query(self, content: str, question: str) -> str:
        result = llm_call(
            self.client,
            [{"role": "system", "content":
              "Answer the question based ONLY on the provided paper content. "
              "Be specific — include exact values, equations, and details."},
             {"role": "user", "content":
              f"## Paper Content\n{content}\n\n## Question\n{question}"}],
            self.model, temperature=0.1, max_tokens=2048,
        )
        log_cost(result, "RLM-query", self.cost_tracker)
        return result["content"]

    def _recursive_query(self, content: str, question: str) -> str:
        """Partition, query each chunk, aggregate (depth=1 recursion)."""
        words = content.split()
        chunks = [" ".join(words[i:i+self.chunk_size])
                  for i in range(0, len(words), self.chunk_size)]
        console.print(f"    [dim]RLM: partitioning into {len(chunks)} chunks[/dim]")

        partials = []
        for idx, chunk in enumerate(chunks):
            r = llm_call(
                self.client,
                [{"role": "system", "content":
                  "Extract information relevant to the question from this chunk. "
                  "If nothing relevant, respond with NO_RELEVANT_INFO."},
                 {"role": "user", "content":
                  f"## Chunk {idx+1}/{len(chunks)}\n{chunk}\n\n## Question\n{question}"}],
                self.model, temperature=0.1, max_tokens=1024,
            )
            log_cost(r, f"RLM-chunk-{idx+1}", self.cost_tracker)
            if "NO_RELEVANT_INFO" not in r["content"]:
                partials.append(r["content"])

        if not partials:
            return "[No relevant information found]"

        agg = llm_call(
            self.client,
            [{"role": "system", "content": "Synthesize these partial answers into one comprehensive response."},
             {"role": "user", "content":
              f"## Partial Answers\n{'---'.join(partials)}\n\n## Question\n{question}"}],
            self.model, temperature=0.1, max_tokens=2048,
        )
        log_cost(agg, "RLM-aggregate", self.cost_tracker)
        return agg["content"]

    def get_context_for_stage(self, stage: str, max_tokens: int = 6000) -> str:
        """AOrchestra-style context curation: stage-relevant slices only."""
        mc = max_tokens * 4
        if stage == "planning":
            parts = [self.get_overview(max_chars=mc)]
            for key in ["abstract", "introduction", "method", "methodology",
                        "approach", "proposed method"]:
                s = self.get_section(key)
                if s:
                    parts.append(f"\n## {key}\n{s[:2000]}")
            return "\n".join(parts)[:mc]
        elif stage == "analysis":
            parts = []
            for key in ["method", "methodology", "approach", "proposed method",
                        "experiments", "experimental setup", "evaluation",
                        "implementation", "training"]:
                s = self.get_section(key)
                if s:
                    parts.append(f"\n## {key}\n{s}")
            return "\n".join(parts)[:mc]
        elif stage == "coding":
            return self.get_full_text_truncated(max_chars=mc)
        return self.get_overview(max_chars=mc)