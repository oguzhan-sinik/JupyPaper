"""
Pydantic models for structured LLM output.

Used with NVIDIA NIM's guided_json to guarantee valid JSON responses,
eliminating the regex-based extract_json() failures.

NIM integration:
  - Hosted API:  extra_body={"nvext": {"guided_json": Model.model_json_schema()}}
  - Fallback:    response_format={"type": "json_object"} + Pydantic validation
  - LangChain:   ChatNVIDIA.with_structured_output(Model)
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


# ── Stage 1a: Paper Analysis ──────────────────────────────────────────────────

class DatasetInfo(BaseModel):
    name: str = Field(default="", description="Dataset name")
    description: str = Field(default="", description="Format, size, preprocessing")
    url_or_source: Optional[str] = Field(default=None, description="URL or null")
    is_public: bool = Field(default=False)


class TrainingDetails(BaseModel):
    optimizer: Optional[str] = None
    learning_rate: Optional[str] = None
    batch_size: Optional[str] = None
    epochs: Optional[str] = None
    loss_function: Optional[str] = None
    regularization: Optional[str] = None


class ModelArchitecture(BaseModel):
    type: Optional[str] = Field(default=None, description="transformer/CNN/RNN/hybrid")
    input_format: Optional[str] = None
    output_format: Optional[str] = None
    key_dimensions: Optional[dict[str, str]] = Field(default_factory=dict)
    num_parameters: Optional[str] = None


class PaperAnalysis(BaseModel):
    """Full paper analysis output."""
    title: str = Field(description="Exact paper title")
    problem: str = Field(default="", description="2-3 sentence problem description")
    methodology: str = Field(default="", description="Detailed method description (500+ words)")
    key_components: list[str] = Field(default_factory=list, description="Key algorithmic components")
    datasets: list[DatasetInfo] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list, description="Evaluation metrics")
    hyperparameters: dict[str, str] = Field(default_factory=dict)
    baselines: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list, description="Required Python libraries")
    hardware: Optional[str] = None
    key_equations: list[str] = Field(default_factory=list, description="LaTeX equations")
    training_details: TrainingDetails = Field(default_factory=TrainingDetails)
    model_architecture: ModelArchitecture = Field(default_factory=ModelArchitecture)
    code_repo_mentioned: Optional[str] = Field(
        default=None,
        description="URL if paper mentions official code, else null"
    )
    figures_summary: Optional[str] = None


# ── Stage 1c: Config Extraction ───────────────────────────────────────────────

class TrainingConfig(BaseModel):
    learning_rate: Optional[str] = None
    lr_schedule: Optional[str] = None
    batch_size: Optional[str] = None
    epochs: Optional[str] = None
    optimizer: Optional[str] = None
    weight_decay: Optional[str] = None
    warmup_steps: Optional[str] = None
    gradient_clipping: Optional[str] = None
    label_smoothing: Optional[str] = None
    dropout: Optional[str] = None


class ModelConfig(BaseModel):
    architecture: Optional[str] = None
    hidden_size: Optional[str] = None
    num_layers: Optional[str] = None
    num_heads: Optional[str] = None
    ffn_dim: Optional[str] = None
    dropout: Optional[str] = None
    vocab_size: Optional[str] = None
    max_seq_length: Optional[str] = None
    embedding_dim: Optional[str] = None
    activation: Optional[str] = None


class DataConfig(BaseModel):
    dataset_name: Optional[str] = None
    dataset_url: Optional[str] = None
    train_size: Optional[str] = None
    val_size: Optional[str] = None
    test_size: Optional[str] = None
    preprocessing: Optional[str] = None
    input_format: Optional[str] = None
    sequence_length: Optional[str] = None
    vocab_size: Optional[str] = None
    tokenizer: Optional[str] = None


class EvalConfig(BaseModel):
    metrics: list[str] = Field(default_factory=list)
    test_sets: list[str] = Field(default_factory=list)
    beam_size: Optional[str] = None


class ConfigExtraction(BaseModel):
    """Full config extraction output."""
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    evaluation: EvalConfig = Field(default_factory=EvalConfig)
    other: dict[str, str] = Field(default_factory=dict)


# ── Stage 1d: Notebook Architecture ──────────────────────────────────────────

class CellSpec(BaseModel):
    cell_id: int
    cell_type: str = Field(description="markdown or code")
    title: str
    description: str = Field(default="", description="2-3 sentence cell content description")
    dependencies: list[int] = Field(default_factory=list, description="Prior cell_ids needed")
    estimated_complexity: str = Field(default="medium", description="low|medium|high")


class NotebookStructure(BaseModel):
    """Notebook architecture output."""
    notebook_title: str = Field(description="ACTUAL paper title")
    notebook_description: str = Field(default="")
    cells: list[CellSpec] = Field(default_factory=list)
    estimated_packages: list[str] = Field(default_factory=list)
    generation_order: list[int] = Field(default_factory=list)