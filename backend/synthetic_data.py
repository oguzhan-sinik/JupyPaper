"""
Synthetic dataset generation via NVIDIA NeMo Data Designer.

When a paper's dataset is not publicly available, this module generates
realistic synthetic data matching the paper's specifications using NeMo's
Data Designer microservice.

Two modes:
  1. INLINE: Generates Python code for the notebook that creates synthetic data
     using torch/numpy (simpler, no external service needed)
  2. NEMO API: Calls the NeMo Data Designer API for richer, LLM-generated
     synthetic data (requires running Data Designer service)

Reference: https://docs.nvidia.com/nemo/microservices/latest/design-synthetic-data-from-scratch-or-seeds/
"""

import os
import json
import time
from typing import Optional

import requests
from utils import console, log_sub


# ── NeMo Data Designer API ────────────────────────────────────────────────────

NEMO_DATA_DESIGNER_URL = os.getenv(
    "NEMO_DATA_DESIGNER_URL",
    "http://localhost:8082"  # default Docker Compose port
)


def is_nemo_available() -> bool:
    """Check if NeMo Data Designer service is reachable."""
    try:
        resp = requests.get(f"{NEMO_DATA_DESIGNER_URL}/health", timeout=5)
        return resp.ok
    except Exception:
        return False


def generate_via_nemo_api(
    data_plan: dict,
    num_records: int = 100,
    model: str = "meta/llama-3.3-70b-instruct",
) -> Optional[dict]:
    """
    Generate synthetic data via NeMo Data Designer API.

    Args:
        data_plan: from planning stage, contains column specs
        num_records: number of records to generate
        model: NIM model to use for LLM-based columns

    Returns:
        {"records": [...], "schema": {...}} or None on failure
    """
    log_sub("NeMo Data Designer", f"Generating {num_records} synthetic records")

    if not is_nemo_available():
        console.print("    [yellow]⚠ NeMo Data Designer not reachable — falling back to inline[/yellow]")
        return None

    try:
        # Build Data Designer configuration
        columns = data_plan.get("columns", [])
        data_type = data_plan.get("data_type", "generic")

        config = {
            "model_configs": [{
                "alias": "generator",
                "model": model,
                "inference_parameters": {
                    "temperature": 0.8,
                    "max_tokens": 1024,
                }
            }],
            "columns": [],
            "num_records": num_records,
        }

        for col in columns:
            col_config = _build_column_config(col, data_type)
            if col_config:
                config["columns"].append(col_config)

        # Call the Data Designer API
        resp = requests.post(
            f"{NEMO_DATA_DESIGNER_URL}/v1/data-designer/preview",
            json=config,
            timeout=120,
        )

        if resp.ok:
            result = resp.json()
            console.print(f"    [green]✓[/green] Generated {len(result.get('records', []))} records")
            return result
        else:
            console.print(f"    [yellow]⚠ Data Designer API returned {resp.status_code}[/yellow]")
            return None

    except Exception as e:
        console.print(f"    [yellow]⚠ Data Designer error: {e}[/yellow]")
        return None


def _build_column_config(col: dict, data_type: str) -> Optional[dict]:
    """Build a NeMo Data Designer column configuration."""
    col_type = col.get("type", "text")
    name = col.get("name", "column")

    if col_type == "categorical":
        return {
            "name": name,
            "type": "sampler",
            "sampler_type": "category",
            "params": {
                "values": [f"class_{i}" for i in range(col.get("num_classes", 2))],
            }
        }
    elif col_type == "token_sequence":
        return {
            "name": name,
            "type": "llm_text",
            "model_alias": "generator",
            "prompt": (
                f"Generate a realistic example {name} sequence of approximately "
                f"{col.get('max_length', 128)} tokens for a {data_type} task. "
                f"Output only the text, no explanation."
            ),
        }
    elif col_type == "text":
        return {
            "name": name,
            "type": "llm_text",
            "model_alias": "generator",
            "prompt": (
                f"Generate a realistic {name} text sample for a {data_type} task. "
                f"Maximum length: {col.get('max_length', 256)} words. "
                f"Output only the text."
            ),
        }
    elif col_type == "numeric":
        return {
            "name": name,
            "type": "sampler",
            "sampler_type": "gaussian",
            "params": {
                "mean": col.get("mean", 0.0),
                "std": col.get("std", 1.0),
            }
        }
    return None


# ── Inline Code Generation ────────────────────────────────────────────────────

def generate_inline_code(data_plan: dict, config: dict) -> str:
    """
    Generate Python code that creates synthetic data inline in the notebook.

    This is the fallback when NeMo Data Designer is not available.
    Produces self-contained code that uses only torch/numpy.
    """
    data_type = data_plan.get("data_type", "generic")
    seq_len = data_plan.get("sequence_length") or 128
    vocab_size = data_plan.get("vocab_size") or 10000
    train_samples = data_plan.get("train_samples", 1000)
    val_samples = data_plan.get("val_samples", 200)

    model_config = config.get("model", {})
    data_config = config.get("data", {})

    if data_type == "sequence_pair":
        return _gen_seq2seq_data(seq_len, vocab_size, train_samples, val_samples, data_plan)
    elif data_type == "text_classification":
        return _gen_classification_data(seq_len, vocab_size, train_samples, val_samples, data_plan)
    elif data_type == "language_modeling":
        return _gen_lm_data(seq_len, vocab_size, train_samples, val_samples, data_plan)
    else:
        return _gen_generic_data(seq_len, vocab_size, train_samples, val_samples, model_config)


def _gen_seq2seq_data(seq_len, vocab_size, n_train, n_val, plan):
    datasets_desc = plan.get("datasets_described", [{}])
    desc = datasets_desc[0].get("description", "sequence-to-sequence translation") if datasets_desc else "seq2seq"
    return f'''# ═══════════════════════════════════════════════════════════════════
# Synthetic Dataset: Sequence-to-Sequence
# Real data: {desc}
# This creates synthetic token sequences for demonstration.
# Replace with real data loading for actual reproduction.
# ═══════════════════════════════════════════════════════════════════

# Synthetic data parameters (matching paper specifications where possible)
VOCAB_SIZE = {vocab_size}
MAX_SEQ_LEN = {seq_len}
NUM_TRAIN = {n_train}
NUM_VAL = {n_val}
PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2

def create_synthetic_seq2seq(num_samples, max_len=MAX_SEQ_LEN, vocab_size=VOCAB_SIZE):
    """
    Create synthetic source-target sequence pairs.
    Sequences have variable length with BOS/EOS tokens.
    Target is a 'noisy copy' of source to simulate translation.
    """
    sources, targets, src_masks, tgt_masks = [], [], [], []

    for _ in range(num_samples):
        # Variable-length sequences (50% to 100% of max_len)
        src_len = torch.randint(max_len // 2, max_len - 1, (1,)).item()
        tgt_len = torch.randint(max_len // 2, max_len - 1, (1,)).item()

        # Source: [BOS, tokens..., EOS, PAD...]
        src_tokens = torch.randint(3, vocab_size, (src_len,))
        src = torch.cat([torch.tensor([BOS_TOKEN]), src_tokens, torch.tensor([EOS_TOKEN])])
        src = torch.nn.functional.pad(src, (0, max_len - len(src)), value=PAD_TOKEN)

        # Target: noisy permutation of source (simulates translation)
        tgt_tokens = src_tokens[torch.randperm(min(src_len, tgt_len))][:tgt_len]
        # Add some random token swaps
        noise_mask = torch.rand(tgt_len) < 0.1
        tgt_tokens[noise_mask] = torch.randint(3, vocab_size, (noise_mask.sum(),))
        tgt = torch.cat([torch.tensor([BOS_TOKEN]), tgt_tokens, torch.tensor([EOS_TOKEN])])
        tgt = torch.nn.functional.pad(tgt, (0, max_len - len(tgt)), value=PAD_TOKEN)

        sources.append(src[:max_len])
        targets.append(tgt[:max_len])
        src_masks.append((src[:max_len] != PAD_TOKEN).long())
        tgt_masks.append((tgt[:max_len] != PAD_TOKEN).long())

    return {{
        "src": torch.stack(sources),
        "tgt": torch.stack(targets),
        "src_mask": torch.stack(src_masks),
        "tgt_mask": torch.stack(tgt_masks),
    }}

# Generate train and validation sets
print("Creating synthetic training data...")
train_data = create_synthetic_seq2seq(NUM_TRAIN)
val_data = create_synthetic_seq2seq(NUM_VAL)

# Create DataLoaders
train_dataset = torch.utils.data.TensorDataset(
    train_data["src"], train_data["tgt"],
    train_data["src_mask"], train_data["tgt_mask"]
)
val_dataset = torch.utils.data.TensorDataset(
    val_data["src"], val_data["tgt"],
    val_data["src_mask"], val_data["tgt_mask"]
)

batch_size = config.get("batch_size", 32)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

print(f"Train: {{len(train_dataset)}} samples, {{len(train_loader)}} batches")
print(f"Val:   {{len(val_dataset)}} samples, {{len(val_loader)}} batches")
print(f"Source shape: {{train_data['src'].shape}}")
print(f"Target shape: {{train_data['tgt'].shape}}")'''


def _gen_classification_data(seq_len, vocab_size, n_train, n_val, plan):
    return f'''# ═══════════════════════════════════════════════════════════════════
# Synthetic Dataset: Text Classification
# Replace with real data for actual reproduction.
# ═══════════════════════════════════════════════════════════════════

NUM_TRAIN = {n_train}
NUM_VAL = {n_val}
NUM_CLASSES = 2
MAX_SEQ_LEN = {seq_len}
VOCAB_SIZE = {vocab_size}

def create_synthetic_classification(num_samples, max_len=MAX_SEQ_LEN):
    """Create synthetic text classification data with label-dependent patterns."""
    inputs, labels, attention_masks = [], [], []

    for _ in range(num_samples):
        label = torch.randint(0, NUM_CLASSES, (1,)).item()
        seq_len = torch.randint(max_len // 4, max_len, (1,)).item()

        # Create tokens with slight class-dependent bias
        offset = label * (VOCAB_SIZE // NUM_CLASSES)
        tokens = torch.randint(offset, offset + VOCAB_SIZE // NUM_CLASSES, (seq_len,))
        padded = torch.nn.functional.pad(tokens, (0, max_len - seq_len), value=0)
        mask = torch.cat([torch.ones(seq_len), torch.zeros(max_len - seq_len)])

        inputs.append(padded[:max_len])
        labels.append(label)
        attention_masks.append(mask[:max_len])

    return {{
        "input_ids": torch.stack(inputs),
        "labels": torch.tensor(labels),
        "attention_mask": torch.stack(attention_masks).long(),
    }}

train_data = create_synthetic_classification(NUM_TRAIN)
val_data = create_synthetic_classification(NUM_VAL)

train_dataset = torch.utils.data.TensorDataset(
    train_data["input_ids"], train_data["attention_mask"], train_data["labels"]
)
val_dataset = torch.utils.data.TensorDataset(
    val_data["input_ids"], val_data["attention_mask"], val_data["labels"]
)

batch_size = config.get("batch_size", 32)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

print(f"Train: {{len(train_dataset)}} samples ({{NUM_CLASSES}} classes)")
print(f"Val:   {{len(val_dataset)}} samples")
print(f"Input shape: {{train_data['input_ids'].shape}}")'''


def _gen_lm_data(seq_len, vocab_size, n_train, n_val, plan):
    return f'''# ═══════════════════════════════════════════════════════════════════
# Synthetic Dataset: Language Modeling
# Replace with real corpus for actual reproduction.
# ═══════════════════════════════════════════════════════════════════

NUM_TRAIN = {n_train}
NUM_VAL = {n_val}
MAX_SEQ_LEN = {seq_len}
VOCAB_SIZE = {vocab_size}

def create_synthetic_lm_data(num_samples, seq_len=MAX_SEQ_LEN):
    """Create synthetic language modeling data (next-token prediction)."""
    sequences = torch.randint(3, VOCAB_SIZE, (num_samples, seq_len + 1))
    input_ids = sequences[:, :-1]   # input tokens
    labels = sequences[:, 1:]       # shifted by 1 for next-token prediction
    attention_mask = torch.ones_like(input_ids)
    return {{"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}}

train_data = create_synthetic_lm_data(NUM_TRAIN)
val_data = create_synthetic_lm_data(NUM_VAL)

train_dataset = torch.utils.data.TensorDataset(
    train_data["input_ids"], train_data["labels"], train_data["attention_mask"]
)
val_dataset = torch.utils.data.TensorDataset(
    val_data["input_ids"], val_data["labels"], val_data["attention_mask"]
)

batch_size = config.get("batch_size", 32)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

print(f"Train: {{len(train_dataset)}} samples, seq_len={{MAX_SEQ_LEN}}")
print(f"Val:   {{len(val_dataset)}} samples")'''


def _gen_generic_data(seq_len, vocab_size, n_train, n_val, model_config):
    hidden = model_config.get("hidden_size") or 256
    return f'''# ═══════════════════════════════════════════════════════════════════
# Synthetic Dataset: Generic tensor data
# Replace with real data for actual reproduction.
# ═══════════════════════════════════════════════════════════════════

NUM_TRAIN = {n_train}
NUM_VAL = {n_val}
INPUT_DIM = {hidden}

# Create random tensor data matching expected input dimensions
train_inputs = torch.randn(NUM_TRAIN, {seq_len}, INPUT_DIM)
train_targets = torch.randint(0, 10, (NUM_TRAIN,))
val_inputs = torch.randn(NUM_VAL, {seq_len}, INPUT_DIM)
val_targets = torch.randint(0, 10, (NUM_VAL,))

train_dataset = torch.utils.data.TensorDataset(train_inputs, train_targets)
val_dataset = torch.utils.data.TensorDataset(val_inputs, val_targets)

batch_size = config.get("batch_size", 32)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

print(f"Train: {{len(train_dataset)}} samples, input shape: {{train_inputs.shape}}")
print(f"Val:   {{len(val_dataset)}} samples")'''