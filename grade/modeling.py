"""Shared model loading utilities for GRADE.

This module is the single place that constructs the IFM model, resolves
checkpoint geometry from config, and validates checkpoint loading strictly.
"""

from pathlib import Path
import torch
from transformers import AutoTokenizer, PretrainedConfig

from .ifm import GradeModel

ID2LABEL = {0: "NEGATIVE", 1: "POSITIVE"}
LABEL2ID = {"NEGATIVE": 0, "POSITIVE": 1}


def resolve_model_geometry(config: PretrainedConfig) -> tuple[int, int]:
    """Resolve cross-attention geometry from model config.

    The checkpoint is only valid for the geometry it was trained with, so these
    values are read from config rather than accepted as CLI overrides.
    """
    config_blocks = getattr(config, "cross_attn_layers", None)
    config_hidden = getattr(config, "cross_attn_hidden", None)

    if config_blocks is None or config_hidden is None:
        raise ValueError(
            "Model config must define cross_attn_layers and cross_attn_hidden; "
            "refusing to guess checkpoint geometry."
        )

    return int(config_blocks), int(config_hidden)


def load_checkpoint_state_dict(model_dir: Path) -> dict:
    """Load checkpoint tensors from a model directory."""
    if (model_dir / "model.safetensors").exists():
        try:
            from safetensors.torch import load_file
        except ImportError:
            raise ImportError("model.safetensors found but safetensors is not installed")
        return load_file(model_dir / "model.safetensors")

    if (model_dir / "pytorch_model.bin").exists():
        return torch.load(model_dir / "pytorch_model.bin", map_location="cpu", weights_only=False)

    raise FileNotFoundError(f"No checkpoint found in {model_dir}")


def load_grade_tokenizer(model_dir: str | Path):
    """Load the tokenizer bundled with a GRADE model directory."""
    return AutoTokenizer.from_pretrained(model_dir)


def load_grade_model(
    model_dir: str | Path,
    *,
    device: torch.device | None = None,
    attn_implementation: str = "sdpa",
) -> GradeModel:
    """Build GRADE and fail fast unless checkpoint keys match exactly."""
    model_dir = Path(model_dir)
    config = PretrainedConfig.from_pretrained(model_dir)
    resolved_blocks, resolved_hidden = resolve_model_geometry(config)

    model = GradeModel(
        config,
        num_cross_attention_blocks=resolved_blocks,
        hidden_size=resolved_hidden,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        attn_implementation=attn_implementation,
    )

    state_dict = load_checkpoint_state_dict(model_dir)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint did not match GradeModel exactly: "
            f"missing keys ({len(missing)}): {missing[:10]}"
            f"; unexpected keys ({len(unexpected)}): {unexpected[:10]}"
        )

    model.eval()
    if device is not None:
        model.to(device)
    return model
