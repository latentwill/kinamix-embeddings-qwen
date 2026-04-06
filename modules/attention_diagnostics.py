"""modules/attention_diagnostics.py — Lightweight attention map diagnostics for DSCI training.

Provides pure functions for computing metrics from attention maps,
plus a hook-based collector for extracting maps during training.
"""
import json
import torch
from pathlib import Path


def compute_spatial_entropy(attn: torch.Tensor) -> float:
    """Compute spatial entropy of concept token attention.

    Args:
        attn: [B, heads, img_tokens, concept_tokens] attention weights.

    Returns:
        Mean entropy across batch, heads, and concept tokens (nats).
    """
    spatial = attn.mean(dim=(1, 3))
    spatial = spatial / (spatial.sum(dim=-1, keepdim=True) + 1e-8)
    entropy = -(spatial * torch.log(spatial + 1e-8)).sum(dim=-1)
    return entropy.mean().item()


def compute_attention_mass(attn: torch.Tensor) -> float:
    """Total attention mass on concept tokens (averaged over batch/heads).

    Args:
        attn: [B, heads, img_tokens, concept_tokens].
    """
    mass_per_img = attn.sum(dim=-1)  # [B, heads, img_tokens]
    return mass_per_img.mean().item()


def compute_max_attention(attn: torch.Tensor) -> float:
    """Maximum attention weight any image patch gives to any concept token."""
    return attn.max().item()


def save_diagnostics(metrics: dict, output_dir: str, step: int):
    """Save diagnostic metrics as JSON."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    out = path / f"attention_step{step:04d}.json"
    with open(out, "w") as f:
        json.dump({"step": step, "metrics": metrics}, f, indent=2)
