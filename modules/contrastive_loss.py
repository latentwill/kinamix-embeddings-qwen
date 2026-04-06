"""
modules/contrastive_loss.py — Contrastive penalty for DSCI training.

Pushes concept tokens away from language basin cluster vectors ("illustration",
"monochrome", etc.) that the text encoder collapses visual concepts into.

Usage in train_dsci.py:
    # Once at training start (text_encoder is already loaded):
    prior_vecs = build_language_priors(text_encoder, tokenizer, device)

    # Each step:
    penalty = contrastive_penalty(dsci.concept_tokens, prior_vecs)
    loss = flow_loss + contrastive_weight * penalty
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Phrases that represent the language basin cluster we want to escape.
# These are the attractor basins DSCI was designed to bypass — keeping concept
# tokens far from these directions in hidden space reinforces that bypass.
LANGUAGE_PRIOR_PHRASES: list[str] = [
    "illustration",
    "a monochrome drawing",
    "charcoal sketch",
    "ink drawing",
    "black and white illustration",
    "digital art",
    "a painting",
]


def build_language_priors(
    text_encoder: object,
    tokenizer: object,
    device: torch.device,
    phrases: list[str] | None = None,
) -> torch.Tensor:
    """Encode language prior phrases and return mean-pooled vectors [K, hidden_dim].

    Call once at training start with the already-loaded text encoder.
    The result is a fixed reference tensor — no gradient flows through it.

    Args:
        text_encoder: Frozen Qwen2.5-VL text encoder.
        tokenizer:    Matching tokenizer.
        device:       CUDA device.
        phrases:      Override default LANGUAGE_PRIOR_PHRASES.

    Returns:
        [K, hidden_dim] float32 tensor on `device`, detached from any graph.
    """
    if phrases is None:
        phrases = LANGUAGE_PRIOR_PHRASES

    # Lazy import to avoid circular dependency (preview.py imports from this
    # project's modules; encode_prompt is the same function used in training).
    from preview import encode_prompt

    vecs: list[torch.Tensor] = []
    with torch.no_grad():
        for phrase in phrases:
            hs, mask = encode_prompt(text_encoder, tokenizer, phrase, device)
            # Mean-pool over valid (non-padding) token positions
            lengths = mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
            pooled = (hs * mask.unsqueeze(-1).float()).sum(dim=1) / lengths
            vecs.append(pooled.squeeze(0))

    return torch.stack(vecs, dim=0).detach()  # [K, hidden_dim]


def contrastive_penalty(
    concept_tokens: torch.Tensor,  # [N, hidden_dim]
    prior_vectors: torch.Tensor,   # [K, hidden_dim]
    margin: float = 0.3,
) -> torch.Tensor:
    """Margin-based penalty that keeps concept tokens far from prior vectors.

    Only penalizes when cosine similarity exceeds (1 - margin), so tokens
    already far from the language basin incur zero loss. This avoids pulling
    tokens toward some arbitrary "anti-illustration" direction.

    Args:
        concept_tokens: Current DSCI token matrix [N, hidden_dim].
        prior_vectors:  Language prior cluster vectors [K, hidden_dim].
        margin:         Minimum cosine distance from any prior (default 0.3).
                        Smaller margin = looser constraint; larger = stricter.

    Returns:
        Scalar penalty tensor (differentiable through concept_tokens).
    """
    tokens_norm = F.normalize(concept_tokens.float(), dim=-1)  # [N, D]
    priors_norm = F.normalize(prior_vectors.float(), dim=-1)   # [K, D]

    # Cosine similarity matrix [N, K]
    sims = tokens_norm @ priors_norm.t()

    # Per-token maximum similarity to any prior [N]
    max_sims, _ = sims.max(dim=-1)

    # Hinge: penalize only when sim > (1 - margin)
    threshold = 1.0 - margin
    penalty = F.relu(max_sims - threshold).mean()

    return penalty
