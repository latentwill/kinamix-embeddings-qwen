"""
modules/low_rank_injection.py — Low-rank factorized DiT concept injection.

Instead of N × hidden_dim free parameters, concept tokens are factorized as
A @ B:
    A : [N, rank]          — per-concept direction coefficients
    B : [rank, hidden_dim] — shared basis vectors

Gradients concentrate through the rank bottleneck, preventing the isotropic
drift observed in full-rank DSCI runs (spectral analysis: tokens drift
uniformly with negligible magnitude at 1.063× init).

Same inject() / update_txt_seq_lens() interface as DiTConceptInjection.
Embedding serialization: concept_tokens property returns A @ B for
Embedding.from_dsci() compatibility.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LowRankDiTConceptInjection(nn.Module):
    """Low-rank factorized DSCI tokens: concept_tokens = A @ B.

    Args:
        hidden_dim:  Text encoder hidden size (3584 for Qwen2.5-VL).
        num_tokens:  Number of concept tokens injected (default 5).
        rank:        Inner rank of the factorization (default 8).
                     Lower rank → stronger gradient concentration.
                     rank=hidden_dim → equivalent to full-rank DSCI.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_tokens: int = 5,
        rank: int = 8,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.rank = rank

        # A: [N, rank]  —  per-concept coefficients
        self.A = nn.Parameter(torch.randn(num_tokens, rank) * 0.02)
        # B: [rank, hidden_dim]  —  shared basis
        self.B = nn.Parameter(torch.randn(rank, hidden_dim) * 0.02)

    @property
    def concept_tokens(self) -> torch.Tensor:
        """Compute materialized tokens [N, hidden_dim] for Embedding.from_dsci()."""
        return self.A @ self.B

    def inject(
        self,
        hidden_states: torch.Tensor,   # [B, seq, hidden_dim]
        attention_mask: torch.Tensor,  # [B, seq]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenate concept tokens to the conditioning sequence."""
        batch = hidden_states.shape[0]
        tokens = (self.A @ self.B).unsqueeze(0).expand(batch, -1, -1)
        tokens = tokens.to(dtype=hidden_states.dtype)
        hidden_states = torch.cat([hidden_states, tokens], dim=1)
        token_mask = torch.ones(
            batch, self.num_tokens,
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        attention_mask = torch.cat([attention_mask, token_mask], dim=1)
        return hidden_states, attention_mask

    def update_txt_seq_lens(self, txt_seq_lens: list[int]) -> list[int]:
        """Extend RoPE seq lengths to account for injected tokens."""
        return [l + self.num_tokens for l in txt_seq_lens]
