"""
modules/cfg_aware_loss.py — CFG-aware training losses for DSCI.

Three additive losses that optimize concept tokens to produce a strong,
consistent concept direction (d = v_full - v_text) for CFG decomposition.

    DMag (dmag_weight):  Maximize ||d|| — larger signal for CFG to amplify.
    CDA  (cda_weight):   Cross-image direction alignment via rolling buffer —
                         forces concept to encode style (shared) not content.
    TID  (tid_weight):   Timestep-invariant direction — same concept direction
                         at different noise levels forces noise-invariant style.

All three are additive to the standard flow matching loss and can be combined.
"""

from __future__ import annotations

from collections import deque

import torch
import torch.nn.functional as F


# ── Inline copy of _pack_latents to avoid circular import ─────────────────────

def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """
    Pack spatial latents into patch sequences for the transformer.

    (B, C, H, W) → (B, (H/2)*(W/2), C*4)

    Inline copy from modules/dataset_and_loss.py to avoid circular import.
    """
    bsz, c, h, w = latents.shape
    latents = latents.view(bsz, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)  # (B, H/2, W/2, C, 2, 2)
    latents = latents.reshape(bsz, (h // 2) * (w // 2), c * 4)
    return latents


# ── Transformer forward helper ────────────────────────────────────────────────

def _run_transformer(
    transformer,
    noisy_packed: torch.Tensor,
    timesteps: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    img_shapes: list,
) -> torch.Tensor:
    """
    Single transformer forward pass, returns velocity prediction (output[0]).

    Args:
        transformer: The frozen DiT model.
        noisy_packed: Packed noisy latents (B, seq, C*4).
        timesteps: Continuous timestep values (B,).
        encoder_hidden_states: Text + concept hidden states (B, seq, hidden).
        attention_mask: Attention mask for encoder states, or None.
        img_shapes: List of (frame, h//2, w//2) tuples for RoPE positioning.

    Returns:
        Velocity prediction tensor, shape matching noisy_packed.
    """
    return transformer(
        hidden_states=noisy_packed,
        timestep=timesteps,
        encoder_hidden_states=encoder_hidden_states,
        encoder_hidden_states_mask=attention_mask,
        img_shapes=img_shapes,
        return_dict=False,
    )[0]


# ── Rolling direction buffer for CDA loss ─────────────────────────────────────

class DirectionBuffer:
    """
    Rolling buffer of recent concept directions for CDA (cross-image direction
    alignment) loss.

    Stores the last `capacity` mean-normalized concept directions from training
    batches. The mean of these buffered directions acts as a stable target that
    the current batch's direction is pulled toward, forcing consistent style
    encoding across different training images.
    """

    def __init__(self, capacity: int = 8) -> None:
        self._buf: deque[torch.Tensor] = deque(maxlen=capacity)
        self.capacity = capacity

    def update(self, d_normed: torch.Tensor) -> None:
        """
        Append a (hidden_dim,) detached direction tensor to the buffer.

        If at capacity, the oldest entry is automatically dropped.

        Args:
            d_normed: Mean-normalized direction vector, shape (hidden_dim,).
                      Must be detached before passing.
        """
        self._buf.append(d_normed.detach().cpu())

    def mean_direction(self) -> torch.Tensor | None:
        """
        Compute the mean of all buffered directions.

        Returns:
            Mean direction tensor, shape (hidden_dim,), on CPU.
            None if fewer than 2 directions have been buffered.
        """
        if len(self._buf) < 2:
            return None
        stacked = torch.stack(list(self._buf), dim=0)  # (N, hidden_dim)
        return stacked.mean(dim=0)

    def direction_variance(self) -> float | None:
        """
        Compute direction variance as 1 - mean(cosine_sim) across all pairs.

        Returns:
            Float in [0, 1] where 0 = all identical, 1 = all orthogonal.
            None if fewer than 2 directions buffered.
        """
        if len(self._buf) < 2:
            return None
        stacked = torch.stack(list(self._buf), dim=0)  # (N, hidden_dim)
        normed = torch.nn.functional.normalize(stacked, p=2, dim=-1)
        # Pairwise cosine similarity matrix
        sim_matrix = normed @ normed.T  # (N, N)
        # Extract upper triangle (excluding diagonal)
        n = sim_matrix.shape[0]
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        pairwise_sims = sim_matrix[mask]
        return float((1.0 - pairwise_sims.mean()).clamp(0.0, 1.0).item())

    def __len__(self) -> int:
        return len(self._buf)


# ── Direction normalization helper ────────────────────────────────────────────

def _mean_normed(d: torch.Tensor) -> torch.Tensor:
    """
    Reduce (B, seq, hidden) → (B, hidden) by averaging over seq, then L2-normalize.

    Gradient flows through d, so this is differentiable when called on v_full-derived
    tensors (where v_full has grad).

    Args:
        d: Direction tensor of shape (B, seq, hidden).

    Returns:
        L2-normalized mean direction, shape (B, hidden).
    """
    d_mean = d.mean(dim=1)  # (B, hidden)
    return F.normalize(d_mean, p=2, dim=-1)


# ── Main CFG-aware loss computation ───────────────────────────────────────────

def compute_cfg_aware_loss(
    transformer,
    scheduler,
    latents: torch.Tensor,
    hidden_states_concept: torch.Tensor,
    attention_mask_concept: torch.Tensor | None,
    hidden_states_text: torch.Tensor,
    attention_mask_text: torch.Tensor | None,
    dmag_weight: float = 0.0,
    cda_weight: float = 0.0,
    direction_buffer: DirectionBuffer | None = None,
    tid_weight: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """
    Compute flow matching loss with optional CFG-aware auxiliary losses.

    The primary flow matching loss uses hidden_states_concept (text + concept
    tokens) so gradient flows to the concept tokens. The auxiliary losses
    operate on d = v_full - v_text, the concept-induced direction in velocity
    space, which CFG amplifies at inference.

    Args:
        transformer: Frozen DiT model.
        scheduler: Flow matching scheduler (unused here but kept for API parity
                   with flow_matching_loss).
        latents: Clean latents (B, C, H, W).
        hidden_states_concept: Hidden states with concept tokens injected (B, seq, hidden).
        attention_mask_concept: Attention mask for concept-injected states.
        hidden_states_text: Hidden states WITHOUT concept tokens (B, seq, hidden).
        attention_mask_text: Attention mask for text-only states.
        dmag_weight: Weight for delta magnitude loss. Maximizes ||v_full - v_text||.
                     0.0 = disabled. Recommended: 0.0005–0.005 (packed latent norms ~8–15).
        cda_weight: Weight for cross-image direction alignment loss.
                    0.0 = disabled. Recommended: 0.2–0.5.
        direction_buffer: Rolling buffer for CDA loss. Required when cda_weight > 0.
        tid_weight: Weight for timestep-invariant direction loss.
                    0.0 = disabled. Recommended: 0.1–0.3.

    Returns:
        Tuple of (total_loss, metrics_dict).
        metrics_dict always contains "flow_loss". Additional keys are present
        when their corresponding loss is active: "d_magnitude", "dmag_loss",
        "cda_loss", "tid_loss".
    """
    bsz = latents.shape[0]
    device = latents.device
    _, _, h, w = latents.shape

    cfg_losses_active = dmag_weight > 0.0 or cda_weight > 0.0 or tid_weight > 0.0

    # ── Sample timestep t1 and noise ─────────────────────────────────────────
    t1 = torch.rand(bsz, device=device, dtype=latents.dtype)
    noise = torch.randn_like(latents)

    t1_view = t1.view(-1, *([1] * (latents.ndim - 1)))
    noisy_t1 = (1 - t1_view) * latents + t1_view * noise
    target = noise - latents

    # Pack for transformer
    noisy_t1_packed = _pack_latents(noisy_t1)
    target_packed = _pack_latents(target)

    img_shapes = [(1, h // 2, w // 2)] * bsz

    # ── Primary forward pass (gradient flows) ─────────────────────────────────
    v_full = _run_transformer(
        transformer,
        noisy_t1_packed,
        t1,
        hidden_states_concept,
        attention_mask_concept,
        img_shapes,
    )

    flow_loss = F.mse_loss(v_full.float(), target_packed.float())
    total_loss = flow_loss
    metrics: dict = {"flow_loss": flow_loss.item()}

    # ── Early exit if no CFG losses ───────────────────────────────────────────
    if not cfg_losses_active:
        return total_loss, metrics

    # ── No-grad baseline: v_text at t1 ───────────────────────────────────────
    with torch.no_grad():
        v_text_t1 = _run_transformer(
            transformer,
            noisy_t1_packed,
            t1,
            hidden_states_text,
            attention_mask_text,
            img_shapes,
        )

    # Concept direction at t1: gradient flows through v_full only
    d_t1 = v_full - v_text_t1.detach()  # (B, seq, hidden)

    # ── DMag: maximize ||d|| ──────────────────────────────────────────────────
    if dmag_weight > 0.0:
        d_norm_val = d_t1.norm(dim=-1).mean()
        dmag_loss = -d_norm_val  # negative = maximize
        total_loss = total_loss + dmag_weight * dmag_loss
        metrics["d_magnitude"] = d_norm_val.item()
        metrics["dmag_loss"] = dmag_loss.item()

    # ── CDA: cross-image direction alignment ──────────────────────────────────
    if cda_weight > 0.0 and direction_buffer is not None:
        d_t1_normed = _mean_normed(d_t1)  # (B, hidden), grad flows
        mean_dir = direction_buffer.mean_direction()
        if mean_dir is not None:
            mean_dir_dev = mean_dir.to(device)
            cda_loss = 1.0 - (d_t1_normed * mean_dir_dev.detach()).sum(dim=-1).mean()
            total_loss = total_loss + cda_weight * cda_loss
            metrics["cda_loss"] = cda_loss.item()
        # Update buffer: mean over batch → (hidden,)
        direction_buffer.update(d_t1_normed.detach().mean(dim=0))

    # ── TID: timestep-invariant direction ─────────────────────────────────────
    if tid_weight > 0.0:
        # Sample a different timestep (same noise realization)
        t2 = torch.rand(bsz, device=device, dtype=latents.dtype)
        t2_view = t2.view(-1, *([1] * (latents.ndim - 1)))
        noisy_t2 = (1 - t2_view) * latents + t2_view * noise
        noisy_t2_packed = _pack_latents(noisy_t2)

        with torch.no_grad():
            v_full_t2 = _run_transformer(
                transformer,
                noisy_t2_packed,
                t2,
                hidden_states_concept,
                attention_mask_concept,
                img_shapes,
            )
            v_text_t2 = _run_transformer(
                transformer,
                noisy_t2_packed,
                t2,
                hidden_states_text,
                attention_mask_text,
                img_shapes,
            )
            d_t2 = v_full_t2 - v_text_t2
            d_t2_normed = _mean_normed(d_t2)  # (B, hidden), all detached

        # Current direction at t1 with grad
        d_t1_normed_tid = _mean_normed(d_t1)  # (B, hidden), grad flows
        tid_loss = 1.0 - (d_t1_normed_tid * d_t2_normed.detach()).sum(dim=-1).mean()
        total_loss = total_loss + tid_weight * tid_loss
        metrics["tid_loss"] = tid_loss.item()

    return total_loss, metrics
