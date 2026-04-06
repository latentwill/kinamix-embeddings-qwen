"""modules/attention_hooks.py — T5 attention hook infrastructure for DSCI training.

Provides non-invasive hooks to extract attention probe metrics from the DiT
during training. Only activates at diagnostic steps to minimize overhead.

Architecture notes (QwenImage double-stream):
- QwenImageTransformer2DModel uses QwenDoubleStreamAttnProcessor2_0
- Attention modules receive separate streams:
    - hidden_states (arg 0) = image tokens (already modulated)
    - encoder_hidden_states (arg 1) = text+concept tokens
- Image projections: to_q, to_k, to_v
- Text projections: add_q_proj, add_k_proj, add_v_proj
- Flash attention doesn't return attention matrices
- We compute a Q/K probe: softmax(to_q(img) @ add_k_proj(txt)[-n_concept:]^T)
- This is a PROBE metric (softmax over concept keys only), not actual attention
"""
import re
from typing import Optional

import torch
import torch.nn as nn

from modules.attention_diagnostics import compute_spatial_entropy


def discover_attention_modules(
    model: nn.Module,
    pattern: str = "attn",
) -> list[tuple[str, nn.Module]]:
    """Find attention modules in a transformer by name pattern.

    Matches modules whose LAST name segment matches the pattern (not parent
    path). E.g., pattern="attn" matches "blocks.0.attn" but NOT
    "blocks.0.attn.to_q" (where "attn" is in the parent path, not the
    module's own name).

    Args:
        model: The transformer model to inspect.
        pattern: Substring to match in the module's own name segment.

    Returns:
        List of (fully_qualified_name, module) pairs.
    """
    results = []
    regex = re.compile(pattern, re.IGNORECASE)
    for name, module in model.named_modules():
        if not name:
            continue
        # Match only the last segment of the dotted name
        last_segment = name.rsplit(".", 1)[-1]
        if regex.search(last_segment):
            results.append((name, module))
    return results


class AttentionCollector:
    """Collect concept-token attention probe metrics from DiT blocks.

    Handles the QwenImage double-stream architecture where:
    - hidden_states (arg 0) = image stream
    - encoder_hidden_states (arg 1 or kwarg) = text+concept stream
    - Image projections: to_q, to_k
    - Text projections: add_q_proj, add_k_proj

    Computes actual attention weights by building the full key set
    (image + text + concept), computing softmax over ALL keys, then
    extracting only the concept-token columns. This gives the true
    fraction of attention each image patch allocates to concept tokens.

    Set collector.active = True only at diagnostic steps to avoid overhead.
    """

    def __init__(
        self,
        model: nn.Module,
        n_img_tokens: int,
        n_concept_tokens: int,
        concept_position: str = "append",
        attn_pattern: str = "attn",
    ):
        self.model = model
        self.n_img = n_img_tokens
        self.n_concept = n_concept_tokens
        self.concept_position = concept_position
        self.attn_pattern = attn_pattern
        self.active = False
        self._hooks: list = []
        self._captured: dict[str, torch.Tensor] = {}

    def register(self):
        """Register forward hooks on discovered attention modules.

        Uses register_forward_pre_hook to capture inputs BEFORE the forward
        pass processes them. This is more reliable than post-hooks with
        with_kwargs, which may not fire on all PyTorch/diffusers versions.
        """
        modules = discover_attention_modules(self.model, pattern=self.attn_pattern)
        for i, (name, module) in enumerate(modules):
            block_key = f"block_{i}"

            def make_hook(key):
                def hook_fn(mod, args, kwargs):
                    if self.active:
                        self._capture_double_stream(mod, args, kwargs, key)
                return hook_fn

            handle = module.register_forward_pre_hook(make_hook(block_key), with_kwargs=True)
            self._hooks.append(handle)

    def _capture_double_stream(self, module, args, kwargs, block_key):
        """Extract Q_img and K_concept from double-stream attention.

        Diffusers Attention.forward() signature:
          forward(hidden_states, encoder_hidden_states=None, ...)
        """
        # Extract hidden_states (always first positional arg)
        img_hs = args[0] if args else kwargs.get("hidden_states")

        # Extract encoder_hidden_states (may be positional or keyword)
        txt_hs = None
        if len(args) > 1 and isinstance(args[1], torch.Tensor):
            txt_hs = args[1]
        if txt_hs is None:
            txt_hs = kwargs.get("encoder_hidden_states")

        # Debug: log once what we're seeing
        if block_key == "block_0" and not hasattr(self, "_debug_logged"):
            self._debug_logged = True
            import logging
            logging.warning(
                f"T5 hook debug: args={len(args)}, "
                f"img_hs={img_hs.shape if isinstance(img_hs, torch.Tensor) else type(img_hs)}, "
                f"txt_hs={'None' if txt_hs is None else txt_hs.shape}, "
                f"kwargs_keys={list(kwargs.keys())}"
            )

        if not isinstance(img_hs, torch.Tensor) or img_hs.dim() != 3:
            return
        if txt_hs is None or not isinstance(txt_hs, torch.Tensor):
            return

            # Find double-stream projections
        # Image: to_q, to_k  |  Text: add_q_proj, add_k_proj
        projs = {}
        for child_name, child in module.named_modules():
            if child_name in ("to_q", "to_k", "add_k_proj"):
                projs[child_name] = child

        if "to_q" not in projs or "to_k" not in projs or "add_k_proj" not in projs:
            return

        with torch.no_grad():
            Q_img = projs["to_q"](img_hs)       # [B, n_img, D]
            K_img = projs["to_k"](img_hs)       # [B, n_img, D]
            K_txt = projs["add_k_proj"](txt_hs)  # [B, n_txt+n_concept, D]

            B, _, D = Q_img.shape

            # Determine head count
            n_heads = getattr(module, "heads", None) or getattr(module, "num_heads", None)
            if n_heads is None and hasattr(projs["to_q"], "out_features"):
                for hd in [128, 64, 32]:
                    if projs["to_q"].out_features % hd == 0:
                        n_heads = projs["to_q"].out_features // hd
                        break
            if n_heads is None:
                n_heads = 1

            head_dim = D // n_heads

            # Build full K: [txt+concept, img] — matching Qwen joint attention order
            K_all = torch.cat([K_txt, K_img], dim=1)  # [B, n_txt+n_concept+n_img, D]
            n_all = K_all.shape[1]

            # Reshape to multi-head
            q = Q_img.view(B, self.n_img, n_heads, head_dim).transpose(1, 2)
            k = K_all.view(B, n_all, n_heads, head_dim).transpose(1, 2)

            # Full attention: softmax over ALL keys (image + text + concept)
            # [B, heads, n_img, n_all]
            attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
            attn_full = torch.softmax(attn_logits, dim=-1)

            # Extract concept token columns from the full attention matrix
            # K_all order: [txt_tokens, concept_tokens, img_tokens]
            n_txt = txt_hs.shape[1]
            if self.concept_position == "prepend":
                concept_start = 0
            else:  # append — concept tokens are at end of text sequence
                concept_start = n_txt - self.n_concept

            attn_concept = attn_full[:, :, :, concept_start:concept_start + self.n_concept]
            # [B, heads, n_img, n_concept] — actual attention weights in context of all keys

            self._captured[block_key] = attn_concept.cpu()

            # Free the large intermediate tensors immediately
            del attn_logits, attn_full, K_all

    def compute_metrics(self) -> dict[str, float]:
        """Compute aggregate and per-block attention probe metrics.

        Returns dict with probe_ prefixed metrics (to distinguish from
        actual attention weights).
        """
        if not self._captured:
            return {}

        metrics: dict[str, float] = {}
        all_entropy = []
        all_mass = []
        all_max = []

        for block_key, attn in sorted(self._captured.items()):
            entropy = compute_spatial_entropy(attn)
            mass = attn.sum(dim=-1).mean().item()
            max_val = attn.max().item()

            metrics[f"{block_key}/probe_entropy"] = entropy
            metrics[f"{block_key}/probe_mass"] = mass
            metrics[f"{block_key}/probe_max"] = max_val

            all_entropy.append(entropy)
            all_mass.append(mass)
            all_max.append(max_val)

        metrics["probe_entropy"] = sum(all_entropy) / len(all_entropy)
        metrics["probe_mass"] = sum(all_mass) / len(all_mass)
        metrics["probe_max"] = max(all_max)
        metrics["n_blocks_captured"] = len(self._captured)

        self._captured.clear()
        return metrics

    def cleanup(self):
        """Remove all hooks and clear captured data."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self._captured.clear()
