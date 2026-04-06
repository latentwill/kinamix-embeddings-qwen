"""
modules/dit_injection.py -- DiT-Side Concept Injection (DSCI)

Learned concept tokens concatenated to encoder_hidden_states BEFORE
passing to the DiT. Gradient flows through DiT back to these tokens only.
The text encoder remains fully frozen.

Architecture investigation findings (QwenImageTransformer2DModel):
─────────────────────────────────────────────────────────────────
1. encoder_hidden_states [B, seq_len, 3584] goes through:
   - txt_norm (RMSNorm, no learned params to break)
   - txt_in (Linear 3584 -> 3072, projects into DiT inner dim)
   Then participates in joint attention with image tokens in every block.

2. txt_seq_lens is used ONLY by QwenEmbedRope.forward():
   - max_len = max(txt_seq_lens)
   - txt_freqs = pos_freqs[max_vid_index : max_vid_index + max_len]
   Adding N concept tokens means max_len increases by N, which simply
   allocates N more RoPE positions from the pre-computed table (4096 entries).

3. RoPE positions for text tokens start AFTER the image positions:
   - max_vid_index = max(height, width) from img_shapes (with scale_rope)
   - Text tokens get positions [max_vid_index, max_vid_index + max_len)
   - Concept tokens get the LAST N positions in this range (appended at end)

4. encoder_hidden_states_mask is passed to each transformer block but the
   QwenDoubleStreamAttnProcessor2_0 does NOT use it as an attention mask.
   It is only carried through for API compatibility. Extending it is safe.

5. No maximum sequence length in the DiT -- the RoPE frequency table has
   4096 entries, far exceeding any practical text + concept length.

6. CONCLUSION: Variable-length conditioning is fully supported. We can
   concatenate N learned tokens to encoder_hidden_states, extend the mask,
   and update txt_seq_lens. No modifications to the DiT internals needed.
"""

import torch
import torch.nn as nn
from typing import Optional


class DiTConceptInjection(nn.Module):
    """
    Concatenates N learned concept tokens to text encoder hidden states.

    The concept tokens are appended AFTER the text tokens in the sequence
    dimension. This places them at the end of the text conditioning, where
    they get RoPE positions naturally extending from the text.

    During training, only these concept tokens receive gradients.
    The text encoder and DiT are completely frozen.

    Args:
        hidden_dim: Dimension of encoder_hidden_states (3584 for Qwen-Image).
        num_tokens: Number of concept tokens to inject (default 4).
    """

    def __init__(self, hidden_dim: int = 3584, num_tokens: int = 4):
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        self.concept_tokens = nn.Parameter(
            torch.randn(num_tokens, hidden_dim) * 0.02
        )

    def inject(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position: str = "append",
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Concatenate concept tokens to encoder hidden states.

        Args:
            hidden_states: Text encoder output [B, seq_len, hidden_dim] (bf16).
            attention_mask: Optional mask [B, seq_len] (long). None is valid.
            position: Where to place concept tokens relative to text tokens.
                "append" (default) - after text tokens (backward compatible)
                "prepend" - before text tokens
                "interleave" - distributed evenly among text tokens

        Returns:
            (modified_hidden_states, modified_mask):
                - modified_hidden_states: [B, seq_len + N, hidden_dim]
                - modified_mask: [B, seq_len + N] or None if input was None
        """
        batch_size = hidden_states.shape[0]

        # Expand concept tokens to batch: (N, D) -> (B, N, D), match device and dtype
        tokens = self.concept_tokens.to(device=hidden_states.device, dtype=hidden_states.dtype)
        tokens = tokens.unsqueeze(0).expand(batch_size, -1, -1)

        if position == "prepend":
            modified_hidden = torch.cat([tokens, hidden_states], dim=1)
        elif position == "interleave":
            seq_len = hidden_states.shape[1]
            n_tokens = self.num_tokens
            spacing = max(1, seq_len // (n_tokens + 1))
            parts = []
            token_idx = 0
            for i in range(seq_len):
                parts.append(hidden_states[:, i:i + 1, :])
                if token_idx < n_tokens and (i + 1) % spacing == 0:
                    parts.append(tokens[:, token_idx:token_idx + 1, :])
                    token_idx += 1
            while token_idx < n_tokens:
                parts.append(tokens[:, token_idx:token_idx + 1, :])
                token_idx += 1
            modified_hidden = torch.cat(parts, dim=1)
        else:  # append (default)
            modified_hidden = torch.cat([hidden_states, tokens], dim=1)

        # Extend mask if provided
        modified_mask: Optional[torch.Tensor] = None
        if attention_mask is not None:
            concept_mask = torch.ones(
                batch_size,
                self.num_tokens,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            if position == "prepend":
                modified_mask = torch.cat([concept_mask, attention_mask], dim=1)
            elif position == "interleave":
                # Interleave weaves tokens in; build a full-length all-ones mask
                modified_mask = torch.ones(
                    batch_size,
                    modified_hidden.shape[1],
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
            else:  # append (default) — preserve original mask values
                modified_mask = torch.cat([attention_mask, concept_mask], dim=1)

        return modified_hidden, modified_mask

    def update_txt_seq_lens(self, txt_seq_lens: list[int]) -> list[int]:
        """
        Update txt_seq_lens to account for the injected concept tokens.

        QwenEmbedRope uses max(txt_seq_lens) to determine how many RoPE
        positions to allocate for the text sequence. Each batch item's
        length must increase by num_tokens.

        Args:
            txt_seq_lens: Original non-padded token counts per batch item.

        Returns:
            Updated list with each length increased by num_tokens.
        """
        return [length + self.num_tokens for length in txt_seq_lens]
