"""
embedding.py
════════════
Embedding handler for DSCI concept tokens.

  embedding = Embedding.load("my_concept.safetensors")
  # Use embedding.tokens directly for DiT injection
  print(embedding.info())

Saved as safetensors with metadata (method, hidden_dim, token_position).
"""

import json
import torch
from pathlib import Path

from safetensors.torch import save_file as _st_save
from safetensors import safe_open as _safe_open


FILE_EXT = ".safetensors"


class Embedding:
    """
    A trained DSCI embedding — N concept tokens in the DiT conditioning space.

    Fields saved to disk (via safetensors metadata):
      method         : "dsci"
      hidden_dim     : int
      token_position : "append" | "prepend" | "interleave"

    Tensor keys:
      tokens : [N, hidden_dim]
    """

    def __init__(self, hidden_dim: int = 3584):
        self.method = "dsci"
        self.hidden_dim = hidden_dim
        self.tokens = None           # (N, hidden_dim) tensor
        self.token_position = "append"

    @classmethod
    def from_dsci(cls, tokens: torch.Tensor, token_position: str = "append") -> "Embedding":
        """Create from DiT-side concept injection tokens."""
        emb = cls(tokens.shape[-1])
        emb.tokens = tokens.detach().cpu()
        emb.token_position = token_position
        return emb

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        path = Path(path)
        if path.suffix != FILE_EXT:
            path = path.with_suffix(FILE_EXT)
        path.parent.mkdir(parents=True, exist_ok=True)

        tensors = {"tokens": self.tokens}
        metadata = {
            "method": self.method,
            "hidden_dim": str(self.hidden_dim),
            "token_position": self.token_position,
        }

        _st_save(tensors, str(path), metadata=metadata)
        size_kb = path.stat().st_size / 1024
        print(f"Saved embedding → {path} ({size_kb:.1f} KB)")
        return path

    @classmethod
    def load(cls, path: str, device="cpu") -> "Embedding":
        path = Path(path)

        if path.suffix == ".pt":
            data = torch.load(str(path), map_location=device, weights_only=True)
            emb = cls(data["hidden_dim"])
            emb.tokens = data["tokens"]
            emb.token_position = data.get("token_position", "append")
            return emb

        with _safe_open(str(path), framework="pt", device=device) as f:
            metadata = f.metadata()
            tensors = {k: f.get_tensor(k) for k in f.keys()}

        emb = cls(int(metadata["hidden_dim"]))
        emb.tokens = tensors["tokens"]
        emb.token_position = metadata.get("token_position", "append")
        return emb

    # ── Utility ───────────────────────────────────────────────────────────────

    def info(self) -> str:
        num_tokens = self.tokens.shape[0]
        total = self.tokens.numel()
        return "\n".join([
            f"method         : {self.method}",
            f"hidden_dim     : {self.hidden_dim}",
            f"tokens         : {num_tokens}",
            f"token_position : {self.token_position}",
            f"params         : {total} ({total * 4 / 1024:.1f} KB)",
            f"norm           : {self.tokens.norm():.4f}",
        ])

    def __repr__(self):
        n = self.tokens.shape[0] if self.tokens is not None else 0
        return f"Embedding(dsci, {n} tokens, dim={self.hidden_dim})"
