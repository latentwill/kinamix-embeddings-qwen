"""
scripts/batch_inference.py — Batch inference: load models once, swap embeddings.

Generates previews for multiple embeddings × multiple CFG schedules without
reloading the text encoder, transformer, VAE, or scheduler between runs.

No leakage between generations:
  - DiTConceptInjection is stateless (no optimizer, no running stats)
  - concept_tokens are overwritten with torch.no_grad() + .copy_()
  - Text encoder, transformer, VAE, scheduler are frozen and never modified
  - Each preview gets fresh random noise from its own seed

Usage:
    python scripts/batch_inference.py \
        --emb_dir ./output/nextgen/embeddings \
        --output_dir ./output/nextgen/previews \
        --schedules constant linear cosine step \
        --concept_scale 3.0 --scale_high 3.0 --scale_low 1.5
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from embedding import Embedding
from modules.dit_injection import DiTConceptInjection
from modules.model_loader import load_models
from preview import generate_preview, generate_preview_cfg


def build_concept_applier(emb, device):
    """Build a concept_applier closure from an Embedding. Returns (applier, dsci)."""
    token_position = getattr(emb, "token_position", "append")

    num_tokens = emb.tokens.shape[0]
    hidden_dim = emb.tokens.shape[1]
    dsci = DiTConceptInjection(hidden_dim=hidden_dim, num_tokens=num_tokens).to(device)
    with torch.no_grad():
        dsci.concept_tokens.copy_(emb.tokens.to(device))

    def applier(hs, mask):
        return dsci.inject(hs, mask, position=token_position)

    return applier, dsci


def main():
    p = argparse.ArgumentParser(description="Batch inference: load models once, swap embeddings")
    p.add_argument("--emb_dir", required=True, help="Directory containing embedding files")
    p.add_argument("--emb_paths", nargs="*", default=None,
                   help="Specific embedding files (overrides --emb_dir glob)")
    p.add_argument("--output_dir", required=True, help="Output base directory")
    p.add_argument("--schedules", nargs="+", default=["constant"],
                   choices=["constant", "linear", "cosine", "step"],
                   help="CFG scale schedules to test")
    p.add_argument("--concept_scale", type=float, default=3.0)
    p.add_argument("--scale_high", type=float, default=3.0)
    p.add_argument("--scale_low", type=float, default=1.5)
    p.add_argument("--include_standard", action="store_true", default=True,
                   help="Also generate standard (scale=1.0) previews")
    p.add_argument("--no_standard", action="store_true", default=False,
                   help="Skip standard (scale=1.0) previews")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--precision", choices=["fp8", "full"], default="fp8")
    p.add_argument("--dit_dtype", choices=["fp8", "bf16"], default="fp8")
    p.add_argument("--include_checkpoints", action="store_true", default=False,
                   help="Also run inference on checkpoint files (contain _step in name)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Collect embeddings
    if args.emb_paths:
        emb_paths = [Path(p) for p in args.emb_paths]
    else:
        emb_paths = sorted(Path(args.emb_dir).glob("*.safetensors"))
        if not args.include_checkpoints:
            emb_paths = [p for p in emb_paths if "_step" not in p.stem]

    if not emb_paths:
        print(f"No embedding files found in {args.emb_dir}")
        return

    # Build run matrix
    include_standard = args.include_standard and not args.no_standard
    runs = []
    for emb_path in emb_paths:
        name = emb_path.stem
        if include_standard:
            runs.append((emb_path, name, "standard", 1.0, "constant"))
        for sched in args.schedules:
            runs.append((emb_path, name, f"cfg_{sched}", args.concept_scale, sched))
    print(f"{'='*60}")
    print(f"  BATCH INFERENCE")
    print(f"  Embeddings: {len(emb_paths)}")
    print(f"  Schedules:  {args.schedules}")
    print(f"  Standard:   {'yes' if include_standard else 'no'}")
    print(f"  Total runs: {len(runs)}")
    print(f"{'='*60}")

    # Load models ONCE
    print(f"\nLoading models...")
    t0 = time.time()
    args_ns = argparse.Namespace(precision=args.precision, dit_dtype=args.dit_dtype)
    models = load_models(
        args_ns, device,
        components={"text_encoder", "tokenizer", "transformer", "scheduler", "vae"},
    )
    if hasattr(models["transformer"], "enable_gradient_checkpointing"):
        models["transformer"].enable_gradient_checkpointing()
    print(f"  Models loaded in {time.time() - t0:.1f}s")

    # Run inference matrix
    for i, (emb_path, name, mode, scale, sched) in enumerate(runs):
        print(f"\n── [{i+1}/{len(runs)}] {name} / {mode}")
        t1 = time.time()

        emb = Embedding.load(str(emb_path), device=device)
        applier, dsci = build_concept_applier(emb, device)

        out_dir = str(Path(args.output_dir) / f"{name}_{mode}")

        common = dict(
            text_encoder=models["text_encoder"],
            tokenizer=models["tokenizer"],
            transformer=models["transformer"],
            vae=models["vae"],
            scheduler=models["scheduler"],
            concept_applier=applier,
            output_dir=out_dir,
            steps=args.steps,
            width=args.width,
            height=args.height,
            title=f"{name}_{mode}",
        )

        if scale != 1.0:
            generate_preview_cfg(
                **common,
                concept_scale=scale,
                concept_scale_schedule=sched,
                scale_high=args.scale_high,
                scale_low=args.scale_low,
            )
        else:
            generate_preview(**common)

        # Clean up per-run objects (NOT models)
        del dsci, applier, emb
        print(f"  Done in {time.time() - t1:.1f}s")

    # Final cleanup
    del models
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"\n{'='*60}")
    print(f"  BATCH COMPLETE — {len(runs)} preview sets")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
