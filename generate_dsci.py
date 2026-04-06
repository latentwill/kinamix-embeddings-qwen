"""
generate_dsci.py — Generate preview images from a trained DSCI embedding.

Usage:
    # Default generation (CFG 3.5, 30 steps)
    python generate_dsci.py --emb_path ./output/my_concept.safetensors

    # Single-pass generation (no CFG)
    python generate_dsci.py --emb_path ./output/my_concept.safetensors --concept_scale 1.0

    # CFG + noise prior blending
    python generate_dsci.py --emb_path ./output/my_concept.safetensors \
        --noise_prior ./output/noise_prior.pt
"""

import argparse
import gc
import os
import torch

from embedding import Embedding
from modules.dit_injection import DiTConceptInjection
from modules.model_loader import load_models
from preview import generate_preview, generate_preview_cfg, generate_preview_cfg_noise


def generate_dsci(
    emb_path: str,
    output_dir: str = "./output/dsci",
    prompts: list[str] | None = None,
    seeds: list[int] | None = None,
    steps: int = 30,
    width: int = 512,
    height: int = 512,
    precision: str = "fp8",
    dit_dtype: str = "fp8",
    concept_scale: float = 3.5,
    noise_prior_path: str | None = None,
    noise_blend: float = 0.3,
    scale_schedule: str = "constant",
    scale_high: float = 3.0,
    scale_low: float = 1.5,
) -> dict:
    """Load a DSCI embedding and generate preview grids.

    Args:
        concept_scale: CFG guidance scale (default 3.5). When != 1.0, uses
                       CFG decomposition (two transformer passes per step).
                       1.0 = single-pass standard generation.
        noise_prior_path: Path to a noise prior .pt file. When provided
                          alongside concept_scale > 1.0, blends the prior
                          centroid into starting noise.
        noise_blend: Weight of prior centroid in blended noise (default 0.3).
        scale_schedule: Timestep-dependent schedule for concept scale.
            "constant" uses concept_scale (backward compatible).
            "linear", "cosine", "step" interpolate between scale_high and scale_low.
        scale_high: Concept scale at t=1.0 (high noise) for non-constant schedules.
        scale_low: Concept scale at t=0.0 (low noise) for non-constant schedules.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb = Embedding.load(emb_path, device=device)

    if emb.method != "dsci":
        raise ValueError(f"Expected dsci embedding, got method={emb.method}")

    token_position = getattr(emb, "token_position", "append")
    num_tokens = emb.tokens.shape[0]
    hidden_dim = emb.tokens.shape[1]
    dsci = DiTConceptInjection(hidden_dim=hidden_dim, num_tokens=num_tokens).to(device)
    with torch.no_grad():
        dsci.concept_tokens.copy_(emb.tokens.to(device))

        def apply_dsci(hs: torch.Tensor, mask: torch.Tensor) -> tuple:
            return dsci.inject(hs, mask, position=token_position)

    args_ns = argparse.Namespace(precision=precision, dit_dtype=dit_dtype)
    models = load_models(
        args_ns, device,
        components={"text_encoder", "tokenizer", "transformer", "scheduler", "vae"},
    )

    if hasattr(models["transformer"], "enable_gradient_checkpointing"):
        models["transformer"].enable_gradient_checkpointing()

    common = dict(
        text_encoder=models["text_encoder"],
        tokenizer=models["tokenizer"],
        transformer=models["transformer"],
        vae=models["vae"],
        scheduler=models["scheduler"],
        concept_applier=apply_dsci,
        output_dir=output_dir,
        prompts=prompts,
        seeds=seeds,
        steps=steps,
        width=width,
        height=height,
        concept_scale=concept_scale,
    )

    if concept_scale != 1.0 and noise_prior_path:
        noise_prior = torch.load(noise_prior_path, map_location=device)
        result = generate_preview_cfg_noise(
            **common,
            noise_prior=noise_prior,
            concept_scale=concept_scale,
            noise_blend=noise_blend,
            concept_scale_schedule=scale_schedule,
            scale_high=scale_high,
            scale_low=scale_low,
        )
    elif concept_scale != 1.0:
        result = generate_preview_cfg(
            **common,
            concept_scale=concept_scale,
            concept_scale_schedule=scale_schedule,
            scale_high=scale_high,
            scale_low=scale_low,
        )
    else:
        result = generate_preview(**common)

    del models
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def parse_args():
    p = argparse.ArgumentParser(description="Generate images from a DSCI embedding")
    p.add_argument("--emb_path", required=True, help="Path to trained .safetensors embedding")
    p.add_argument("--output_dir", default="./output", help="Output directory")
    p.add_argument("--prompt", nargs="+", default=None, help="Custom prompts")
    p.add_argument("--seed", type=int, nargs="+", default=None, help="Random seeds")
    p.add_argument("--steps", type=int, default=30, help="Denoising steps")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--precision", choices=["fp8", "full"], default="fp8")
    p.add_argument("--dit_dtype", choices=["fp8", "bf16"], default="fp8")
    p.add_argument("--concept_scale", type=float, default=3.5,
                   help="CFG guidance scale (default: 3.5). Set to 1.0 for single-pass")
    p.add_argument("--noise_prior", default=None,
                   help="Path to noise prior .pt file")
    p.add_argument("--noise_blend", type=float, default=0.3,
                   help="Noise prior blend weight (default: 0.3)")
    p.add_argument("--scale_schedule", choices=["constant", "linear", "cosine", "step"],
                   default="constant",
                   help="Concept scale schedule for CFG decomposition (default: constant)")
    p.add_argument("--scale_high", type=float, default=3.0,
                   help="Concept scale at high-noise timesteps for non-constant schedules")
    p.add_argument("--scale_low", type=float, default=1.5,
                   help="Concept scale at low-noise timesteps for non-constant schedules")
    return p.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args = parse_args()

    result = generate_dsci(
        emb_path=args.emb_path,
        output_dir=args.output_dir,
        prompts=args.prompt,
        seeds=args.seed,
        steps=args.steps,
        width=args.width,
        height=args.height,
        precision=args.precision,
        dit_dtype=args.dit_dtype,
        concept_scale=args.concept_scale,
        noise_prior_path=args.noise_prior,
        noise_blend=args.noise_blend,
        scale_schedule=args.scale_schedule,
        scale_high=args.scale_high,
        scale_low=args.scale_low,
    )
    print(f"\nDone. Output saved to: {args.output_dir}")
