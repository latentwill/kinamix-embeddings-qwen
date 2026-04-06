"""
preview_cfg.py -- CFG decomposition preview for concept embeddings.

Generates preview grids with independently controllable text_scale and
concept_scale, visualizing how trained concept embeddings affect generation
at different strengths.

Uses decomposed classifier-free guidance:
  v_guided = v_uncond + text_scale*(v_text - v_uncond) + concept_scale*(v_full - v_text)

Usage:
    python preview_cfg.py --emb_path path/to/trained.safetensors \
        --concept_scales 1 2 3 4 5 \
        --text_scale 7.0 \
        --output_dir ./output/cfg_test
"""

import argparse
from pathlib import Path

import torch

from preview import (
    DEFAULT_PREVIEW_PROMPTS,
    DEFAULT_PREVIEW_SEEDS,
    _denoise,
    _denoise_cfg,
    _latents_to_pil,
    create_grid,
    encode_prompt,
)
from modules.dataset_and_loss import _pack_latents
from modules.dit_injection import DiTConceptInjection
from embedding import Embedding


def generate_cfg_preview(
    text_encoder: object,
    tokenizer: object,
    transformer: object,
    vae: object,
    scheduler: object,
    emb_path: str,
    concept_scales: list[float],
    text_scale: float,
    output_dir: str,
    prompts: list[str] | None = None,
    seeds: list[int] | None = None,
    steps: int = 20,
    width: int = 512,
    height: int = 512,
) -> dict:
    """
    Generate CFG decomposition preview grids with varying concept_scale.

    For each concept_scale, generates a prompt x seed grid using _denoise_cfg.
    Also generates a baseline grid using standard _denoise (no concept).

    Args:
        text_encoder: Frozen Qwen2.5-VL text encoder.
        tokenizer: Tokenizer for the text encoder.
        transformer: QwenImageTransformer2DModel (frozen).
        vae: AutoencoderKLQwenImage for decoding latents.
        scheduler: FlowMatchEulerDiscreteScheduler.
        emb_path: Path to trained .safetensors embedding.
        concept_scales: List of concept_scale values to test.
        text_scale: Text guidance scale (held constant across grids).
        output_dir: Directory where grids will be saved.
        prompts: List of text prompts (default: DEFAULT_PREVIEW_PROMPTS).
        seeds: List of random seeds (default: DEFAULT_PREVIEW_SEEDS).
        steps: Number of denoising steps.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Dict with:
            - grid_paths: list[str] (paths to all grid images)
            - baseline_grid_path: str (path to baseline grid)
            - concept_scales: list[float]
            - text_scale: float
            - num_prompts: int
            - num_seeds: int
            - concept_direction_magnitudes: list[float] (L2 norm per prompt)
    """
    if prompts is None:
        prompts = DEFAULT_PREVIEW_PROMPTS
    if seeds is None:
        seeds = DEFAULT_PREVIEW_SEEDS

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = next(
        (p.device for p in transformer.parameters() if hasattr(p, "device")),
        torch.device("cpu"),
    )

    latent_h = height // 8
    latent_w = width // 8

    # Load the embedding and create concept_applier
    emb = Embedding.load(emb_path, device=device)
    num_tokens = emb.tokens.shape[0]
    hidden_dim = emb.tokens.shape[1]
    dsci = DiTConceptInjection(hidden_dim=hidden_dim, num_tokens=num_tokens).to(device)
    with torch.no_grad():
        dsci.concept_tokens.copy_(emb.tokens.to(device))

    def concept_applier(
        hidden_states: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return dsci.inject(hidden_states, mask)

    # Encode all prompts and compute concept direction magnitudes
    concept_direction_magnitudes: list[float] = []
    text_encodings: list[tuple[torch.Tensor, torch.Tensor]] = []
    concept_encodings: list[tuple[torch.Tensor, torch.Tensor]] = []

    for prompt in prompts:
        with torch.no_grad():
            text_hs, text_mask = encode_prompt(
                text_encoder, tokenizer, prompt, device
            )
        concept_hs, concept_mask = concept_applier(
            text_hs.clone(), text_mask.clone()
        )

        # Concept direction magnitude = ||injected_tokens|| L2 norm
        # concept_hs has DSCI tokens concatenated at the end (longer sequence)
        # so we measure the norm of the extra tokens, not the difference
        num_concept_tokens = concept_hs.shape[1] - text_hs.shape[1]
        if num_concept_tokens > 0:
            injected = concept_hs[:, -num_concept_tokens:, :]
            magnitude = torch.norm(injected.float(), p=2).item()
        else:
            magnitude = 0.0
        concept_direction_magnitudes.append(magnitude)

        text_encodings.append((text_hs, text_mask))
        concept_encodings.append((concept_hs, concept_mask))

    # Generate baseline grid (standard _denoise, no concept)
    baseline_images: list[Image.Image] = []
    baseline_labels: list[str] = []

    for prompt_idx, prompt in enumerate(prompts):
        text_hs, text_mask = text_encodings[prompt_idx]
        for seed in seeds:
            generator = torch.Generator(device=device).manual_seed(seed)
            noise = torch.randn(
                1, 16, latent_h, latent_w,
                device=device, dtype=torch.bfloat16,
                generator=generator,
            )
            packed = _pack_latents(noise)

            with torch.no_grad():
                denoised = _denoise(
                    transformer, scheduler,
                    packed, text_hs, text_mask,
                    height, width, steps,
                )
                img = _latents_to_pil(vae, denoised, height, width)

            baseline_images.append(img)
            baseline_labels.append(f"BASE s{seed} | {prompt[:25]}")

    # Save individual baseline images
    singles_dir = output_path / "singles"
    singles_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    for pi, prompt in enumerate(prompts):
        safe_prompt = prompt[:30].replace(" ", "_").replace("/", "_")
        for seed in seeds:
            baseline_images[idx].save(str(singles_dir / f"baseline_p{pi}_{safe_prompt}_s{seed}.png"))
            idx += 1

    cols = len(seeds)
    baseline_grid = create_grid(
        baseline_images, baseline_labels, cols=cols,
        title=f"BASELINE text_scale={text_scale}",
    )
    baseline_grid_path = output_path / "cfg_baseline.png"
    baseline_grid.save(str(baseline_grid_path))

    grid_paths: list[str] = [str(baseline_grid_path)]

    # Generate one grid per concept_scale
    for cs in concept_scales:
        scale_images: list[Image.Image] = []
        scale_labels: list[str] = []

        for prompt_idx, prompt in enumerate(prompts):
            text_hs, text_mask = text_encodings[prompt_idx]
            concept_hs, concept_mask = concept_encodings[prompt_idx]

            for seed in seeds:
                generator = torch.Generator(device=device).manual_seed(seed)
                noise = torch.randn(
                    1, 16, latent_h, latent_w,
                    device=device, dtype=torch.bfloat16,
                    generator=generator,
                )
                packed = _pack_latents(noise)

                with torch.no_grad():
                    denoised = _denoise_cfg(
                        transformer, scheduler,
                        packed, text_hs, text_mask,
                        concept_hs, concept_mask,
                        height, width, steps,
                        text_scale=text_scale,
                        concept_scale=cs,
                    )
                    img = _latents_to_pil(vae, denoised, height, width)

                scale_images.append(img)
                scale_labels.append(
                    f"cs={cs} ts={text_scale} s{seed} | {prompt[:20]}"
                )

        # Save individual CFG images
        idx = 0
        for pi, prompt in enumerate(prompts):
            safe_prompt = prompt[:30].replace(" ", "_").replace("/", "_")
            for seed in seeds:
                scale_images[idx].save(str(singles_dir / f"cfg_cs{cs}_p{pi}_{safe_prompt}_s{seed}.png"))
                idx += 1

        title = f"concept_scale={cs} text_scale={text_scale}"
        scale_grid = create_grid(
            scale_images, scale_labels, cols=cols, title=title,
        )
        grid_filename = f"cfg_concept_scale_{cs}.png"
        grid_path = output_path / grid_filename
        scale_grid.save(str(grid_path))
        grid_paths.append(str(grid_path))

    print(f"CFG preview grids saved to {output_dir}:")
    for gp in grid_paths:
        print(f"  {gp}")
    print(f"  Singles:  {singles_dir}/")
    print(f"Concept direction magnitudes: {concept_direction_magnitudes}")

    return {
        "grid_paths": grid_paths,
        "baseline_grid_path": str(baseline_grid_path),
        "concept_scales": concept_scales,
        "text_scale": text_scale,
        "num_prompts": len(prompts),
        "num_seeds": len(seeds),
        "concept_direction_magnitudes": concept_direction_magnitudes,
    }


def main() -> None:
    """CLI entry point for CFG decomposition preview."""
    parser = argparse.ArgumentParser(
        description="Generate CFG decomposition preview grids for concept embeddings."
    )
    parser.add_argument(
        "--emb_path", type=str, required=True,
        help="Path to trained .safetensors embedding",
    )
    parser.add_argument(
        "--concept_scales", type=float, nargs="+", default=[1, 2, 3, 4, 5],
        help="Concept scale values to test (default: 0 1 3 5 7)",
    )
    parser.add_argument(
        "--text_scale", type=float, default=7.0,
        help="Text guidance scale (default: 7.0)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output/cfg_test",
        help="Output directory for grid images",
    )
    parser.add_argument(
        "--steps", type=int, default=20,
        help="Number of denoising steps (default: 20)",
    )
    parser.add_argument(
        "--width", type=int, default=512,
        help="Image width in pixels (default: 512)",
    )
    parser.add_argument(
        "--height", type=int, default=512,
        help="Image height in pixels (default: 512)",
    )

    parser.add_argument(
        "--precision", choices=["fp8", "full"], default="fp8",
        help="Model precision (default: fp8)",
    )
    parser.add_argument(
        "--dit_dtype", choices=["fp8", "bf16"], default="fp8",
        help="DiT dtype (default: fp8)",
    )

    args = parser.parse_args()

    # Load models (requires GPU environment with diffusers installed)
    from modules.model_loader import load_models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(
        args, device,
        components={"text_encoder", "tokenizer", "transformer", "scheduler", "vae"},
    )

    result = generate_cfg_preview(
        text_encoder=models["text_encoder"],
        tokenizer=models["tokenizer"],
        transformer=models["transformer"],
        vae=models["vae"],
        scheduler=models["scheduler"],
        emb_path=args.emb_path,
        concept_scales=args.concept_scales,
        text_scale=args.text_scale,
        output_dir=args.output_dir,
        steps=args.steps,
        width=args.width,
        height=args.height,
    )

    print("\nResults:")
    print(f"  Concept scales: {result['concept_scales']}")
    print(f"  Text scale: {result['text_scale']}")
    print(f"  Grid count: {len(result['grid_paths'])}")
    for i, mag in enumerate(result["concept_direction_magnitudes"]):
        print(f"  Prompt {i} concept direction magnitude: {mag:.4f}")


if __name__ == "__main__":
    main()
