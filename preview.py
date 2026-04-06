"""
preview.py -- Generate preview images after training.

Generates preview images after training. Uses manual inference to allow
custom concept application between text encoding and denoising.

The training script passes a concept_applier callable:
  concept_applier(hidden_states, attention_mask) -> (modified_hs, modified_mask)

Manual inference avoids the diffusers pipeline, which does its own text
encoding internally and cannot be hooked for post-encoder modifications.
"""

import math
from pathlib import Path
from typing import Callable

import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from modules.dataset_and_loss import _pack_latents, _unpack_latents


# Default preview prompts -- concept-neutral to test generalization
DEFAULT_PREVIEW_PROMPTS = [
    "a mountain landscape",
    "a portrait of a woman",
    "a still life with flowers",
]
DEFAULT_PREVIEW_SEEDS = [69, 2222, 1000]

# Chat template for text encoding (same as modules/model_loader.py)
_PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, quantity, "
    "text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
_PROMPT_TEMPLATE_DROP_TOKENS = 34


def encode_prompt(
    text_encoder: object,
    tokenizer: object,
    prompt: str | list[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode prompt through Qwen2.5-VL text encoder using the chat template.

    Returns (hidden_states, attention_mask) with the system prefix dropped,
    matching the DiT's expected conditioning format.

    Standalone copy to avoid importing model_loader at module level.
    """
    if isinstance(prompt, str):
        prompt = [prompt]

    drop = _PROMPT_TEMPLATE_DROP_TOKENS
    formatted = [_PROMPT_TEMPLATE.format(p) for p in prompt]

    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512 + drop,
    ).to(device)

    outputs = text_encoder(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        output_hidden_states=True,
    )

    hidden_states = outputs.hidden_states[-1][:, drop:]
    attention_mask = inputs.attention_mask[:, drop:]
    return hidden_states, attention_mask


def _load_font(size: int = 14) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a monospace font, falling back to default."""
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def create_grid(
    images: list[Image.Image],
    labels: list[str],
    cols: int,
    title: str | None = None,
) -> Image.Image:
    """
    Arrange images into a labeled grid with an optional title banner.

    Each cell gets a text label drawn above the image. The grid has `cols`
    columns and as many rows as needed. If `title` is provided, a banner
    is drawn across the top of the grid.

    This is a standalone copy of evaluate.create_grid to avoid importing
    evaluate.py (which transitively imports diffusers).
    """
    if len(labels) != len(images):
        raise ValueError(
            f"Number of labels ({len(labels)}) must match number of images ({len(images)})"
        )

    cell_w = images[0].width
    cell_h = images[0].height
    label_h = 30
    title_h = 40 if title else 0
    rows = math.ceil(len(images) / cols)

    grid_w = cols * cell_w
    grid_h = title_h + rows * (cell_h + label_h)
    grid = Image.new("RGB", (grid_w, grid_h), color=(32, 32, 32))
    draw = ImageDraw.Draw(grid)

    font = _load_font(14)
    title_font = _load_font(18)

    # Draw title banner
    if title:
        draw.rectangle([(0, 0), (grid_w, title_h)], fill=(20, 20, 20))
        draw.text((8, 10), title, fill=(100, 200, 255), font=title_font)

    for i, (img, label) in enumerate(zip(images, labels)):
        row = i // cols
        col = i % cols
        x = col * cell_w
        y = title_h + row * (cell_h + label_h)
        draw.text((x + 4, y + 4), label, fill=(255, 255, 255), font=font)
        grid.paste(img, (x, y + label_h))

    return grid


def _denormalize_latents(vae: object, latents: torch.Tensor) -> torch.Tensor:
    """
    Inverse of _normalize_latents from dataset_and_loss.py.

    _normalize_latents does:
      - scaling_factor mode: latents * scaling_factor
      - per-channel mode:    (latents - mean) / std

    So denormalize does:
      - scaling_factor mode: latents / scaling_factor
      - per-channel mode:    latents * std + mean
    """
    if hasattr(vae.config, "scaling_factor") and vae.config.scaling_factor is not None:
        return latents / vae.config.scaling_factor
    # Per-channel latents_mean/latents_std
    mean = torch.tensor(vae.config.latents_mean, device=latents.device, dtype=latents.dtype)
    std = torch.tensor(vae.config.latents_std, device=latents.device, dtype=latents.dtype)
    shape = [1, -1] + [1] * (latents.ndim - 2)
    return latents * std.view(*shape) + mean.view(*shape)


def _denoise(
    transformer: object,
    scheduler: object,
    latents: torch.Tensor,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    height: int,
    width: int,
    num_steps: int,
    show_progress: bool = False,
) -> torch.Tensor:
    """
    Run the denoising loop (flow matching Euler discrete).

    Args:
        transformer: QwenImageTransformer2DModel (frozen).
        scheduler: FlowMatchEulerDiscreteScheduler.
        latents: Packed noisy latents [B, seq, C*4].
        hidden_states: Text encoder output [B, seq, hidden_dim].
        attention_mask: Text attention mask [B, seq].
        height: Image height in pixels.
        width: Image width in pixels.
        num_steps: Number of denoising steps.

    Returns:
        Denoised packed latents [B, seq, C*4].
    """
    # Compute mu for dynamic shifting (required by QwenImage scheduler)
    image_seq_len = latents.shape[1]
    base_seq_len = getattr(scheduler.config, "base_image_seq_len", 256)
    max_seq_len = getattr(scheduler.config, "max_image_seq_len", 4096)
    base_shift = getattr(scheduler.config, "base_shift", 0.5)
    max_shift = getattr(scheduler.config, "max_shift", 1.15)
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b

    scheduler.set_timesteps(num_steps, device=latents.device, mu=mu)

    # img_shapes: post-patchify (frame, height/2, width/2)
    # height/width are pixel dimensions, latent space is /8, patchify is /2
    img_shapes = [(1, height // 16, width // 16)]

    timesteps = scheduler.timesteps
    if show_progress:
        timesteps = tqdm(timesteps, desc="Denoising", leave=False)

    for t in timesteps:
        # Pipeline divides timestep by 1000 before passing to transformer
        timestep = t.expand(latents.shape[0]).to(latents.dtype) / 1000
        noise_pred = transformer(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=hidden_states,
            encoder_hidden_states_mask=attention_mask,
            img_shapes=img_shapes,
            return_dict=False,
        )[0]
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    return latents


def _denoise_cfg(
    transformer: object,
    scheduler: object,
    latents: torch.Tensor,
    text_hidden_states: torch.Tensor,
    text_mask: torch.Tensor,
    concept_hidden_states: torch.Tensor,
    concept_mask: torch.Tensor,
    height: int,
    width: int,
    num_steps: int,
    text_scale: float,
    concept_scale: float,
    show_progress: bool = False,
    concept_scale_schedule: str = "constant",
    scale_high: float = 3.0,
    scale_low: float = 1.5,
    decomposition: str = "standard",
    scale_low_freq: float = 2.0,
    scale_high_freq: float = 3.0,
    fft_cutoff: float = 0.3,
) -> torch.Tensor:
    """
    Run the denoising loop with concept scale guidance.

    Qwen-Image is a flow matching model without classifier-free guidance
    training, so there is no unconditional pass. Instead, we compare
    text-only vs text+concept predictions:

    Per timestep:
      1. v_text = transformer(latents, t, text_hs, text_mask)
      2. v_full = transformer(latents, t, concept_hs, concept_mask)
      3. v_guided = v_text + concept_scale * (v_full - v_text)
      4. latents = scheduler.step(v_guided, t, latents)

    concept_scale=0 → pure text (baseline)
    concept_scale=1 → standard DSCI injection
    concept_scale>1 → amplified concept (extrapolation)

    Args:
        transformer: QwenImageTransformer2DModel (frozen).
        scheduler: FlowMatchEulerDiscreteScheduler.
        latents: Packed noisy latents [B, seq, C*4].
        text_hidden_states: Text-only encoder output [B, seq, hidden_dim].
        text_mask: Text-only attention mask [B, seq].
        concept_hidden_states: Text+concept encoder output [B, seq, hidden_dim].
        concept_mask: Text+concept attention mask [B, seq].
        height: Image height in pixels.
        width: Image width in pixels.
        num_steps: Number of denoising steps.
        text_scale: Guidance scale for text component.
        concept_scale: Guidance scale for concept component (used when
            concept_scale_schedule == "constant").
        concept_scale_schedule: Timestep-dependent schedule for concept scale.
            "constant" uses the fixed concept_scale value (backward compatible).
            "linear", "cosine", "step" interpolate between scale_high and scale_low.
        scale_high: Scale at t=1.0 (high noise) for non-constant schedules.
        scale_low: Scale at t=0.0 (low noise) for non-constant schedules.

    Returns:
        Denoised packed latents [B, seq, C*4].
    """
    # Compute mu for dynamic shifting (same as _denoise)
    image_seq_len = latents.shape[1]
    base_seq_len = getattr(scheduler.config, "base_image_seq_len", 256)
    max_seq_len = getattr(scheduler.config, "max_image_seq_len", 4096)
    base_shift = getattr(scheduler.config, "base_shift", 0.5)
    max_shift = getattr(scheduler.config, "max_shift", 1.15)
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b

    scheduler.set_timesteps(num_steps, device=latents.device, mu=mu)

    # img_shapes: post-patchify (frame, height/2, width/2)
    img_shapes = [(1, height // 16, width // 16)]

    from modules.scale_schedules import get_concept_scale

    timesteps_raw = scheduler.timesteps
    num_timesteps = len(timesteps_raw)

    timesteps = timesteps_raw
    if show_progress:
        timesteps = tqdm(timesteps, desc="Denoising (CFG)", leave=False)

    for i, t in enumerate(timesteps):
        timestep = t.expand(latents.shape[0]).to(latents.dtype) / 1000

        # Determine the concept scale for this timestep.
        # t_normalized: 1.0 at the start (high noise) → 0.0 at the end (clean image).
        if concept_scale_schedule != "constant":
            t_normalized = 1.0 - (i / (num_timesteps - 1)) if num_timesteps > 1 else 0.5
            current_scale = get_concept_scale(
                t_normalized,
                schedule=concept_scale_schedule,
                scale_high=scale_high,
                scale_low=scale_low,
            )
        else:
            current_scale = concept_scale

        # 1. Text-only pass (baseline)
        v_text = transformer(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=text_hidden_states,
            encoder_hidden_states_mask=text_mask,
            img_shapes=img_shapes,
            return_dict=False,
        )[0]

        # 2. Full (text + concept) pass
        v_full = transformer(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=concept_hidden_states,
            encoder_hidden_states_mask=concept_mask,
            img_shapes=img_shapes,
            return_dict=False,
        )[0]

        # 3. Concept guidance
        v_guided = v_text + current_scale * (v_full - v_text)

        # 4. Scheduler step
        latents = scheduler.step(v_guided, t, latents, return_dict=False)[0]

    return latents


def _latents_to_pil(vae: object, latents: torch.Tensor, height: int, width: int) -> Image.Image:
    """
    Decode packed latents to a PIL image.

    Steps:
      1. Unpack from patch sequence to spatial: (B, seq, C*4) -> (B, C, H/8, W/8)
      2. Denormalize (inverse of training normalization)
      3. Add frame dimension for 3D VAE: (B, C, H/8, W/8) -> (B, C, 1, H/8, W/8)
      4. VAE decode -> (B, 3, 1, H, W)
      5. Remove frame dim, clamp, scale to [0, 255], convert to PIL
    """
    latent_h = height // 8
    latent_w = width // 8

    # Unpack: (B, (H/2)*(W/2), C*4) -> (B, C, H, W) in latent space
    spatial = _unpack_latents(latents, latent_h, latent_w)

    # Denormalize
    spatial = _denormalize_latents(vae, spatial)

    # Add frame dimension for 3D VAE
    spatial = spatial.unsqueeze(2)  # (B, C, 1, H/8, W/8)

    # Decode
    decoded = vae.decode(spatial).sample  # (B, 3, 1, H, W)

    # Remove frame dim, convert to PIL
    image_tensor = decoded.squeeze(2)  # (B, 3, H, W)
    image_tensor = image_tensor.float().clamp(-1, 1)
    image_tensor = (image_tensor + 1) / 2 * 255
    image_array = image_tensor[0].permute(1, 2, 0).byte().cpu().numpy()
    return Image.fromarray(image_array)


def generate_preview(
    text_encoder: object,
    tokenizer: object,
    transformer: object,
    vae: object,
    scheduler: object,
    concept_applier: Callable,
    output_dir: str,
    prompts: list[str] | None = None,
    seeds: list[int] | None = None,
    steps: int = 30,
    width: int = 512,
    height: int = 512,
    title: str | None = None,
    show_progress: bool = True,
    concept_scale: float = 3.5,
    concept_scale_schedule: str = "constant",
    scale_high: float = 3.0,
    scale_low: float = 1.5,
) -> dict:
    """
    Generate preview grids: one with concept applied, one baseline (no concept).

    By default uses CFG decomposition (concept_scale=3.5) which produces
    visible concept signal. Set concept_scale=1.0 for single-pass injection.

    Args:
        text_encoder: Frozen Qwen2.5-VL text encoder.
        tokenizer: Tokenizer for the text encoder.
        transformer: QwenImageTransformer2DModel (frozen).
        vae: AutoencoderKLQwenImage for decoding latents.
        scheduler: FlowMatchEulerDiscreteScheduler.
        concept_applier: Callable (hidden_states, mask) -> (hidden_states, mask).
            Concept injection callable:
            - DSCI: lambda hs, m: dsci.inject(hs, m)
        output_dir: Directory where grids will be saved.
        prompts: List of text prompts (default: DEFAULT_PREVIEW_PROMPTS).
        seeds: List of random seeds (default: DEFAULT_PREVIEW_SEEDS).
        steps: Number of denoising steps (default 30).
        width: Image width in pixels.
        height: Image height in pixels.
        title: Optional title drawn as a banner on the grid image.
        concept_scale: CFG guidance scale (default 3.5). When != 1.0, uses
            two-pass CFG decomposition for concept images. 1.0 = single-pass.
        concept_scale_schedule: Timestep-dependent schedule ("constant",
            "linear", "cosine", "step"). Default "constant".
        scale_high: Concept scale at t=1.0 for non-constant schedules.
        scale_low: Concept scale at t=0.0 for non-constant schedules.

    Returns:
        Dict with:
            - concept_grid_path: str (path to concept preview grid)
            - baseline_grid_path: str (path to baseline preview grid)
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

    # Latent dimensions
    latent_h = height // 8
    latent_w = width // 8

    concept_images: list[Image.Image] = []
    concept_labels: list[str] = []
    baseline_images: list[Image.Image] = []
    baseline_labels: list[str] = []

    pairs = [(p, s) for p in prompts for s in seeds]
    outer = tqdm(pairs, desc="Generating previews", disable=not show_progress)

    use_cfg = concept_scale != 1.0

    for prompt, seed in outer:
        generator = torch.Generator(device=device).manual_seed(seed)

        # Encode prompt (shared between concept and baseline)
        with torch.no_grad():
            hidden_states, attention_mask = encode_prompt(
                text_encoder, tokenizer, prompt, device
            )

        # -- Concept image --
        concept_hs, concept_mask = concept_applier(
            hidden_states.clone(), attention_mask.clone()
        )

        # Random latents (use same seed for concept and baseline)
        noise_concept = torch.randn(
            1, 16, latent_h, latent_w,
            device=device, dtype=torch.bfloat16,
            generator=generator,
        )
        packed_concept = _pack_latents(noise_concept)

        with torch.no_grad():
            if use_cfg:
                denoised_concept = _denoise_cfg(
                    transformer, scheduler,
                    packed_concept,
                    hidden_states, attention_mask,
                    concept_hs, concept_mask,
                    height, width, steps,
                    text_scale=1.0,
                    concept_scale=concept_scale,
                    show_progress=show_progress,
                    concept_scale_schedule=concept_scale_schedule,
                    scale_high=scale_high,
                    scale_low=scale_low,
                )
            else:
                denoised_concept = _denoise(
                    transformer, scheduler,
                    packed_concept, concept_hs, concept_mask,
                    height, width, steps,
                    show_progress=show_progress,
                )
            concept_img = _latents_to_pil(vae, denoised_concept, height, width)

        concept_images.append(concept_img)
        scale_label = f"cfg×{concept_scale} " if use_cfg else ""
        concept_labels.append(f"{scale_label}s{seed} | {prompt[:30]}")

        # -- Baseline image (same seed, no concept applied) --
        generator_baseline = torch.Generator(device=device).manual_seed(seed)
        noise_baseline = torch.randn(
            1, 16, latent_h, latent_w,
            device=device, dtype=torch.bfloat16,
            generator=generator_baseline,
        )
        packed_baseline = _pack_latents(noise_baseline)

        with torch.no_grad():
            denoised_baseline = _denoise(
                transformer, scheduler,
                packed_baseline, hidden_states, attention_mask,
                height, width, steps,
                show_progress=show_progress,
            )
            baseline_img = _latents_to_pil(vae, denoised_baseline, height, width)

        baseline_images.append(baseline_img)
        baseline_labels.append(f"BASE s{seed} | {prompt[:25]}")

    # Save individual images
    singles_dir = output_path / "singles"
    singles_dir.mkdir(parents=True, exist_ok=True)
    individual_paths: list[str] = []

    idx = 0
    for pi, prompt in enumerate(prompts):
        for seed in seeds:
            safe_prompt = prompt[:30].replace(" ", "_").replace("/", "_")
            concept_images[idx].save(str(singles_dir / f"concept_p{pi}_{safe_prompt}_s{seed}.png"))
            baseline_images[idx].save(str(singles_dir / f"baseline_p{pi}_{safe_prompt}_s{seed}.png"))
            individual_paths.append(str(singles_dir / f"concept_p{pi}_{safe_prompt}_s{seed}.png"))
            individual_paths.append(str(singles_dir / f"baseline_p{pi}_{safe_prompt}_s{seed}.png"))
            idx += 1

    # Create grids
    cols = len(seeds)
    concept_grid = create_grid(
        concept_images, concept_labels, cols=cols, title=title,
    )
    baseline_grid = create_grid(
        baseline_images, baseline_labels, cols=cols,
        title=f"BASELINE — {title}" if title else "BASELINE",
    )

    # Use title in filename if provided, so grids from different runs never collide
    if title:
        safe_name = title.replace("/", "_").replace(" ", "_")
        concept_grid_path = output_path / f"{safe_name}_concept.png"
        baseline_grid_path = output_path / f"{safe_name}_baseline.png"
    else:
        concept_grid_path = output_path / "preview_grid.png"
        baseline_grid_path = output_path / "baseline_grid.png"

    concept_grid.save(str(concept_grid_path))
    baseline_grid.save(str(baseline_grid_path))

    print(f"Preview grids saved:")
    print(f"  Concept:  {concept_grid_path}")
    print(f"  Baseline: {baseline_grid_path}")
    print(f"  Singles:  {singles_dir}/ ({len(individual_paths)} images)")

    return {
        "concept_grid_path": str(concept_grid_path),
        "baseline_grid_path": str(baseline_grid_path),
        "individual_paths": individual_paths,
    }


def generate_preview_cfg(
    text_encoder: object,
    tokenizer: object,
    transformer: object,
    vae: object,
    scheduler: object,
    concept_applier: Callable,
    output_dir: str,
    concept_scale: float = 3.5,
    prompts: list[str] | None = None,
    seeds: list[int] | None = None,
    steps: int = 30,
    width: int = 512,
    height: int = 512,
    title: str | None = None,
    show_progress: bool = True,
    concept_scale_schedule: str = "constant",
    scale_high: float = 3.0,
    scale_low: float = 1.5,
    decomposition: str = "standard",
    scale_low_freq: float = 2.0,
    scale_high_freq: float = 3.0,
    fft_cutoff: float = 0.3,
) -> dict:
    """Generate previews using CFG decomposition (text-only vs text+concept).

    Each image uses two transformer passes per step:
        v_guided = v_text + concept_scale * (v_full - v_text)

    concept_scale=1 → standard injection.
    concept_scale>1 → amplified concept signal (extrapolation).

    Args:
        concept_scale: Amplification of the concept guidance vector (default 3.5).
            Used when concept_scale_schedule == "constant".
        concept_scale_schedule: Timestep-dependent schedule ("constant", "linear",
            "cosine", "step"). "constant" uses concept_scale (backward compatible).
        scale_high: Concept scale at t=1.0 (high noise) for non-constant schedules.
        scale_low: Concept scale at t=0.0 (low noise) for non-constant schedules.
        All other args: same as generate_preview().

    Returns:
        Dict with cfg_grid_path and individual_paths.
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

    cfg_images: list[Image.Image] = []
    cfg_labels: list[str] = []
    individual_paths: list[str] = []
    singles_dir = output_path / "singles"
    singles_dir.mkdir(parents=True, exist_ok=True)

    pairs = [(p, s) for p in prompts for s in seeds]
    outer = tqdm(pairs, desc="Generating CFG previews", disable=not show_progress)

    idx = 0
    for prompt, seed in outer:
        with torch.no_grad():
            text_hs, text_mask = encode_prompt(text_encoder, tokenizer, prompt, device)
        concept_hs, concept_mask = concept_applier(text_hs.clone(), text_mask.clone())

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
                packed, text_hs, text_mask, concept_hs, concept_mask,
                height, width, steps,
                text_scale=1.0, concept_scale=concept_scale,
                show_progress=show_progress,
                concept_scale_schedule=concept_scale_schedule,
                scale_high=scale_high,
                scale_low=scale_low,
                decomposition=decomposition,
                scale_low_freq=scale_low_freq,
                scale_high_freq=scale_high_freq,
                fft_cutoff=fft_cutoff,
            )
            img = _latents_to_pil(vae, denoised, height, width)

        scale_label = f"sched={concept_scale_schedule}" if concept_scale_schedule != "constant" else f"cfg×{concept_scale}"
        cfg_images.append(img)
        cfg_labels.append(f"{scale_label} s{seed} | {prompt[:25]}")

        safe_prompt = prompt[:30].replace(" ", "_").replace("/", "_")
        path = str(singles_dir / f"cfg_p{idx // len(seeds)}_{safe_prompt}_s{seed}.png")
        img.save(path)
        individual_paths.append(path)
        idx += 1

    cols = len(seeds)
    grid = create_grid(cfg_images, cfg_labels, cols=cols, title=title)
    safe_name = (title or "cfg").replace("/", "_").replace(" ", "_")
    grid_path = output_path / f"{safe_name}_cfg_scale{concept_scale}.png"
    grid.save(str(grid_path))

    print(f"CFG preview grid saved: {grid_path}")
    return {"cfg_grid_path": str(grid_path), "individual_paths": individual_paths}


def generate_preview_cfg_noise(
    text_encoder: object,
    tokenizer: object,
    transformer: object,
    vae: object,
    scheduler: object,
    concept_applier: Callable,
    noise_prior: torch.Tensor,
    output_dir: str,
    concept_scale: float = 3.5,
    noise_blend: float = 0.3,
    prompts: list[str] | None = None,
    seeds: list[int] | None = None,
    steps: int = 30,
    width: int = 512,
    height: int = 512,
    title: str | None = None,
    show_progress: bool = True,
    concept_scale_schedule: str = "constant",
    scale_high: float = 3.0,
    scale_low: float = 1.5,
    decomposition: str = "standard",
    scale_low_freq: float = 2.0,
    scale_high_freq: float = 3.0,
    fft_cutoff: float = 0.3,
) -> dict:
    """Generate CFG-decomposed previews with noise prior blending.

    Starting noise = (1 - noise_blend) * random + noise_blend * prior_centroid.
    This grounds generation closer to the training image distribution,
    reducing seed-dependent quality variation.

    Args:
        noise_prior: Centroid noise tensor [1, 16, H/8, W/8] from
                     flow_inversion.compute_noise_prior().
        noise_blend: Weight of the prior centroid (default 0.3 = 30% prior).
        concept_scale: CFG amplification scale (default 2.0). Used when
            concept_scale_schedule == "constant".
        concept_scale_schedule: Timestep-dependent schedule ("constant", "linear",
            "cosine", "step"). "constant" uses concept_scale (backward compatible).
        scale_high: Concept scale at t=1.0 (high noise) for non-constant schedules.
        scale_low: Concept scale at t=0.0 (low noise) for non-constant schedules.
        All other args: same as generate_preview().

    Returns:
        Dict with cfg_noise_grid_path and individual_paths.
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

    prior = noise_prior.to(device=device, dtype=torch.bfloat16)

    cfg_noise_images: list[Image.Image] = []
    cfg_noise_labels: list[str] = []
    individual_paths: list[str] = []
    singles_dir = output_path / "singles"
    singles_dir.mkdir(parents=True, exist_ok=True)

    pairs = [(p, s) for p in prompts for s in seeds]
    outer = tqdm(pairs, desc="Generating CFG+noise previews", disable=not show_progress)

    idx = 0
    for prompt, seed in outer:
        with torch.no_grad():
            text_hs, text_mask = encode_prompt(text_encoder, tokenizer, prompt, device)
        concept_hs, concept_mask = concept_applier(text_hs.clone(), text_mask.clone())

        generator = torch.Generator(device=device).manual_seed(seed)
        rand_noise = torch.randn(
            1, 16, latent_h, latent_w,
            device=device, dtype=torch.bfloat16,
            generator=generator,
        )
        # Blend: most of random noise + a push from the training distribution
        blended = (1.0 - noise_blend) * rand_noise + noise_blend * prior
        packed = _pack_latents(blended)

        with torch.no_grad():
            denoised = _denoise_cfg(
                transformer, scheduler,
                packed, text_hs, text_mask, concept_hs, concept_mask,
                height, width, steps,
                text_scale=1.0, concept_scale=concept_scale,
                show_progress=show_progress,
                concept_scale_schedule=concept_scale_schedule,
                scale_high=scale_high,
                scale_low=scale_low,
                decomposition=decomposition,
                scale_low_freq=scale_low_freq,
                scale_high_freq=scale_high_freq,
                fft_cutoff=fft_cutoff,
            )
            img = _latents_to_pil(vae, denoised, height, width)

        scale_label = f"sched={concept_scale_schedule}" if concept_scale_schedule != "constant" else f"cfg×{concept_scale}"
        cfg_noise_images.append(img)
        cfg_noise_labels.append(
            f"{scale_label}+prior s{seed} | {prompt[:20]}"
        )

        safe_prompt = prompt[:30].replace(" ", "_").replace("/", "_")
        path = str(singles_dir / f"cfg_noise_p{idx // len(seeds)}_{safe_prompt}_s{seed}.png")
        img.save(path)
        individual_paths.append(path)
        idx += 1

    cols = len(seeds)
    grid = create_grid(cfg_noise_images, cfg_noise_labels, cols=cols, title=title)
    safe_name = (title or "cfg_noise").replace("/", "_").replace(" ", "_")
    grid_path = output_path / f"{safe_name}_cfg_noise_blend{noise_blend}.png"
    grid.save(str(grid_path))

    print(f"CFG+noise prior grid saved: {grid_path}")
    return {"cfg_noise_grid_path": str(grid_path), "individual_paths": individual_paths}
