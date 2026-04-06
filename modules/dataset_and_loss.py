"""
modules/dataset_and_loss.py — Dataset and loss functions for concept training.

ConceptDataset      — Loads training images with prompt templates.
CachedLatentDataset — Pre-encodes images to latents for memory-constrained training.
                      Enables VAE deletion from GPU before loading text encoder + DiT.
flow_matching_loss  — Flow matching loss for QwenImage / FlowMatchEulerDiscreteScheduler.
"""

# ──────────────────────────────────────────────────────────────────────────────
# dataset.py
# ──────────────────────────────────────────────────────────────────────────────

from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


def _normalize_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    """Normalize VAE latents using scaling_factor or per-channel stats."""
    if hasattr(vae.config, "scaling_factor") and vae.config.scaling_factor is not None:
        return latents * vae.config.scaling_factor
    # QwenImage VAE uses per-channel latents_mean/latents_std
    mean = torch.tensor(vae.config.latents_mean, device=latents.device, dtype=latents.dtype)
    std = torch.tensor(vae.config.latents_std, device=latents.device, dtype=latents.dtype)
    # Reshape for broadcasting: (C,) → (1, C, 1, 1) or (1, C, 1, 1, 1)
    shape = [1, -1] + [1] * (latents.ndim - 2)
    return (latents - mean.view(*shape)) / std.view(*shape)


# ── Default templates: style-neutral so the concept token carries all meaning ──
DEFAULT_TEMPLATES = [
    "artwork by {name}",
    "art by {name}",
    "a piece by {name}",
    "an image by {name}",
    "a work by {name}",
    "an illustration by {name}",
    "a creation by {name}",
    "a composition by {name}",
]

DEFAULT_CAPTION_TEMPLATES = [
    "{caption}, artwork by {name}",
    "{caption}, art by {name}",
    "{caption}, a piece by {name}",
    "{caption}, an image by {name}",
    "{caption}, a work by {name}",
    "{caption}, an illustration by {name}",
    "{caption}, a creation by {name}",
    "{caption}, a composition by {name}",
]

DEFAULT_FILEWORDS_TEMPLATES = [
    "artwork of {filewords} by {name}",
    "art of {filewords} by {name}",
    "a piece depicting {filewords} by {name}",
    "an image of {filewords} by {name}",
    "a work showing {filewords} by {name}",
    "an illustration of {filewords} by {name}",
    "a creation of {filewords} by {name}",
    "a composition of {filewords} by {name}",
]


# ── Example templates for --prompt_file (not used by default) ─────────────
# See examples/prompts_style.txt and examples/prompts_object.txt

STYLE_TEMPLATES = [
    "a painting, art by {name}",
    "a rendering, art by {name}",
    "a cropped painting, art by {name}",
    "the painting, art by {name}",
    "a clean painting, art by {name}",
    "a dirty painting, art by {name}",
    "a dark painting, art by {name}",
    "a picture, art by {name}",
    "a cool painting, art by {name}",
    "a close-up painting, art by {name}",
    "a bright painting, art by {name}",
    "a good painting, art by {name}",
    "a rendition, art by {name}",
    "a nice painting, art by {name}",
    "a small painting, art by {name}",
    "a weird painting, art by {name}",
    "a large painting, art by {name}",
]

OBJECT_TEMPLATES = [
    "a photo of a {name}",
    "a rendering of a {name}",
    "a cropped photo of the {name}",
    "the photo of a {name}",
    "a photo of a clean {name}",
    "a photo of a dirty {name}",
    "a dark photo of the {name}",
    "a photo of my {name}",
    "a photo of the cool {name}",
    "a close-up photo of a {name}",
    "a bright photo of the {name}",
    "a cropped photo of a {name}",
    "a photo of the {name}",
    "a good photo of the {name}",
    "a photo of one {name}",
    "a close-up photo of the {name}",
    "a rendition of the {name}",
    "a photo of the clean {name}",
    "a rendition of a {name}",
    "a photo of a nice {name}",
    "a good photo of a {name}",
    "a photo of the nice {name}",
    "a photo of the small {name}",
    "a photo of the weird {name}",
    "a photo of the large {name}",
    "a photo of a cool {name}",
    "a photo of a small {name}",
]


def _load_caption(image_path: Path) -> str | None:
    """Load caption from a .txt file with the same stem as the image."""
    txt_path = image_path.with_suffix(".txt")
    if txt_path.exists():
        return txt_path.read_text().strip()
    # Also check parent dir when loading from .latent_cache/
    parent_txt = image_path.parent.parent / f"{image_path.stem}.txt"
    if parent_txt.exists():
        return parent_txt.read_text().strip()
    return None


def _filewords(path: Path) -> str:
    """Extract caption words from a filename. 'mountain_sunset.jpg' → 'mountain sunset'."""
    import re
    stem = path.stem
    # Replace underscores, hyphens, camelCase boundaries with spaces
    stem = re.sub(r'([a-z])([A-Z])', r'\1 \2', stem)
    stem = stem.replace('_', ' ').replace('-', ' ')
    # Remove leading numbers/indices (e.g. "01 mountain" → "mountain")
    stem = re.sub(r'^\d+\s*', '', stem)
    return stem.strip().lower()


class ConceptDataset(Dataset):
    """
    Loads a folder of concept images and pairs them with a templated prompt.

    Templates use {name} for the concept token and optionally {filewords}
    for caption words derived from the image filename.

    Examples:
        "a painting, art by {name}"              → "a painting, art by rftemb"
        "a painting of {filewords}, art by {name}" → "a painting of mountain sunset, art by rftemb"

    Recommended: 5–20 images for a style concept, more for a subject concept.
    """

    def __init__(
        self,
        image_dir: str,
        concept_token: str,
        image_size: int = 512,
        prompt_templates: list[str] | None = None,
        use_captions: bool = False,
    ):
        self.image_dir = Path(image_dir)
        self.concept_token = concept_token
        self.image_paths = sorted(
            [p for p in self.image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
        )
        if not self.image_paths:
            raise ValueError(f"No images found in {image_dir}")

        self.use_captions = use_captions
        self.captions: dict[str, str] = {}
        if use_captions:
            for img_path in self.image_paths:
                caption = _load_caption(img_path)
                if caption:
                    self.captions[str(img_path)] = caption
            caption_count = len(self.captions)
            fallback_count = len(self.image_paths) - caption_count
            print(f"Captions: {caption_count} loaded, {fallback_count} using filename fallback")

        if use_captions and prompt_templates is None:
            self.templates = DEFAULT_CAPTION_TEMPLATES
        else:
            self.templates = prompt_templates or DEFAULT_TEMPLATES
        self.transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.LANCZOS),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),  # → [-1, 1]
        ])

    def __len__(self):
        return len(self.image_paths) * len(self.templates)

    def __getitem__(self, idx):
        img_idx = idx % len(self.image_paths)
        tmpl_idx = idx // len(self.image_paths)

        image = Image.open(self.image_paths[img_idx]).convert("RGB")
        image = self.transform(image)

        img_path = self.image_paths[img_idx]
        caption = self.captions.get(str(img_path)) if self.use_captions else None
        if caption is not None:
            template = self.templates[tmpl_idx % len(self.templates)]
            prompt = template.format(name=self.concept_token, caption=caption,
                                     filewords=_filewords(img_path))
        else:
            fallback = DEFAULT_TEMPLATES[tmpl_idx % len(DEFAULT_TEMPLATES)]
            prompt = fallback.format(name=self.concept_token,
                                     filewords=_filewords(img_path))

        return {"pixel_values": image, "prompt": prompt}


# ──────────────────────────────────────────────────────────────────────────────
# cached_latent_dataset.py — Pre-encoded latents for memory-constrained training
# ──────────────────────────────────────────────────────────────────────────────


class CachedLatentDataset(Dataset):
    """
    Loads pre-encoded latent tensors from disk instead of raw images.

    Used for memory-constrained training (e.g. 24GB VRAM):
      1. Load VAE, encode all images to latents, save to {image_dir}/.latent_cache/
      2. Delete VAE from GPU
      3. Load text encoder + DiT
      4. Train using this dataset (loads cached .pt latent files)

    Each cached file contains the latent tensor for one image.
    The cached .pt filename stems match the original image stems,
    so filewords can be extracted from them.
    """

    def __init__(
        self,
        cache_dir: str,
        concept_token: str,
        prompt_templates: list[str] | None = None,
        use_captions: bool = False,
        image_dir: str | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.concept_token = concept_token

        self.latent_paths = sorted(self.cache_dir.glob("*.pt"))
        if not self.latent_paths:
            raise ValueError(f"No cached latent files found in {cache_dir}")

        self.use_captions = use_captions
        self.captions: dict[str, str] = {}
        if use_captions:
            # Determine where to look for .txt files
            caption_search_dir = Path(image_dir) if image_dir else self.cache_dir.parent
            for lat_path in self.latent_paths:
                # Try the caption search dir first
                txt_path = caption_search_dir / f"{lat_path.stem}.txt"
                if txt_path.exists():
                    self.captions[str(lat_path)] = txt_path.read_text().strip()
                else:
                    # Fall back to _load_caption which checks parent dir too
                    caption = _load_caption(lat_path)
                    if caption:
                        self.captions[str(lat_path)] = caption
            caption_count = len(self.captions)
            fallback_count = len(self.latent_paths) - caption_count
            print(f"Captions: {caption_count} loaded, {fallback_count} using filename fallback")

        if use_captions and prompt_templates is None:
            self.templates = DEFAULT_CAPTION_TEMPLATES
        else:
            self.templates = prompt_templates or DEFAULT_TEMPLATES

    def __len__(self):
        return len(self.latent_paths) * len(self.templates)

    def __getitem__(self, idx):
        lat_idx = idx % len(self.latent_paths)
        tmpl_idx = idx // len(self.latent_paths)

        lat_path = self.latent_paths[lat_idx]
        latents = torch.load(lat_path, weights_only=True)
        caption = self.captions.get(str(lat_path)) if self.use_captions else None
        if caption is not None:
            template = self.templates[tmpl_idx % len(self.templates)]
            prompt = template.format(name=self.concept_token, caption=caption,
                                     filewords=_filewords(lat_path))
        else:
            fallback = DEFAULT_TEMPLATES[tmpl_idx % len(DEFAULT_TEMPLATES)]
            prompt = fallback.format(name=self.concept_token,
                                     filewords=_filewords(lat_path))

        return {"latents": latents, "prompt": prompt}


def cache_latents(vae, image_dir: str, image_size: int = 512, device=None):
    """
    Pre-encode all images in image_dir to latents and save as .pt files.

    Args:
        vae: The VAE model (on GPU)
        image_dir: Directory containing source images
        image_size: Size to resize/crop images to
        device: Device to use for encoding

    Returns:
        Path to the cache directory ({image_dir}/.latent_cache/)
    """
    if device is None:
        device = next(vae.parameters()).device

    image_dir = Path(image_dir)
    cache_dir = image_dir / ".latent_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    transform = T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.LANCZOS),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ])

    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )

    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    cached_count = 0
    skipped_count = 0

    for img_path in image_paths:
        cache_path = cache_dir / f"{img_path.stem}.pt"

        # Skip if already cached
        if cache_path.exists():
            skipped_count += 1
            continue

        image = Image.open(img_path).convert("RGB")
        # Shape: (1, C, 1, H, W) — VAE expects 5D with frame dimension
        pixel_values = transform(image).unsqueeze(0).unsqueeze(2).to(device, dtype=torch.bfloat16)

        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = _normalize_latents(vae, latents)
            # Remove frame dim for storage: (1, C, 1, H, W) → (C, H, W)
            latents = latents.squeeze(2)

        # Save to disk (CPU tensor to save GPU memory)
        torch.save(latents.squeeze(0).cpu(), cache_path)
        cached_count += 1

    total = cached_count + skipped_count
    print(f"Latent cache: {cached_count} encoded, {skipped_count} already cached, {total} total")
    print(f"Cache dir: {cache_dir}")

    return str(cache_dir)


# ──────────────────────────────────────────────────────────────────────────────
# loss.py
# ──────────────────────────────────────────────────────────────────────────────

import torch.nn.functional as F


def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """
    Pack spatial latents into patch sequences for the transformer.

    (B, C, H, W) → (B, (H/2)*(W/2), C*4)

    This matches the packing in QwenImagePipeline._pack_latents:
    each 2x2 spatial patch is flattened into the channel dimension.
    """
    bsz, c, h, w = latents.shape
    latents = latents.view(bsz, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)  # (B, H/2, W/2, C, 2, 2)
    latents = latents.reshape(bsz, (h // 2) * (w // 2), c * 4)
    return latents


def _unpack_latents(latents: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """
    Unpack patch sequences back to spatial latents.

    (B, (H/2)*(W/2), C*4) → (B, C, H, W)

    Inverse of _pack_latents.
    """
    bsz = latents.shape[0]
    c = latents.shape[2] // 4
    latents = latents.view(bsz, h // 2, w // 2, c, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)  # (B, C, H/2, 2, W/2, 2)
    latents = latents.reshape(bsz, c, h, w)
    return latents


def flow_matching_loss(
    transformer,
    scheduler,
    latents: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Flow matching (rectified flow) training loss for QwenImageTransformer2DModel.

    The scheduler adds noise at a random timestep. We predict the velocity field
    (v = noise - data direction) and compute MSE against the model's prediction.

    This is the standard objective for FlowMatchEulerDiscreteScheduler.
    """
    bsz = latents.shape[0]
    device = latents.device
    _, _, h, w = latents.shape

    # Sample random timesteps as continuous values in [0, 1] for flow matching.
    # FlowMatchEulerDiscreteScheduler uses continuous timesteps, not discrete indices.
    # Must match latents dtype (bf16) to avoid dtype mismatch in transformer.
    timesteps = torch.rand(bsz, device=device, dtype=latents.dtype)

    # Sample noise (same dtype as latents)
    noise = torch.randn_like(latents)

    # Flow matching linear interpolation: x_t = (1-t)*data + t*noise
    # Reshape t for broadcasting: (bsz,) → (bsz, 1, 1, 1)
    t = timesteps.view(-1, *([1] * (latents.ndim - 1)))
    noisy_latents = (1 - t) * latents + t * noise

    # Rectified flow velocity target: v = noise - data
    # dx/dt = noise - latents. The model predicts this velocity field.
    target = noise - latents

    # Pack spatial latents into patch sequences for the transformer.
    # (B, C, H, W) → (B, (H/2)*(W/2), C*4) — matches QwenImagePipeline._pack_latents
    noisy_latents = _pack_latents(noisy_latents)
    target = _pack_latents(target)

    # img_shapes: post-patchify (frame, height/2, width/2) for RoPE positioning.
    img_shapes = [(1, h // 2, w // 2)] * bsz

    # Forward through transformer (MMDiT).
    # Per-block gradient checkpointing is enabled on the transformer in the training script,
    # so only one block's activations are in memory at a time during backward.
    model_pred = transformer(
        hidden_states=noisy_latents,
        timestep=timesteps,
        encoder_hidden_states=encoder_hidden_states,
        encoder_hidden_states_mask=attention_mask,
        img_shapes=img_shapes,
        return_dict=False,
    )[0]

    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
    return loss
