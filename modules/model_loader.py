"""
modules/model_loader.py — Centralized model loading for FP8 and full precision.

Supports two precision modes:
  - fp8: bf16 text encoder (from Qwen/Qwen-Image bundle) + FP8 DiT with layerwise
         casting. Text encoder uses ~14GB, DiT ~10GB → fits 24-48GB VRAM.
  - full: Original HF repos, full bf16. Needs 48GB+ VRAM.

Usage:
    from modules.model_loader import load_models, add_precision_arg

    add_precision_arg(parser)
    args = parser.parse_args()
    text_encoder, tokenizer, transformer, vae, scheduler = load_models(args)
"""

import argparse
import time
import torch
from pathlib import Path

# ── Default paths ────────────────────────────────────────────────────────────
FULL_PIPELINE_PATH = "./models/qwen-image"
FULL_TEXT_ENCODER_PATH = "./models/qwen-image/text_encoder"
TOKENIZER_PATH = "./models/qwen-image/tokenizer"

# DiT transformer path (diffusers format from Qwen/Qwen-Image)
DIFFUSERS_DIT_DIR = str(Path(FULL_PIPELINE_PATH) / "transformer")

# ── Sampler / sigma schedule selection ───────────────────────────────────────

SAMPLER_CLASSES = {
    "euler": "FlowMatchEulerDiscreteScheduler",
    "heun": "FlowMatchHeunDiscreteScheduler",
}

SIGMA_SCHEDULE_KWARGS = {
    "simple": {},
    "karras": {"use_karras_sigmas": True},
    "exponential": {"use_exponential_sigmas": True},
    "beta": {"use_beta_sigmas": True},
}


def load_scheduler(
    sampler: str = "euler",
    sigma_schedule: str = "simple",
):
    """Load a flow-matching scheduler by sampler name and sigma schedule.

    Args:
        sampler: "euler" (1 NFE/step) or "heun" (2 NFE/step, higher order).
        sigma_schedule: "simple" (linear), "karras", "exponential", or "beta".

    Returns:
        A diffusers SchedulerMixin instance loaded from the model config.
    """
    if sampler not in SAMPLER_CLASSES:
        raise ValueError(f"Unknown sampler '{sampler}'. Valid: {list(SAMPLER_CLASSES.keys())}")
    if sigma_schedule not in SIGMA_SCHEDULE_KWARGS:
        raise ValueError(f"Unknown sigma_schedule '{sigma_schedule}'. Valid: {list(SIGMA_SCHEDULE_KWARGS.keys())}")

    import diffusers

    cls = getattr(diffusers, SAMPLER_CLASSES[sampler])
    kwargs = SIGMA_SCHEDULE_KWARGS[sigma_schedule]
    return cls.from_pretrained(FULL_PIPELINE_PATH, subfolder="scheduler", **kwargs)


# ── Chat template for text encoding ─────────────────────────────────────────
# The Qwen2.5-VL text encoder in transformers>=5.0 expects input in chat format.
# Without this template, the VL model's forward produces 0-length hidden states.
# This matches exactly what diffusers' QwenImagePipeline.encode_prompt() uses.
PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, quantity, "
    "text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
# Number of tokens in the system prefix to drop from hidden states.
# The pipeline drops these so the returned embeddings represent only the
# user prompt + assistant tokens — matching what the DiT was trained on.
PROMPT_TEMPLATE_DROP_TOKENS = 34


def _file_size_str(path: str) -> str:
    """Human-readable file size, or '' if file doesn't exist."""
    p = Path(path)
    if not p.exists():
        return ""
    size = p.stat().st_size
    if size >= 1 << 30:
        return f" ({size / (1 << 30):.1f} GB)"
    if size >= 1 << 20:
        return f" ({size / (1 << 20):.0f} MB)"
    return f" ({size / (1 << 10):.0f} KB)"


def _step(msg: str):
    """Print a loading step with a timestamp prefix and flush immediately."""
    print(f"  [{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_embed_layer(text_encoder):
    """
    Resolve the embed_tokens layer from a Qwen2.5-VL text encoder.

    Qwen2_5_VLForConditionalGeneration nests the text model under .model.language_model,
    not .model directly. This helper abstracts the path so callers don't need to know.

    Returns the nn.Embedding layer (embed_tokens).
    """
    model = text_encoder.model
    # Qwen2_5_VLModel has .language_model (text) and .visual (vision)
    if hasattr(model, "language_model"):
        return model.language_model.embed_tokens
    # Fallback for non-VL models that have embed_tokens directly
    return model.embed_tokens


def get_text_layers(text_encoder):
    """
    Resolve the transformer layer list from a Qwen2.5-VL text encoder.

    VL models nest the text layers under .model.language_model.layers.
    """
    model = text_encoder.model
    if hasattr(model, "language_model"):
        return model.language_model.layers
    return model.layers


def get_text_hidden_size(text_encoder) -> int:
    """
    Get the text model's hidden dimension from a Qwen2.5-VL text encoder.

    VL models nest the text config under config.text_config.
    """
    config = text_encoder.config
    if hasattr(config, "text_config"):
        return config.text_config.hidden_size
    return config.hidden_size


def add_precision_arg(parser: argparse.ArgumentParser):
    """Add --precision and --dit_dtype flags to an argument parser."""
    parser.add_argument(
        "--precision",
        choices=["fp8", "full"],
        default="fp8",
        help="Model precision: fp8 (bf16 text enc + FP8 DiT, 24-48GB VRAM) or full (original HF, 48GB+)",
    )
    parser.add_argument(
        "--dit_dtype",
        choices=["fp8", "bf16"],
        default="fp8",
        help="DiT transformer dtype: fp8 (layerwise casting, ~10GB VRAM) or bf16 (full, ~20GB VRAM)",
    )


def load_fp8_models(args, device, components=None):
    """
    Load models for FP8 mode: bf16 text encoder + FP8 DiT with layerwise casting.

    The text encoder is loaded via standard from_pretrained from the bundled
    Qwen/Qwen-Image/text_encoder/ directory. This produces correct outputs
    (verified). The Comfy-Org FP8 text encoder is
    NOT compatible with diffusers and produces garbled images.

    components: set of which components to load. Default: all.
                Valid values: {"text_encoder", "tokenizer", "transformer", "vae", "scheduler"}

    Returns: dict with requested components.
    """
    from diffusers import (
        AutoencoderKLQwenImage,
        FlowMatchEulerDiscreteScheduler,
        QwenImageTransformer2DModel,
    )
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
    from safetensors.torch import load_file

    if components is None:
        components = {"text_encoder", "tokenizer", "transformer", "vae", "scheduler"}

    result = {}

    # ── Text encoder (bf16 from bundled Qwen/Qwen-Image/text_encoder/) ───
    if "text_encoder" in components or "tokenizer" in components:
        te_path = FULL_TEXT_ENCODER_PATH
        print(f"\n── Loading Qwen2.5-VL text encoder (bf16) from {te_path} ──")

        t0 = time.time()
        _step("Loading via from_pretrained (bf16)...")
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            te_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )

        # NOTE: We keep the visual encoder loaded (~2GB) even though it's not
        # used for text-to-image. The VL model's forward routing in transformers>=5.0
        # needs self.model.visual to exist for correct token handling, even with
        # text-only input. Deleting it causes 0-length hidden states in attention.
        # The ~2GB overhead is negligible on 40GB+ GPUs.

        _step(f"Moving to {device}...")
        text_encoder = text_encoder.to(device=device)

        print(f"── Qwen2.5-VL text encoder ready ({time.time()-t0:.1f}s total) ──\n")
        result["text_encoder"] = text_encoder

    if "tokenizer" in components:
        # Load tokenizer from pipeline root — the text_encoder/ subfolder has
        # an incomplete vocab (size 1). The pipeline root has the full tokenizer.
        result["tokenizer"] = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    # ── DiT transformer (diffusers format) ─────────────────────────────
    if "transformer" in components:
        dit_dir = DIFFUSERS_DIT_DIR
        dit_dtype = getattr(args, "dit_dtype", "fp8")
        print(f"\n── Loading Qwen-Image DiT from {dit_dir} ({dit_dtype}) ──")

        t0 = time.time()

        import glob
        st_files = glob.glob(str(Path(dit_dir) / "*.safetensors"))
        if not st_files:
            raise FileNotFoundError(
                f"No safetensors files in {dit_dir}\n"
                f"Run 'bash setup.sh' to download the transformer weights."
            )

        # Load safetensors to CPU first. DiT weights are ~20GB in bf16.
        # For FP8: layerwise casting compresses before GPU transfer (~10GB).
        # For bf16: transferred as-is (~20GB), needs more VRAM.
        _step("Reading safetensors to CPU...")
        dit_state_dict = {}
        for st_file in st_files:
            dit_state_dict.update(load_file(st_file, device="cpu"))
        _step(f"Loaded ({len(dit_state_dict)} tensors, {time.time()-t0:.1f}s)")

        from accelerate import init_empty_weights
        dit_config = QwenImageTransformer2DModel.load_config(dit_dir)
        dit_config.pop("pooled_projection_dim", None)
        with init_empty_weights():
            transformer = QwenImageTransformer2DModel.from_config(dit_config)

        transformer.load_state_dict(dit_state_dict, strict=True, assign=True)
        del dit_state_dict

        # Enable layerwise casting BEFORE moving to GPU (FP8 only).
        # Weights transfer as FP8 (~10GB) instead of bf16 (~20GB).
        if dit_dtype == "fp8" and hasattr(transformer, "enable_layerwise_casting"):
            t3 = time.time()
            _step("Enabling layerwise casting (bf16 → FP8, may take a moment)...")
            transformer.enable_layerwise_casting(
                storage_dtype=torch.float8_e4m3fn,
                compute_dtype=torch.bfloat16,
            )
            _step(f"Layerwise casting enabled ({time.time()-t3:.1f}s)")

        t4 = time.time()
        _step(f"Moving to {device}...")
        transformer = transformer.to(device=device)
        _step(f"On device ({time.time()-t4:.1f}s)")

        print(f"── Qwen-Image DiT ready ({time.time()-t0:.1f}s total) ──\n")
        result["transformer"] = transformer

    # ── VAE (full precision from original HF repo — small, not quantized) ─
    if "vae" in components:
        vae_dir = str(Path(FULL_PIPELINE_PATH) / "vae")
        t0 = time.time()
        _step(f"Loading VAE from {vae_dir}...")

        vae = AutoencoderKLQwenImage.from_pretrained(
            vae_dir,
            torch_dtype=torch.bfloat16,
        ).to(device)

        _step(f"VAE ready ({time.time()-t0:.1f}s)")
        result["vae"] = vae

    # ── Scheduler (no weights, just config) ──────────────────────────────
    if "scheduler" in components:
        _step("Loading scheduler config...")
        result["scheduler"] = load_scheduler(
            sampler=getattr(args, "sampler", "euler"),
            sigma_schedule=getattr(args, "sigma_schedule", "simple"),
        )

    return result


def load_full_models(args, device, components=None):
    """
    Load models from original HF repositories (full precision).
    This is the existing loading path, unchanged behavior.

    Returns: dict with requested components.
    """
    from diffusers import QwenImagePipeline
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

    if components is None:
        components = {"text_encoder", "tokenizer", "transformer", "vae", "scheduler"}

    result = {}

    # Text encoder loaded separately for direct embed_tokens access
    if "text_encoder" in components or "tokenizer" in components:
        te_path = FULL_TEXT_ENCODER_PATH
        print(f"\n── Loading text encoder (full precision) ──")
        t0 = time.time()
        _step(f"Loading from {te_path}...")
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            te_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        ).to(device)
        _step(f"Text encoder ready ({time.time()-t0:.1f}s)")
        result["text_encoder"] = text_encoder

    if "tokenizer" in components:
        result["tokenizer"] = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    # Pipeline for transformer, VAE, scheduler
    if any(c in components for c in ("transformer", "vae", "scheduler")):
        pipe_path = FULL_PIPELINE_PATH
        print(f"\n── Loading pipeline (full precision) ──")
        t0 = time.time()
        _step(f"Loading from {pipe_path}...")
        pipe = QwenImagePipeline.from_pretrained(
            pipe_path,
            text_encoder=result.get("text_encoder"),
            tokenizer=result.get("tokenizer"),
            torch_dtype=torch.bfloat16,
        ).to(device)
        _step(f"Pipeline ready ({time.time()-t0:.1f}s)")

        if "transformer" in components:
            result["transformer"] = pipe.transformer
        if "vae" in components:
            result["vae"] = pipe.vae
        if "scheduler" in components:
            result["scheduler"] = pipe.scheduler

    return result


def load_models(args, device=None, components=None):
    """
    Load model components based on --precision flag.

    Args:
        args: Parsed arguments (must have .precision attribute)
        device: torch device (default: cuda if available)
        components: set of components to load (default: all)
                    Valid: {"text_encoder", "tokenizer", "transformer", "vae", "scheduler"}

    Returns: dict with keys for each requested component.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    precision = getattr(args, "precision", "fp8")
    dit_dtype = getattr(args, "dit_dtype", "fp8")
    comp_str = ", ".join(sorted(components)) if components else "all"
    print(f"\n{'='*60}")
    print(f"Loading models ({precision} precision, DiT {dit_dtype}): {comp_str}")
    print(f"{'='*60}")

    t0 = time.time()
    if precision == "fp8":
        result = load_fp8_models(args, device, components)
    else:
        result = load_full_models(args, device, components)

    print(f"{'='*60}")
    print(f"Loaded in {time.time()-t0:.1f}s")
    print(f"{'='*60}\n")

    return result
