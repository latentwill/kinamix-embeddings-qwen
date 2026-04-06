"""
train_dsci.py -- DiT-Side Concept Injection (DSCI) Training

Trains N learned concept tokens that are concatenated to text encoder hidden
states BEFORE passing to the DiT. The text encoder is completely frozen --
gradient flows through the DiT back to the concept tokens only.

Unlike textual inversion:
  - No hooks on embed_tokens, no token manipulation
  - Text encoder is fully frozen (no gradient checkpointing needed)
  - Concept tokens operate in the DiT's conditioning space, not the
    text encoder's language space -- bypassing language attractor basins

Usage:
    python train_dsci.py \\
        --image_dir ./my_concept_images \\
        --output_path ./concepts/my_concept.safetensors \\
        --steps 500 \\
        --lr 1e-3 \\
        --num_tokens 4

    # Or call as a function:
    from train_dsci import train_dsci
    result = train_dsci(image_dir="./images", output_path="./out.safetensors", steps=500, lr=1e-3)
"""

import argparse
import csv
import gc
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules.dit_injection import DiTConceptInjection
from modules.low_rank_injection import LowRankDiTConceptInjection
from modules.dataset_and_loss import (
    CachedLatentDataset,
    cache_latents,
    flow_matching_loss,
    DEFAULT_TEMPLATES,
    DEFAULT_CAPTION_TEMPLATES,
)
from modules.model_loader import (
    load_models,
    get_text_hidden_size,
)
from embedding import Embedding
from modules.model_loader import PROMPT_TEMPLATE, PROMPT_TEMPLATE_DROP_TOKENS


def encode_prompt(text_encoder, tokenizer, prompt, device):
    """
    Encode prompt through Qwen2.5-VL text encoder using the chat template.

    Returns (hidden_states, attention_mask) with the system prefix dropped,
    matching the DiT's expected conditioning format.
    """
    if isinstance(prompt, str):
        prompt = [prompt]

    drop = PROMPT_TEMPLATE_DROP_TOKENS
    formatted = [PROMPT_TEMPLATE.format(p) for p in prompt]

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


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    schedule: str,
    steps: int,
    warmup_frac: float = 0.1,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """
    Build an LR scheduler for DSCI training.

    Args:
        optimizer: The optimizer to schedule.
        schedule: One of "constant", "warmup_constant", "warmup_linear_decay",
                  "one_cycle", "warmup_exp_decay", "cosine_restarts".
        steps: Total training steps.
        warmup_frac: Fraction of steps for warmup (default 10%).

    Returns:
        LR scheduler, or None for "constant".
    """
    warmup_steps = max(1, int(steps * warmup_frac))

    if schedule == "constant":
        return None

    if schedule == "warmup_constant":
        # Linear ramp 0→LR over warmup, then hold
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.01, total_iters=warmup_steps
                ),
                torch.optim.lr_scheduler.ConstantLR(
                    optimizer, factor=1.0, total_iters=steps - warmup_steps
                ),
            ],
            milestones=[warmup_steps],
        )

    if schedule == "warmup_linear_decay":
        # Triangle: ramp up, then linearly decay to 0
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.01, total_iters=warmup_steps
                ),
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=1.0, end_factor=0.01,
                    total_iters=steps - warmup_steps
                ),
            ],
            milestones=[warmup_steps],
        )

    if schedule == "one_cycle":
        # Super-convergence: low→high→very low in one cycle
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optimizer.defaults["lr"],
            total_steps=steps,
            pct_start=0.3,       # 30% ramp-up
            anneal_strategy="cos",
            div_factor=10,       # start at LR/10
            final_div_factor=100,  # end at LR/1000
        )

    if schedule == "warmup_exp_decay":
        # Ramp up, then exponential decay (half-life = 1/3 of remaining steps)
        decay_steps = steps - warmup_steps
        # gamma per step to decay to 1% by end
        gamma = (0.01) ** (1.0 / max(1, decay_steps))
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.01, total_iters=warmup_steps
                ),
                torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma),
            ],
            milestones=[warmup_steps],
        )

    if schedule == "cosine_restarts":
        # Cosine annealing with warm restarts every 1/3 of training
        cycle_len = max(1, steps // 3)
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=cycle_len, T_mult=1, eta_min=0
        )

    raise ValueError(f"Unknown schedule: {schedule}")


VALID_SCHEDULES = [
    "constant",
    "warmup_constant",
    "warmup_linear_decay",
    "one_cycle",
    "warmup_exp_decay",
    "cosine_restarts",
]


def train_dsci(
    image_dir: str,
    output_path: str,
    steps: int,
    lr: float,
    num_tokens: int = 4,
    seed: int | None = None,
    concept_token: str = "<|emb|>",
    precision: str = "fp8",
    dit_dtype: str = "fp8",
    image_size: int = 512,
    no_preview: bool = False,
    lr_schedule: str = "constant",
    warmup_frac: float = 0.1,
    checkpoint_interval: int | None = None,
    use_captions: bool = False,
    low_rank: bool = False,
    low_rank_rank: int = 8,
    contrastive_weight: float = 0.0,
    dmag_weight: float = 0.0,
    cda_weight: float = 0.0,
    cda_buffer_size: int = 8,
    tid_weight: float = 0.0,
    preview_checkpoints: bool = False,
    token_position: str = "append",
    norm_encourage: bool = False,
    norm_alpha: float = 1e-6,
    attention_diag: bool = False,
    diag_interval: int = 100,
    adaptive_cfg: bool = False,
    target_norm: float = 1.0,
    mass_target: float = 0.1,
    init_from: str | None = None,
) -> dict:
    """
    Train DiT-Side Concept Injection (DSCI) tokens on concept images.

    Args:
        image_dir: Directory containing concept training images.
        output_path: Path where the .safetensors file will be saved.
        steps: Number of training steps.
        lr: Learning rate for the concept tokens.
        num_tokens: Number of concept tokens to inject (default 4).
        seed: Random seed for reproducibility (optional).
        concept_token: Placeholder token for prompts.
        precision: Model precision mode ("fp8" or "full").
        dit_dtype: DiT dtype ("fp8" or "bf16").
        image_size: Training image resolution.
        no_preview: Skip preview generation.
        lr_schedule: LR schedule ("constant", "warmup_constant",
                     "warmup_linear_decay", "one_cycle",
                     "warmup_exp_decay", "cosine_restarts").
        warmup_frac: Fraction of steps for warmup (default 0.1).
        checkpoint_interval: Save checkpoint every N steps (None = no checkpoints).
        low_rank: Use low-rank factorized tokens (A @ B) instead of full-rank.
        low_rank_rank: Inner rank for low-rank factorization (default 8).
        contrastive_weight: Weight for contrastive penalty against language priors.
                            0.0 = disabled. Recommended range: 0.05–0.2.
        dmag_weight: Weight for delta magnitude loss. Maximizes ||v_full - v_text||.
                     0.0 = disabled. Recommended: 0.0005–0.005 (packed latent norms ~8–15).
        cda_weight: Weight for cross-image direction alignment. Uses rolling buffer
                    to enforce consistent concept direction across training images.
                    0.0 = disabled. Recommended: 0.2–0.5.
        cda_buffer_size: Number of recent directions to buffer for CDA (default 8).
        tid_weight: Weight for timestep-invariant direction loss. Enforces same
                    concept direction at different noise levels.
                    0.0 = disabled. Recommended: 0.1–0.3.

        norm_encourage: Add norm encouragement loss to reward token drift from init.
                        Gently pushes tokens away from their random initialization.
        norm_alpha: Weight for norm encouragement (default: 1e-6, very gentle).
        attention_diag: Log attention map diagnostics during training.
        diag_interval: Steps between attention diagnostic snapshots (default: 100).
        adaptive_cfg: Enable three-sensor adaptive gating for DMag+CDA weights.
                      Requires dmag_weight or cda_weight > 0. Auto-enables attention_diag.
        target_norm: Concept direction magnitude target for adaptive DMag gating (default 1.0).
        mass_target: Attention mass fraction target for adaptive gating (default 0.1).

    Returns:
        Dict with "initial_loss", "final_loss", "elapsed_s", "finished_at",
        "checkpoint_paths" (list of saved checkpoint paths), and
        "metrics_csv" (path to per-step CSV with loss/lr/d_magnitude/etc).
    """
    if seed is not None:
        torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build a namespace that load_models expects
    args = argparse.Namespace(precision=precision, dit_dtype=dit_dtype)

    # Step 1: Load VAE, cache latents, free VAE
    # Disable cuDNN for VAE conv3d — workaround for CUDNN_STATUS_NOT_INITIALIZED
    # on environments with cuDNN/CUDA version mismatches
    _cudnn_was_enabled = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    models = load_models(args, device, components={"vae"})
    vae = models["vae"]
    cache_dir = cache_latents(vae, image_dir, image_size, device)
    del vae, models
    torch.backends.cudnn.enabled = _cudnn_was_enabled
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Step 2: Load text encoder + DiT
    models = load_models(
        args, device,
        components={"text_encoder", "tokenizer", "transformer", "scheduler"},
    )
    text_encoder = models["text_encoder"]
    tokenizer = models["tokenizer"]
    transformer = models["transformer"]
    scheduler = models["scheduler"]

    # Freeze everything
    for param in text_encoder.parameters():
        param.requires_grad = False
    for param in transformer.parameters():
        param.requires_grad = False

    # Enable gradient checkpointing on DiT to avoid OOM
    if hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()

    # Step 3: Create DSCI module -- the ONLY trainable thing
    hidden_dim = get_text_hidden_size(text_encoder)
    print(f"\n━━━ DSCI TRAINING ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    if low_rank:
        dsci = LowRankDiTConceptInjection(
            hidden_dim=hidden_dim, num_tokens=num_tokens, rank=low_rank_rank
        ).to(device)
        print(f"  Mode:      low-rank DSCI (rank={low_rank_rank}, params={num_tokens * low_rank_rank + low_rank_rank * hidden_dim})")
        init_tokens = dsci.concept_tokens.data.clone().cpu()
    else:
        dsci = DiTConceptInjection(hidden_dim=hidden_dim, num_tokens=num_tokens).to(device)
        print(f"  Mode:      full-rank DSCI (params={num_tokens * hidden_dim})")
        init_tokens = dsci.concept_tokens.data.clone().cpu()

    # Initialize from previous embedding (for phased training)
    if init_from is not None:
        prev = Embedding.load(init_from)
        if prev.tokens is not None and prev.tokens.shape == dsci.concept_tokens.shape:
            dsci.concept_tokens.data.copy_(prev.tokens.to(device))
            init_tokens = dsci.concept_tokens.data.clone().cpu()
            print(f"  Initialized from: {init_from}")
        else:
            print(f"  WARNING: Shape mismatch ({prev.tokens.shape} vs {dsci.concept_tokens.shape}), using random init")

    # Set up CFG-aware losses
    from modules.cfg_aware_loss import compute_cfg_aware_loss, DirectionBuffer
    cfg_losses_active = dmag_weight > 0 or cda_weight > 0 or tid_weight > 0
    direction_buffer = DirectionBuffer(capacity=cda_buffer_size) if cda_weight > 0 else None
    if cfg_losses_active:
        print(f"  CFG-aware losses: dmag={dmag_weight}, cda={cda_weight} (buf={cda_buffer_size}), tid={tid_weight}")

    # Set up adaptive CFG gating
    adaptive_gating = None
    if adaptive_cfg:
        if not cfg_losses_active:
            print("  WARNING: --adaptive_cfg requires --dmag_weight and/or --cda_weight > 0")
        else:
            from modules.adaptive_cfg import AdaptiveCFGGating
            import math
            _img_seq_len_adapt = (image_size // 16) * (image_size // 16)
            adaptive_gating = AdaptiveCFGGating(
                dmag_base=dmag_weight,
                cda_base=cda_weight,
                target_norm=target_norm,
                mass_target=mass_target,
                max_entropy=math.log(_img_seq_len_adapt + num_tokens),
                variance_ceiling=0.5,
                floor_frac=0.1,
            )
            # Force attention diagnostics on for adaptive mode
            if not attention_diag:
                attention_diag = True
                diag_interval = 50
                print(f"  Adaptive mode: enabled attention_diag with interval={diag_interval}")
            print(f"  Adaptive CFG: target_norm={target_norm}, mass_target={mass_target}")

    # Build language prior vectors for contrastive loss (once, before training)
    prior_vectors = None
    if contrastive_weight > 0.0:
        from modules.contrastive_loss import build_language_priors
        print(f"  Contrastive weight: {contrastive_weight} — building language priors...")
        prior_vectors = build_language_priors(text_encoder, tokenizer, device)
        print(f"  Language priors built: {prior_vectors.shape[0]} cluster vectors")

    # Step 4: Dataset from cached latents
    dataset = CachedLatentDataset(
        cache_dir=cache_dir,
        concept_token=concept_token,
        prompt_templates=DEFAULT_CAPTION_TEMPLATES if use_captions else DEFAULT_TEMPLATES,
        use_captions=use_captions,
        image_dir=image_dir,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Step 5: Optimizer + LR scheduler
    optimizer = torch.optim.AdamW(dsci.parameters(), lr=lr)
    lr_scheduler = _build_lr_scheduler(optimizer, lr_schedule, steps, warmup_frac)
    print(f"  LR:        {lr_schedule} | base {lr} | warmup {warmup_frac:.0%}")
    print(f"  Steps:     {steps} | images: {len(dataset)} | output: {output_path}")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # Step 6: Training loop
    initial_loss = None
    final_loss = None
    data_iter = iter(dataloader)
    checkpoint_paths: list[str] = []
    out_stem = Path(output_path).stem
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Open metrics CSV alongside the embedding file
    _CSV_COLS = [
        "step", "loss", "lr", "flow_loss",
        "d_magnitude", "dmag_loss", "cda_loss", "tid_loss", "contrastive_penalty",
        "mag", "direction_variance",
        "probe_mass_mean", "probe_entropy_mean",
        "dmag_effective", "cda_effective",
        "mag_gate", "var_gate", "attn_mass_gate", "entropy_gate",
    ]
    metrics_csv_path = str(out_dir / f"{out_stem}_metrics.csv")
    _csv_file = open(metrics_csv_path, "w", newline="")
    _csv_writer = csv.DictWriter(_csv_file, fieldnames=_CSV_COLS, extrasaction="ignore")
    _csv_writer.writeheader()

    # Initialize T5 attention probe collector
    attn_collector = None
    if attention_diag:
        from modules.attention_hooks import AttentionCollector
        _img_seq_len = (image_size // 16) * (image_size // 16)
        attn_collector = AttentionCollector(
            model=transformer,
            n_img_tokens=_img_seq_len,
            n_concept_tokens=num_tokens,
            concept_position=token_position,
            attn_pattern="attn",
        )
        attn_collector.register()
        print(f"  T5 attention hooks: {len(attn_collector._hooks)} blocks, diag every {diag_interval} steps")

    # Adaptive gating sensor cache (initialized to defaults for step 0)
    _last_d_magnitude = 0.0
    _last_probe_mass = 0.0
    _last_probe_entropy = 0.0

    t_start = time.time()

    pbar = tqdm(range(steps), desc="Training", unit="step", dynamic_ncols=True)
    for step in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        latents = batch["latents"].to(device, dtype=torch.bfloat16)
        prompts = batch["prompt"]

        # Encode prompt (text encoder is frozen, no grad needed through it)
        with torch.no_grad():
            hidden_states, attention_mask = encode_prompt(
                text_encoder, tokenizer, prompts, device
            )

        # Save pre-injection states for CFG-aware losses (v_text baseline)
        hidden_states_text = hidden_states
        attention_mask_text = attention_mask

        # Inject concept tokens -- this IS where gradient flows
        hidden_states_concept, attention_mask_concept = dsci.inject(
            hidden_states, attention_mask, position=token_position
        )

        # Enable attention capture for this step (before forward pass)
        if attn_collector is not None and step % diag_interval == 0:
            attn_collector.active = True

        # Compute effective CFG weights (adaptive or fixed)
        if cfg_losses_active and adaptive_gating is not None:
            _dir_var = direction_buffer.direction_variance() if direction_buffer is not None else 0.0
            if _dir_var is None:
                _dir_var = 0.0
            effective_dmag, effective_cda = adaptive_gating.compute(
                mag=_last_d_magnitude,
                direction_variance=_dir_var,
                probe_mass=_last_probe_mass,
                probe_entropy=_last_probe_entropy,
            )
        elif cfg_losses_active:
            effective_dmag = dmag_weight
            effective_cda = cda_weight
        else:
            effective_dmag = 0.0
            effective_cda = 0.0

        # Compute loss
        if cfg_losses_active:
            loss, cfg_metrics = compute_cfg_aware_loss(
                transformer=transformer,
                scheduler=scheduler,
                latents=latents,
                hidden_states_concept=hidden_states_concept,
                attention_mask_concept=attention_mask_concept,
                hidden_states_text=hidden_states_text,
                attention_mask_text=attention_mask_text,
                dmag_weight=effective_dmag,
                cda_weight=effective_cda,
                direction_buffer=direction_buffer,
                tid_weight=tid_weight,
            )
        else:
            loss = flow_matching_loss(
                transformer=transformer,
                scheduler=scheduler,
                latents=latents,
                encoder_hidden_states=hidden_states_concept,
                attention_mask=attention_mask_concept,
            )

        # Consume attention metrics (for adaptive gating and/or diagnostics logging)
        _attn_step_metrics = {}
        if attn_collector is not None and attn_collector._captured:
            _attn_step_metrics = attn_collector.compute_metrics()
            if "probe_mass" in _attn_step_metrics:
                _last_probe_mass = _attn_step_metrics["probe_mass"]
            if "probe_entropy" in _attn_step_metrics:
                _last_probe_entropy = _attn_step_metrics["probe_entropy"]
            attn_collector.active = False

        # Cache d_magnitude for next step's adaptive gating
        if cfg_losses_active and "d_magnitude" in cfg_metrics:
            _last_d_magnitude = cfg_metrics["d_magnitude"]

        # Contrastive penalty (push concept tokens away from language basin priors)
        if prior_vectors is not None and contrastive_weight > 0.0:
            from modules.contrastive_loss import contrastive_penalty
            penalty = contrastive_penalty(dsci.concept_tokens, prior_vectors)
            loss = loss + contrastive_weight * penalty

        # Norm encouragement: reward tokens drifting away from initialization
        if norm_encourage:
            token_drift_val = (dsci.concept_tokens - init_tokens.to(device)).norm(dim=-1).mean()
            norm_loss = -norm_alpha * token_drift_val
            loss = loss + norm_loss

        # Record losses
        loss_val = loss.item()
        if initial_loss is None:
            initial_loss = loss_val
        final_loss = loss_val

        # Attention diagnostics snapshot
        if attention_diag and step % diag_interval == 0:
            from modules.attention_diagnostics import save_diagnostics
            diag_metrics = {
                "token_norms": dsci.concept_tokens.data.norm(dim=-1).cpu().tolist(),
                "token_drift": (dsci.concept_tokens.data.cpu() - init_tokens).norm(dim=-1).mean().item(),
                "loss": loss_val,
            }
            # T5: attention probe metrics (already consumed above)
            if _attn_step_metrics:
                diag_metrics["attention"] = _attn_step_metrics
            diag_dir = str(out_dir / f"diagnostics_{out_stem}")
            save_diagnostics(diag_metrics, diag_dir, step)

        # Backward + step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dsci.parameters(), max_norm=1.0)
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        current_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else lr

        # Write every step to CSV
        row: dict = {"step": step, "loss": loss_val, "lr": current_lr}
        if cfg_losses_active:
            row["flow_loss"] = cfg_metrics.get("flow_loss", "")
            row["d_magnitude"] = cfg_metrics.get("d_magnitude", "")
            row["dmag_loss"] = cfg_metrics.get("dmag_loss", "")
            row["cda_loss"] = cfg_metrics.get("cda_loss", "")
            row["tid_loss"] = cfg_metrics.get("tid_loss", "")
        if prior_vectors is not None and contrastive_weight > 0.0:
            row["contrastive_penalty"] = (loss_val - (cfg_metrics.get("flow_loss", loss_val) if cfg_losses_active else loss_val))
        if adaptive_gating is not None and adaptive_gating.last_metrics:
            row.update(adaptive_gating.last_metrics)
            # Rename probe keys for CSV clarity
            row["probe_mass_mean"] = row.pop("probe_mass", "")
            row["probe_entropy_mean"] = row.pop("probe_entropy", "")
        _csv_writer.writerow(row)
        if step % 50 == 0:
            _csv_file.flush()

        # Update tqdm postfix with key metrics
        postfix = {"loss": f"{loss_val:.4f}", "lr": f"{current_lr:.2e}"}
        if cfg_losses_active and "d_magnitude" in cfg_metrics:
            postfix["|d|"] = f"{cfg_metrics['d_magnitude']:.4f}"
        if cfg_losses_active and "cda_loss" in cfg_metrics:
            postfix["cda"] = f"{cfg_metrics['cda_loss']:.4f}"
        if adaptive_gating is not None and adaptive_gating.last_metrics:
            am = adaptive_gating.last_metrics
            postfix["dmag_eff"] = f"{am['dmag_effective']:.6f}"
        pbar.set_postfix(postfix, refresh=step % 5 == 0)

        # Save checkpoint at interval (step+1 so we save after completing that step)
        if (
            checkpoint_interval is not None
            and (step + 1) % checkpoint_interval == 0
            and (step + 1) < steps  # don't checkpoint at final step (saved separately)
        ):
            ckpt_path = str(out_dir / f"{out_stem}_step{step + 1}.safetensors")
            ckpt_emb = Embedding.from_dsci(dsci.concept_tokens.data, token_position=token_position)
            ckpt_emb.save(ckpt_path)
            checkpoint_paths.append(ckpt_path)

    elapsed = time.time() - t_start
    finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    _csv_file.close()
    print(f"  Metrics saved: {metrics_csv_path}")

    # Step 7: Save embedding
    emb = Embedding.from_dsci(dsci.concept_tokens.data, token_position=token_position)
    emb.save(output_path)

    # Step 8: Generate preview samples (unless --no-preview)
    if not no_preview:
        from preview import generate_preview

        # Offload text encoder + transformer to CPU to free VRAM for VAE
        text_encoder.to("cpu")
        transformer.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Disable cuDNN for VAE conv3d — workaround for CUDNN_STATUS_NOT_INITIALIZED
        _cudnn_was_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False

        # Reload VAE for decoding (was freed in Step 1 to save VRAM)
        preview_models = load_models(args, device, components={"vae"})
        preview_vae = preview_models["vae"]

        # Move text encoder + transformer back to GPU for preview generation
        text_encoder.to(device)
        transformer.to(device)

        def apply_dsci(hs: torch.Tensor, mask: torch.Tensor) -> tuple:
            return dsci.inject(hs, mask, position=token_position)

        # Derive unique preview dir from the embedding filename
        emb_stem = Path(output_path).stem  # e.g. "dsci_4tok_lr1.85em04_one_cycle"
        preview_dir = str(Path(output_path).parent / f"preview_{emb_stem}")

        generate_preview(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            vae=preview_vae,
            scheduler=scheduler,
            concept_applier=apply_dsci,
            output_dir=preview_dir,
            title=emb_stem,
        )

        if preview_checkpoints and checkpoint_paths:
            print(f"  Generating previews for {len(checkpoint_paths)} checkpoints...")
            for ckpt_path in checkpoint_paths:
                ckpt_emb = Embedding.load(ckpt_path, device=device)

                ckpt_dsci = DiTConceptInjection(hidden_dim=hidden_dim, num_tokens=num_tokens).to(device)
                with torch.no_grad():
                    ckpt_dsci.concept_tokens.copy_(ckpt_emb.tokens.to(device))

                def apply_ckpt(hs, mask, _dsci=ckpt_dsci, _pos=token_position):
                    return _dsci.inject(hs, mask, position=_pos)

                ckpt_stem = Path(ckpt_path).stem
                ckpt_preview_dir = str(out_dir / f"preview_{ckpt_stem}")
                generate_preview(
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    transformer=transformer,
                    vae=preview_vae,
                    scheduler=scheduler,
                    concept_applier=apply_ckpt,
                    output_dir=ckpt_preview_dir,
                    title=ckpt_stem,
                )
                del ckpt_dsci, ckpt_emb

        del preview_vae, preview_models
        torch.backends.cudnn.enabled = _cudnn_was_enabled

    # Compute token metrics before cleanup (dsci is deleted below)
    token_norms = dsci.concept_tokens.data.norm(dim=-1).cpu().tolist()
    token_drift = (dsci.concept_tokens.data.cpu() - init_tokens).norm(dim=-1).mean().item()

    # Clean up attention hooks before freeing models
    if attn_collector is not None:
        attn_collector.cleanup()

    # Free all models and CUDA memory before returning (critical for grid runs)
    del text_encoder, tokenizer, transformer, scheduler, dsci, optimizer
    del dataloader, dataset, data_iter
    if lr_scheduler is not None:
        del lr_scheduler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "elapsed_s": elapsed,
        "finished_at": finished_at,
        "checkpoint_paths": checkpoint_paths,
        "metrics_csv": metrics_csv_path,
        "token_norms": token_norms,
        "token_drift": token_drift,
        "steps_per_sec": steps / elapsed if elapsed > 0 else 0.0,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Train a DSCI concept embedding")
    p.add_argument("--image_dir", required=True, help="Directory of concept images")
    p.add_argument("--output_path", default="./output/dsci_concept.safetensors",
                   help="Output .safetensors file path")
    p.add_argument("--steps", type=int, default=500, help="Training steps")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--num_tokens", type=int, default=4, help="Number of concept tokens")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--image_size", type=int, default=512, help="Training image size")
    p.add_argument("--precision", choices=["fp8", "full"], default="fp8")
    p.add_argument("--dit_dtype", choices=["fp8", "bf16"], default="fp8")
    p.add_argument("--no-preview", action="store_true", default=False,
                   help="Skip preview image generation after training")
    p.add_argument("--lr_schedule", choices=VALID_SCHEDULES, default="constant",
                   help="LR schedule (default: constant)")
    p.add_argument("--warmup_frac", type=float, default=0.1,
                   help="Fraction of steps for warmup (default: 0.1)")
    p.add_argument("--checkpoint_interval", type=int, default=None,
                   help="Save checkpoint every N steps (default: no checkpoints)")
    p.add_argument("--use_captions", action="store_true", default=False,
                   help="Use .txt caption files alongside images instead of filename-derived words")
    p.add_argument("--low_rank", action="store_true", default=False,
                   help="Use low-rank factorized tokens (A @ B) instead of full-rank")
    p.add_argument("--low_rank_rank", type=int, default=8,
                   help="Inner rank for low-rank factorization (default: 8)")
    p.add_argument("--contrastive_weight", type=float, default=0.0,
                   help="Weight for contrastive penalty against language priors (0.0 = disabled)")
    p.add_argument("--dmag_weight", type=float, default=0.0,
                   help="Delta magnitude loss weight (0=disabled). Maximizes ||v_full - v_text||. Range: 0.0005-0.005")
    p.add_argument("--cda_weight", type=float, default=0.0,
                   help="Cross-image direction alignment weight (0=disabled). Enforces consistent concept direction. Range: 0.2-0.5")
    p.add_argument("--cda_buffer_size", type=int, default=8,
                   help="Number of recent directions to buffer for CDA loss (default: 8)")
    p.add_argument("--tid_weight", type=float, default=0.0,
                   help="Timestep-invariant direction loss weight (0=disabled). Enforces noise-level-invariant style. Range: 0.1-0.3")
    p.add_argument("--preview_checkpoints", action="store_true", default=False,
                   help="Generate preview images for each checkpoint, not just final")
    p.add_argument("--token_position", choices=["prepend", "append", "interleave"],
                   default="append", help="Where to inject concept tokens in the sequence")
    p.add_argument("--norm_encourage", action="store_true", default=False,
                   help="Add norm encouragement loss to reward token drift from init")
    p.add_argument("--norm_alpha", type=float, default=1e-6,
                   help="Weight for norm encouragement (default: 1e-6, very gentle)")
    p.add_argument("--attention_diag", action="store_true", default=False,
                   help="Log attention map diagnostics during training")
    p.add_argument("--diag_interval", type=int, default=100,
                   help="Steps between attention diagnostic snapshots (default: 100)")
    p.add_argument("--adaptive_cfg", action="store_true", default=False,
                   help="Use three-sensor adaptive gating for DMag+CDA weights")
    p.add_argument("--target_norm", type=float, default=1.0,
                   help="Concept direction magnitude target for adaptive DMag gating")
    p.add_argument("--mass_target", type=float, default=0.1,
                   help="Attention mass target for adaptive gating")
    p.add_argument("--init_from", type=str, default=None,
                   help="Path to embedding file to initialize concept tokens from (for phased training)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    result = train_dsci(
        image_dir=args.image_dir,
        output_path=args.output_path,
        steps=args.steps,
        lr=args.lr,
        num_tokens=args.num_tokens,
        seed=args.seed,
        precision=args.precision,
        dit_dtype=args.dit_dtype,
        image_size=args.image_size,
        no_preview=args.no_preview,
        lr_schedule=args.lr_schedule,
        warmup_frac=args.warmup_frac,
        checkpoint_interval=args.checkpoint_interval,
        use_captions=args.use_captions,
        low_rank=args.low_rank,
        low_rank_rank=args.low_rank_rank,
        contrastive_weight=args.contrastive_weight,
        dmag_weight=args.dmag_weight,
        cda_weight=args.cda_weight,
        cda_buffer_size=args.cda_buffer_size,
        tid_weight=args.tid_weight,
        preview_checkpoints=args.preview_checkpoints,
        token_position=args.token_position,
        norm_encourage=args.norm_encourage,
        norm_alpha=args.norm_alpha,
        attention_diag=args.attention_diag,
        diag_interval=args.diag_interval,
        adaptive_cfg=args.adaptive_cfg,
        target_norm=args.target_norm,
        mass_target=args.mass_target,
        init_from=args.init_from,
    )

    elapsed = result["elapsed_s"]
    mins, secs = divmod(int(elapsed), 60)
    print(f"\n━━━ TRAINING COMPLETE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  ✓ Initial loss:  {result['initial_loss']:.4f}")
    print(f"  ✓ Final loss:    {result['final_loss']:.4f}")
    print(f"  ✓ Time:          {mins}m {secs}s ({result.get('steps_per_sec', 0):.2f} steps/s)")
    print(f"  ✓ Finished:      {result['finished_at']}")
    if result.get("token_norms"):
        norms_str = ", ".join(f"{n:.3f}" for n in result["token_norms"])
        print(f"  ✓ Token norms:   [{norms_str}]")
        print(f"  ✓ Token drift:   {result.get('token_drift', 0):.4f} (L2 from init)")
    if result["checkpoint_paths"]:
        print(f"  ✓ Checkpoints:   {len(result['checkpoint_paths'])} saved")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
