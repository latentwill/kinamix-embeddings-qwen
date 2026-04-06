# Kinamix Embeddings for Qwen-Image

Train custom concept embeddings for **Qwen-Image** (MM-DiT + Qwen2.5-VL text encoder). Teaches the model new visual concepts — styles, textures, objects — through learned embedding tokens injected directly into the DiT's conditioning space. No model fine-tuning required.

## Why Embeddings Over LoRA?

LoRA modifies the model's weights directly. Every LoRA is tied to one specific model checkpoint, and stacking multiple LoRAs creates interference and quality loss.

Embeddings work differently — they operate in the model's **conditioning space**, not its weight space:

- **No weight modification.** The base model stays completely frozen. Your concept lives as a small portable file.
- **Composable.** Multiple embeddings can coexist in the same prompt without interference.
- **Tiny files.** A trained embedding is ~56 KB vs ~100-500 MB for a LoRA.
- **Model-agnostic within architecture.** An embedding trained on Qwen-Image works with any checkpoint sharing the same text encoder + DiT architecture.
- **Fast to train.** 500-1000 steps, 5-15 minutes on an A40. No large datasets or overnight runs.

The trade-off: embeddings have a narrower influence than LoRA. They're best for styles, textures, and visual motifs rather than fine anatomical detail.

## How It Works

**DiT-Side Concept Injection (DSCI)** trains N learnable tokens that are concatenated to the text encoder's output before entering the DiT transformer. The text encoder stays completely frozen — gradients flow through the DiT back to the concept tokens only.

This bypasses the language model's attractor basins, giving concepts more freedom to encode visual features that don't map cleanly to words.

## Requirements

- NVIDIA GPU with **48GB+ VRAM** (A40, A100, H100)
- Python 3.11+
- CUDA 12.x
- [Hugging Face token](https://huggingface.co/settings/tokens)

---

## RunPod Setup (Recommended)

### 1. Launch a pod

- **Template:** RunPod PyTorch 2.x
- **GPU:** A40 (48GB) or A100 (80GB)
- **Disk:** 80GB+ container disk
- **Cloud type:** Secure Cloud or Community

### 2. Clone and install

```bash
# SSH into the pod or use the web terminal
cd /workspace

git clone https://github.com/latentwill/kinamix-embeddings-qwen.git
cd kinamix-embeddings-qwen

# Set your Hugging Face token
echo 'HF_TOKEN=hf_xxxxx' > .env

# Run setup — installs deps and downloads models (~35GB)
bash setup.sh --precision full
```

Setup takes ~10 minutes. It downloads the Qwen-Image pipeline, validates CUDA, and runs a quick sanity check.

### 3. Prepare training images

Upload 5-20 images of your concept to a directory:

```bash
mkdir -p /workspace/training
# Upload images via RunPod file manager or scp
# Supported: .png, .jpg, .jpeg, .webp
# Square crops work best — images are resized to 512px
```

### 4. Train

```bash
python train_dsci.py \
    --image_dir /workspace/training \
    --output_path /workspace/output/my_concept.safetensors \
    --steps 1000 \
    --lr 1.85e-4 \
    --num_tokens 5 \
    --lr_schedule one_cycle \
    --checkpoint_interval 250 \
    --precision full
```

Training produces:
- `my_concept.safetensors` — the trained embedding (~56 KB)
- `my_concept_metrics.csv` — per-step loss and learning rate data
- `preview_my_concept/` — auto-generated preview images
- Checkpoint files at every 250 steps

### 5. Generate images

```bash
# DFG preview — grid of concept at different scales
python generate_dsci.py \
    --emb_path /workspace/output/my_concept.safetensors \
    --concept_scale 3.0 \
    --steps 30 \
    --precision full

# Custom prompts
python generate_dsci.py \
    --emb_path /workspace/output/my_concept.safetensors \
    --prompt "a forest" "a portrait" "a cityscape at night" \
    --concept_scale 2.5
```

### 6. Download results

Download the output directory via RunPod file manager, `scp`, or `rsync`.

---

## Training Reference

### Core options

```bash
python train_dsci.py \
    --image_dir ./my_images \
    --output_path ./output/my_concept.safetensors \
    --steps 1000 \
    --lr 1.85e-4 \
    --num_tokens 5 \
    --lr_schedule one_cycle \
    --precision full
```

| Flag | Default | Description |
|------|---------|-------------|
| `--image_dir` | required | Directory of concept images (5-20 images) |
| `--output_path` | `./output/dsci_concept.safetensors` | Output file path |
| `--steps` | `500` | Training steps |
| `--lr` | `1e-3` | Learning rate |
| `--num_tokens` | `4` | Number of concept tokens to inject |
| `--seed` | random | Random seed for reproducibility |
| `--image_size` | `512` | Training image resolution |
| `--precision` | `fp8` | Model precision (`fp8` or `full`) |
| `--dit_dtype` | `fp8` | DiT dtype (`fp8` or `bf16`) |
| `--lr_schedule` | `constant` | See LR schedules below |
| `--warmup_frac` | `0.1` | Fraction of steps for warmup |
| `--checkpoint_interval` | none | Save checkpoint every N steps |
| `--no-preview` | off | Skip preview generation after training |
| `--preview_checkpoints` | off | Generate previews for each checkpoint |
| `--init_from` | none | Initialize from a previous embedding (phased training) |
| `--use_captions` | off | Use `.txt` caption files alongside images |

**LR schedules:** `constant`, `warmup_constant`, `warmup_linear_decay`, `one_cycle`, `warmup_exp_decay`, `cosine_restarts`

### Recommended starting config

For style concepts with 10-20 images:

```bash
--steps 1000 --lr 1.85e-4 --num_tokens 5 --lr_schedule one_cycle
```

For quick tests (verify pipeline works):

```bash
--steps 100 --lr 1e-3 --num_tokens 4
```

<details>
<summary><strong>Experimental options</strong></summary>

These flags are functional but considered experimental.

| Flag | Default | Description |
|------|---------|-------------|
| `--token_position` | `append` | Where to inject tokens: `prepend`, `append`, `interleave` |
| `--low_rank` | off | Use low-rank factorized tokens (A @ B) |
| `--low_rank_rank` | `8` | Inner rank for low-rank mode |
| `--dmag_weight` | `0.0` | Delta magnitude loss (range: 0.0005-0.005) |
| `--cda_weight` | `0.0` | Cross-image direction alignment (range: 0.2-0.5) |
| `--cda_buffer_size` | `8` | Rolling buffer size for CDA loss |
| `--tid_weight` | `0.0` | Timestep-invariant direction loss (range: 0.1-0.3) |
| `--contrastive_weight` | `0.0` | Contrastive penalty against language priors |
| `--norm_encourage` | off | Reward token drift from initialization |
| `--norm_alpha` | `1e-6` | Weight for norm encouragement |
| `--attention_diag` | off | Log attention map diagnostics |
| `--diag_interval` | `100` | Steps between attention snapshots |
| `--adaptive_cfg` | off | Adaptive gating for DMag+CDA weights |
| `--target_norm` | `1.0` | Target norm for adaptive DMag gating |
| `--mass_target` | `0.1` | Attention mass target for adaptive gating |

</details>

---

## Generation Reference

### Single embedding

```bash
# Default (CFG decomposition, concept_scale=3.5)
python generate_dsci.py \
    --emb_path ./output/my_concept.safetensors

# Custom prompts and scale
python generate_dsci.py \
    --emb_path ./output/my_concept.safetensors \
    --prompt "a mountain landscape" "a quiet street" \
    --concept_scale 2.5 \
    --steps 30

# Scale scheduling (higher at noisy steps, lower at clean)
python generate_dsci.py \
    --emb_path ./output/my_concept.safetensors \
    --scale_schedule linear \
    --scale_high 3.0 \
    --scale_low 1.5
```

### DFG (Decomposed Feature Generation) preview

Generates a grid comparing different concept scales against a fixed text scale:

```bash
python preview_cfg.py \
    --emb_path ./output/my_concept.safetensors \
    --concept_scales 1 2 3 4 5 \
    --text_scale 7.0 \
    --precision full
```

This is the primary way to evaluate how well your concept was learned. Look for the scale where the concept is clearly present without overwhelming prompt composition.

### Batch inference

Load models once, generate previews for multiple embeddings:

```bash
python scripts/batch_inference.py \
    --emb_dir ./output/embeddings \
    --output_dir ./output/previews \
    --concept_scale 3.0 \
    --schedules constant linear \
    --precision full
```

---

## Embedding Format

Embeddings are saved as `.safetensors` files with metadata:

```python
from embedding import Embedding

# Load and inspect
emb = Embedding.load("my_concept.safetensors")
print(emb.info())
# method         : dsci
# hidden_dim     : 3584
# tokens         : 5
# token_position : append
# params         : 17920 (70.0 KB)
# norm           : 0.1234

# Access the raw tokens
print(emb.tokens.shape)  # torch.Size([5, 3584])
```

---

## Training Images

- **5-20 images** of the concept you want to teach
- Square crops work best; images are resized to `--image_size` (default 512)
- Supported formats: `.png`, `.jpg`, `.jpeg`, `.webp`
- Place in a flat directory (no subdirectories needed)
- For caption-based training (`--use_captions`): place `.txt` files alongside images with matching filenames

## VRAM Usage

Training uses three-phase VRAM isolation to fit within GPU memory:

1. **Phase A:** VAE loads alone, caches all image latents to disk, then unloads
2. **Phase B:** Text encoder + DiT load for training (VAE never on GPU)
3. **Phase C:** For preview generation, text encoder + DiT offload to CPU, VAE reloads

| Mode | VRAM Required | Status |
|------|--------------|--------|
| Full precision | ~48 GB | Tested (A40, A100) |
| FP8 | ~20-24 GB | Experimental |

---

## Project Structure

```
kinamix-embeddings-qwen/
  train_dsci.py           # Training script
  generate_dsci.py        # Image generation
  preview_cfg.py          # DFG decomposition preview
  preview.py              # Preview engine (manual inference)
  embedding.py            # Embedding save/load
  setup.sh                # Environment setup
  requirements.txt        # Python dependencies
  modules/
    model_loader.py       # FP8 + full precision model loading
    dit_injection.py      # DSCI token injection into DiT
    dataset_and_loss.py   # Dataset, latent caching, flow matching loss
    cfg_aware_loss.py     # CFG-aware training losses (DMag, CDA, TID)
    adaptive_cfg.py       # Adaptive gating for CFG losses
    contrastive_loss.py   # Language basin contrastive penalty
    attention_hooks.py    # Attention map analysis hooks
    scale_schedules.py    # Timestep-dependent scale schedules
    low_rank_injection.py # Low-rank factorized token variant
  scripts/
    batch_inference.py    # Batch generation across multiple embeddings
  examples/
    prompts_style.txt     # Style concept prompt templates
    prompts_object.txt    # Object concept prompt templates
```

## ComfyUI Integration

Use trained embeddings in ComfyUI with the companion node pack:

**[kinamix-embeddings-comfyui](https://github.com/latentwill/kinamix-embeddings-comfyui)** — Load `.safetensors` embeddings and apply DSCI concept injection in ComfyUI workflows.

## License

MIT
