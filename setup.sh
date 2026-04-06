#!/bin/bash
# setup.sh
# Run once on a fresh GPU instance to install deps and download models.
#
# Usage:
#   bash setup.sh                  # Default: bf16 text enc + FP8 DiT (~34GB VRAM)
#   bash setup.sh --precision fp8  # Same as default
#   bash setup.sh --precision full # Original HF models (~35GB, needs 48GB+ VRAM)
#
# Option A: Set HF_TOKEN as an environment variable before running.
# Option B: Create a .env file in this directory with HF_TOKEN=hf_xxxxx

set -e

# ── Parse arguments ─────────────────────────────────────────────────────────
PRECISION="fp8"
while [[ $# -gt 0 ]]; do
    case $1 in
        --precision)
            PRECISION="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash setup.sh [--precision fp8|full]"
            exit 1
            ;;
    esac
done

if [[ "$PRECISION" != "fp8" && "$PRECISION" != "full" ]]; then
    echo "ERROR: --precision must be 'fp8' or 'full' (got: $PRECISION)"
    exit 1
fi

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   Qwen Concept Embedding — GPU Setup             ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "  Precision:  $PRECISION"
echo "  Download:   $([ "$PRECISION" = "fp8" ] && echo "~34GB (bf16 TE + FP8 DiT)" || echo "~35GB (full precision)")"
echo "  VRAM:       $([ "$PRECISION" = "fp8" ] && echo "~34GB (A40/A6000/A100)" || echo "~48GB+ (pro GPU)")"
echo ""

# ── GPU/CUDA/Driver Stack Validation ─────────────────────────────────────────
echo "━━━ GPU/CUDA STACK VALIDATION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

STACK_OK=1

# nvidia-smi check
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "query failed")
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo "unknown")
    # Parse CUDA version from nvidia-smi banner (works on all driver versions)
    CUDA_DRIVER=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "unknown")
    echo "  ✓ GPU:            $GPU_NAME"
    echo "  ✓ Driver:         $DRIVER_VER"
    echo "  ✓ CUDA (driver):  $CUDA_DRIVER"
else
    echo "  ✗ nvidia-smi not found — no GPU detected"
    STACK_OK=0
fi

# PyTorch pre-check (informational only — will be fixed by install step)
python3 -c "
import sys
try:
    import torch
    print(f'  Current PyTorch:  {torch.__version__}')
    if torch.cuda.is_available():
        print(f'  CUDA available:   True')
        print(f'  CUDA (torch):     {torch.version.cuda}')
    else:
        print(f'  CUDA available:   False — will fix during install')
except ImportError:
    print('  PyTorch not yet installed — will install below')
" 2>&1 || true

# Disk space check (try /workspace first, fall back to current directory)
DISK_CHECK_PATH="/workspace"
if ! df -k "$DISK_CHECK_PATH" &>/dev/null; then
    DISK_CHECK_PATH="."
fi
DISK_FREE_KB=$(df -k "$DISK_CHECK_PATH" 2>/dev/null | tail -1 | awk '{print $4}' || echo "0")
DISK_FREE_GB=$((DISK_FREE_KB / 1024 / 1024))
if [[ $DISK_FREE_GB -lt 20 ]]; then
    echo "  ✗ Disk free: ${DISK_FREE_GB}GB — LOW (need 20GB+)"
    STACK_OK=0
else
    echo "  ✓ Disk free:      ${DISK_FREE_GB}GB"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [[ $STACK_OK -eq 1 ]]; then
    echo "  Stack validation: PASSED"
else
    echo "  Stack validation: ISSUES FOUND (see above)"
    echo "  Continuing with setup — fix issues before training"
fi
echo ""

# ── Load .env if present ─────────────────────────────────────────────────────
if [ -f .env ]; then
    echo "Loading .env..."
    export $(grep -v '^#' .env | xargs)
fi

# ── Authenticate ─────────────────────────────────────────────────────────────
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not set. Either:"
    echo "  1. Set it as an environment variable, or"
    echo "  2. Create a .env file: echo 'HF_TOKEN=hf_xxxxx' > .env"
    exit 1
fi

# ── Detect compatible CUDA version for PyTorch ──────────────────────────────
echo "▸ Detecting CUDA compatibility..."
CUDA_DRIVER_VER=""
if command -v nvidia-smi &>/dev/null; then
    # Parse major.minor from nvidia-smi (e.g., "12.8" from driver)
    CUDA_DRIVER_VER=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "")
fi

# Determine the right PyTorch CUDA index URL
TORCH_INDEX=""
if [[ -n "$CUDA_DRIVER_VER" ]]; then
    CUDA_MAJOR=$(echo "$CUDA_DRIVER_VER" | cut -d. -f1)
    CUDA_MINOR=$(echo "$CUDA_DRIVER_VER" | cut -d. -f2)

    # Pick the highest compatible cu* index that doesn't exceed the driver
    if [[ "$CUDA_MAJOR" -ge 13 ]]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu128"
        echo "  Driver CUDA $CUDA_DRIVER_VER → using PyTorch cu128"
    elif [[ "$CUDA_MAJOR" -eq 12 ]]; then
        if [[ "$CUDA_MINOR" -ge 8 ]]; then
            TORCH_INDEX="https://download.pytorch.org/whl/cu124"
            echo "  Driver CUDA $CUDA_DRIVER_VER → using PyTorch cu124"
        elif [[ "$CUDA_MINOR" -ge 4 ]]; then
            TORCH_INDEX="https://download.pytorch.org/whl/cu124"
            echo "  Driver CUDA $CUDA_DRIVER_VER → using PyTorch cu124"
        elif [[ "$CUDA_MINOR" -ge 1 ]]; then
            TORCH_INDEX="https://download.pytorch.org/whl/cu121"
            echo "  Driver CUDA $CUDA_DRIVER_VER → using PyTorch cu121"
        else
            TORCH_INDEX="https://download.pytorch.org/whl/cu121"
            echo "  Driver CUDA $CUDA_DRIVER_VER → using PyTorch cu121 (best match)"
        fi
    elif [[ "$CUDA_MAJOR" -eq 11 ]]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
        echo "  Driver CUDA $CUDA_DRIVER_VER → using PyTorch cu118"
    else
        echo "  ⚠ Unknown CUDA $CUDA_DRIVER_VER — installing default PyTorch (may fail)"
    fi
else
    echo "  ⚠ Could not detect CUDA version — installing default PyTorch"
fi

# ── System deps ──────────────────────────────────────────────────────────────
echo "▸ Installing Python dependencies..."
pip install -q --root-user-action=ignore --upgrade pip

# Install PyTorch with the correct CUDA version FIRST
# Must uninstall existing torch to prevent pip from skipping the downgrade
if [[ -n "$TORCH_INDEX" ]]; then
    echo "  Removing incompatible PyTorch..."
    pip uninstall -y -q torch torchvision torchaudio 2>/dev/null || true
    echo "  Installing PyTorch from $TORCH_INDEX ..."
    pip install --root-user-action=ignore \
        torch torchvision torchaudio \
        --index-url "$TORCH_INDEX"
else
    pip install --root-user-action=ignore torch torchvision torchaudio
fi

# Verify CUDA works before continuing
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available after PyTorch install'" 2>&1 || {
    echo "  ✗ CUDA still not available after PyTorch install."
    echo "  Try manually: pip install torch --index-url https://download.pytorch.org/whl/cu124"
    exit 1
}
echo "  ✓ PyTorch + CUDA verified"

# Install remaining deps (no torch — already installed)
pip install -q --root-user-action=ignore --upgrade \
    "diffusers>=0.36.0" \
    transformers \
    accelerate \
    huggingface_hub \
    hf_transfer \
    Pillow \
    safetensors \
    tiktoken \
    protobuf \
    tensorboard \
    qwen-vl-utils \
    python-dotenv \
    requests
echo "  ✓ Dependencies installed"
echo ""

# Enable hf_transfer for faster downloads (C-based, multi-connection)
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "▸ Authenticating with Hugging Face..."
python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
echo "  ✓ Authenticated"
echo ""

# ── Create directories ───────────────────────────────────────────────────────
mkdir -p models concepts outputs

# ── Download models ──────────────────────────────────────────────────────────
if [ "$PRECISION" = "fp8" ]; then
    # FP8 path: bf16 text encoder + FP8 DiT from Qwen/Qwen-Image bundle
    python - <<'DLEOF'
import os, sys, time
from huggingface_hub import snapshot_download

qi_dir = "./models/qwen-image"

# Download the full Qwen/Qwen-Image pipeline (text_encoder + transformer + VAE + configs)
# The text_encoder/ subfolder contains bf16 weights (~14GB).
# The transformer/ subfolder contains diffusers-format DiT weights (~20GB).
# The vae/ subfolder contains the VAE weights (~0.2GB).
dit_weights = os.path.join(qi_dir, "transformer", "diffusion_pytorch_model-00001-of-00007.safetensors")
te_weights = os.path.join(qi_dir, "text_encoder", "model-00001-of-00005.safetensors")

if os.path.exists(dit_weights) and os.path.exists(te_weights):
    print("  ✓ Qwen-Image pipeline (text encoder + DiT + VAE + configs) — already exists")
else:
    print("┌─────────────────────────────────────────────────┐")
    print("│  Downloading Qwen/Qwen-Image pipeline           │")
    print("│  Includes: bf16 text encoder, DiT, VAE, configs │")
    print("│  Size: ~34 GB → ./models/qwen-image/            │")
    print("└─────────────────────────────────────────────────┘")
    print()

    for attempt in range(1, 4):
        try:
            t0 = time.time()
            snapshot_download(
                repo_id="Qwen/Qwen-Image",
                local_dir=qi_dir,
                ignore_patterns=["*.bin", "*.msgpack", "*.h5", "flax_model*"],
            )
            elapsed = time.time() - t0
            print(f"  ✓ Qwen-Image pipeline downloaded ({elapsed:.0f}s)")
            break
        except Exception as e:
            if attempt < 3:
                wait = attempt * 10
                print(f"  ⚠ Attempt {attempt}/3 failed: {e}")
                print(f"  ↻ Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  ✗ Failed after 3 attempts: {e}")
                sys.exit(1)
    print()
DLEOF

    echo "╔══════════════════════════════════════════════════╗"
    echo "║            Setup complete (FP8)                 ║"
    echo "╠══════════════════════════════════════════════════╣"
    echo "║  Text enc (bf16): ./models/qwen-image/text_encoder/ ║"
    echo "║  DiT (bf16→FP8):  ./models/qwen-image/transformer/ ║"
    echo "║  VAE + configs:   ./models/qwen-image/             ║"
    echo "║  VRAM needed:     ~34 GB                           ║"
    echo "╚══════════════════════════════════════════════════╝"

else
    # Full precision path: original HF repos (~35GB total, needs 48GB+ VRAM)
    python - <<'PYEOF'
import os, sys, time
from huggingface_hub import snapshot_download

qi_dir = "./models/qwen-image"

# Download full Qwen/Qwen-Image pipeline (includes text encoder)
dit_weights = os.path.join(qi_dir, "transformer", "diffusion_pytorch_model-00001-of-00007.safetensors")
te_weights = os.path.join(qi_dir, "text_encoder", "model-00001-of-00005.safetensors")

if os.path.exists(dit_weights) and os.path.exists(te_weights):
    print("  ✓ Qwen-Image pipeline — already exists")
else:
    print("┌─────────────────────────────────────────────────┐")
    print("│  Downloading full-precision models              │")
    print("│  Total: ~35 GB → ./models/                      │")
    print("└─────────────────────────────────────────────────┘")
    print()

    for attempt in range(1, 4):
        try:
            t0 = time.time()
            snapshot_download(
                repo_id="Qwen/Qwen-Image",
                local_dir=qi_dir,
                ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
            )
            elapsed = time.time() - t0
            print(f"  ✓ Done in {elapsed:.0f}s")
            break
        except Exception as e:
            if attempt < 3:
                wait = attempt * 10
                print(f"  ⚠ Attempt {attempt}/3 failed: {e}")
                print(f"  ↻ Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  ✗ Failed after 3 attempts: {e}")
                sys.exit(1)
    print()
PYEOF

    echo "╔══════════════════════════════════════════════════╗"
    echo "║         Setup complete (full precision)         ║"
    echo "╠══════════════════════════════════════════════════╣"
    echo "║  Pipeline:     ./models/qwen-image/              ║"
    echo "║  Text encoder: ./models/qwen-image/text_encoder/ ║"
    echo "║  VRAM needed:  ~48 GB+                           ║"
    echo "╚══════════════════════════════════════════════════╝"
fi

echo ""
echo "━━━ POST-SETUP STACK RE-VALIDATION ━━━━━━━━━━━━━━━━━━━━━━━"
python3 -c "
import torch
print(f'  PyTorch:        {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA (torch):   {torch.version.cuda}')
    print(f'  GPU:            {torch.cuda.get_device_name(0)}')
    # Quick smoke test: allocate a small tensor on GPU
    try:
        t = torch.randn(100, 100, device='cuda')
        del t
        print(f'  GPU alloc test:  PASSED')
    except Exception as e:
        print(f'  GPU alloc test:  FAILED — {e}')
else:
    print(f'  WARNING: CUDA not available after setup')
" 2>&1
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "Next steps:"
echo ""
echo "  # Train a single embedding"
echo "  python train_dsci.py --image_dir ./training --output_path ./output/test.safetensors --steps 5000 --lr 1.85e-4 --num_tokens 5 --lr_schedule constant"
echo ""
