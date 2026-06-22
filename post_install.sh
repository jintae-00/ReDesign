#!/bin/bash
# ============================================================================
# ReDesign — post-installation script
#
# Run AFTER creating the conda environment from environment.yml:
#     conda env create -f environment.yml
#     conda activate agent_qwen_layerd
#     bash post_install.sh
#
# Installs the components that cannot be pinned in environment.yml (CUDA-specific
# wheels and a locally-compiled CUDA extension) and verifies the key imports.
# Run from the repository root.
# ============================================================================
set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ----------------------------------------------------------------------------
# CUDA / driver note: the wheels below are built for CUDA 12.8 (cu128), which
# runs on any NVIDIA driver providing CUDA >= 12.8 (e.g. driver 570+/CUDA 12.9).
# NVIDIA drivers are backward compatible, so cu128 wheels work on 12.8/12.9/13.x
# drivers. If your driver is OLDER than 12.8, install the matching index instead
# (e.g. .../whl/cu126 or .../whl/cu124) for both PyTorch and PaddlePaddle.
# ----------------------------------------------------------------------------

echo "=== [1/6] PaddlePaddle GPU 3.1.0 (cu126; runs on CUDA>=12.6 drivers) ==="
pip install paddlepaddle-gpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

echo "=== [2/6] PyTorch (stable, cu128) ==="
# Pinned to a stable cu128 build for reproducibility and driver compatibility.
# Installed AFTER PaddlePaddle so PyTorch's CUDA libraries (incl. NCCL) win.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo "=== [3/6] diffusers (from git: provides QwenImageLayeredPipeline) ==="
# QwenImageLayeredPipeline is not in the pinned PyPI diffusers; install from git.
pip install -U "git+https://github.com/huggingface/diffusers.git"

echo "=== [4/6] SAM 2 (segment-anything-2) ==="
pip install "git+https://github.com/facebookresearch/sam2.git"

echo "=== [5/6] Build GroundingDINO CUDA extension (bundled in modules/) ==="
# Needs a CUDA toolkit (nvcc) matching the torch build. Non-fatal: the tool can
# still run on the pure-Python fallback, just slower.
if [ -f "modules/grounding_dino/setup.py" ]; then
    ( cd modules/grounding_dino && pip install -e . --no-build-isolation ) \
        || echo "[WARN] GroundingDINO CUDA extension build failed (nvcc missing?). The GDINO tool will use the slower fallback."
else
    echo "[WARN] modules/grounding_dino/setup.py not found; skipping."
fi

echo "=== [6/6] Verification ==="
python - <<'PY'
import importlib, sys
ok = True
def check(mod, attr=None, label=None):
    global ok
    label = label or mod
    try:
        m = importlib.import_module(mod)
        if attr and not hasattr(m, attr):
            print(f"[FAIL] {label}: missing {attr}"); ok = False
        else:
            print(f"[ OK ] {label}")
    except Exception as e:
        print(f"[FAIL] {label}: {e}"); ok = False

import torch
print(f"[INFO] PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
check("paddle", label="paddlepaddle")
check("sam2", label="sam2 (pip)")
check("diffusers", "QwenImageLayeredPipeline", "diffusers.QwenImageLayeredPipeline")
check("transformers"); check("langchain_openai", label="langchain-openai")
check("paddleocr", label="paddleocr"); check("lpips"); check("vtracer"); check("cv2", label="opencv")
sys.exit(0 if ok else 1)
PY
status=$?
echo
if [ "$status" -eq 0 ]; then
    echo "=== Environment ready. ==="
else
    echo "=== Some checks FAILED — see [FAIL] lines above. ==="
fi
exit $status
