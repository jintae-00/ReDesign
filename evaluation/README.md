# `evaluation/`

Two evaluation axes from the paper, for the agent **and** the baselines:

- **Reconstruction accuracy** — render each method's editable output back to a
  raster with a fixed renderer and compare it to the input image
  (element-level L1/IoU, panoptic quality PQ/SQ/RQ, and composite
  L1/PSNR/SSIM/LPIPS/DINO).
- **Editability** (edit-replay) — apply an atomic edit to the ground-truth design
  *and* to the matched element in the prediction, render both, and compare. Six
  edits: `delete / opacity / recolor / rotation / transition / z_order`.

Run every script **from the repository root** (so `evaluation`, `ReDesign`, and
`modules` resolve). The metric models run on the GPU set by `--gpu-ids <GPU_ID>`
(PaddleOCR text eval uses `--ocr-gpu <GPU_ID>`); both default to GPU 0.

## Contents

| File | Purpose |
|---|---|
| `eval_accuracy_baselines_figma.py` | Reconstruction accuracy on Figma |
| `eval_accuracy_baselines_crello.py` | Reconstruction accuracy on Crello |
| `eval_editability_figma.py` | Atomic-edit editability (6 edits) |
| `eval_editability_text_figma.py` | Text-editability (OCR-based content recognition + modification) |
| `before_eval_editability_precompute_matches.py` | GT↔prediction element matching (auto-run by the editability eval; can also be run standalone) |
| `figma_metrics.py` / `crello_metrics.py` | Shared metric engine (element extraction, matching, metrics) |
| `baseline_model_configs.py` | GT and model-output discovery |
| `eval_editability_baselines.py` | Editability subtask runners used by `eval_editability_figma.py` |
| `assets/atomic_selected_subset.json` | Frozen episode/edit subset for paper-reproducible editability numbers |

## Prerequisite

Evaluation scores inference outputs, so first run the agent (and any baselines you
want to compare) — see the top-level README §5. Each model writes
`episodes/<id>/parse.json` (agent / multi-tools / sparse-verif) or `layer_*.png`
(qwen / layered) under its output dir. You pass the agent dir with `--agent-dir`
and each baseline dir with `--<model>-dir` (defaults: `outputs/baseline_<model>`).

## 1. Reconstruction accuracy

**One run evaluates every model listed in `--models`** — the agent and the
baselines together — and prints/saves a single comparison table.

```bash
# Figma
python evaluation/eval_accuracy_baselines_figma.py \
    --figma-data figma_data \
    --models agent qwen layered multi_tools vtracer \
    --agent-dir outputs/figma_agent \
    --gpu-ids <GPU_ID> \
    --output outputs/eval_accuracy_figma

# Crello
python evaluation/eval_accuracy_baselines_crello.py \
    --crello-subset crello_data/records \
    --models agent qwen layered multi_tools vtracer \
    --agent-dir outputs/crello_agent \
    --gpu-ids <GPU_ID> \
    --output outputs/eval_accuracy_crello
```

Baseline dirs default to `outputs/baseline_<model>`; override any with
`--qwen-dir / --layered-dir / --multi-tools-dir / --vtracer-dir / …`.

**Results** land in a timestamped subfolder of `--output`, e.g.
`outputs/eval_accuracy_figma/<timestamp>/`:
- `comparison_accuracy.md` / `comparison_accuracy.csv` — the agent-vs-baselines table
- `evaluation_unified_summary.json` — full per-model / per-episode metrics
- `worker_*_gpu*.log` — per-worker logs

## 2. Editability (atomic edits)

The same command works for the **agent and the baselines** — just list them in
`--models`. The required GT↔prediction matches are **auto-precomputed per model on
first use** (into `REDESIGN_MATCH_ROOT`), so there is no separate manual step.

```bash
# Agent only
REDESIGN_FIGMA_DATA=figma_data REDESIGN_AGENT_DIR=outputs/figma_agent \
    python evaluation/eval_editability_figma.py --models agent

# Agent + baselines (qwen/layered/multi_tools/vtracer) in one comparison
REDESIGN_FIGMA_DATA=figma_data REDESIGN_AGENT_DIR=outputs/figma_agent \
    python evaluation/eval_editability_figma.py \
    --models agent qwen layered multi_tools vtracer
```

Paths are configured via environment variables (sensible defaults under `outputs/`):

| Env var | Default | Meaning |
|---|---|---|
| `REDESIGN_FIGMA_DATA` | `figma_data` | dataset root |
| `REDESIGN_AGENT_DIR` | `outputs/figma_agent` | agent inference output dir |
| `REDESIGN_<MODEL>_DIR` | `outputs/baseline_<model>` | per-baseline output dir (qwen included, e.g. `REDESIGN_QWEN_DIR` → `outputs/baseline_qwen`) |
| `REDESIGN_MATCH_ROOT` | `outputs/editability_matches` | GT↔pred matches (auto-built) |
| `REDESIGN_EDIT_OUTPUT` | `outputs/eval_editability_figma` | this script's output |
| `REDESIGN_SUBSET_FILE` | `evaluation/assets/atomic_selected_subset.json` | frozen subset for reproducibility |

**Results** in `outputs/eval_editability_figma/`:
- `comparison_editability.md` / `.csv` — per-model means + per-subtask detail
- `atomic_<model>_overview.json`, `per_episode_<subtask>.json`

**Text editability** (content recognition + word-replacement, OCR-based):

```bash
python evaluation/eval_editability_text_figma.py \
    --figma-data figma_data --models agent qwen layered multi_tools vtracer \
    --agent-dir outputs/figma_agent --ocr-gpu <GPU_ID> \
    --output outputs/eval_editability_text
```

> Editability depends on LPIPS / DINO (auto-downloaded); text editability also
> uses PaddleOCR PP-OCRv5 (auto-downloaded, GPU).
