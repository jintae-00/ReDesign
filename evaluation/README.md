# `evaluation/`

Reconstruction-accuracy and editability evaluation for ReDesign and the baselines.

Run every script **from the repository root** so the `evaluation`,
`REDESIGN`, and `modules` packages resolve.

## Contents

| File | Purpose |
|---|---|
| `eval_accuracy_baselines_figma.py` | Reconstruction-accuracy eval on Figma (L1/SSIM/LPIPS/DINO, element + composite) |
| `eval_accuracy_baselines_crello.py` | Reconstruction-accuracy eval on Crello |
| `before_eval_editability_precompute_matches.py` | **Pre-step** for editability: precompute GT↔prediction element matches |
| `eval_editability_figma.py` | Editability eval (6 atomic edits: delete/opacity/recolor/rotation/transition/z_order) |
| `eval_editability_text_figma.py` | Text-editability eval (content recognition + modification, OCR-based) |
| `figma_metrics.py` / `crello_metrics.py` | Shared metric engine (element extraction, matching, metrics) |
| `baseline_model_configs.py` | GT / model-output discovery (supports the merged dataset layout) |
| `eval_editability_baselines.py` | Editability subtask runners (used by `eval_editability_figma.py`) |
| `assets/atomic_selected_subset.json` | Frozen episode subset for paper-reproducible editability numbers |

## Prerequisite: generate model outputs

Evaluation scores **inference outputs**, so first run the agent (and any baselines
you want to compare) to produce `episodes/<id>/parse.json` etc.:

```bash
python -m REDESIGN.run_agent_figma --data_dir figma_data --output_dir outputs/figma_agent
```

`baseline_model_configs.collect_gt_episodes` auto-detects the **merged** dataset
(`figma_data/valid_frames/…`); GT discovery then ignores the `gt_prefix` field of
`--exp-pairs` (every episode is collected). Model-output discovery accepts both the
flat `episodes/` layout (produced by `run_agent_figma.py`) and the legacy
`split_*/episodes/` layout.

## 1. Reconstruction accuracy

```bash
python evaluation/eval_accuracy_baselines_figma.py \
    --figma-data figma_data \
    --models agent \
    --exp-pairs outputs/figma_agent:outputs/figma_qwen:merged \
    --output outputs/eval_accuracy_figma

python evaluation/eval_accuracy_baselines_crello.py \
    --crello-subset crello_data/records \
    --models agent --agent-dir outputs/crello_agent \
    --output outputs/eval_accuracy_crello
```

## 2. Editability (two-step)

Editability requires the precomputed GT↔pred matches **before** scoring:

```bash
# Step A — precompute matches (per model)
python evaluation/before_eval_editability_precompute_matches.py \
    --figma-data figma_data \
    --model agent --model-dir outputs/figma_agent \
    --exp-pairs outputs/figma_agent:outputs/figma_qwen:merged \
    --output outputs/editability_matches

# Step B — atomic-edit editability
python evaluation/eval_editability_figma.py        # paths via env vars (below)

# Text editability
python evaluation/eval_editability_text_figma.py \
    --figma-data figma_data --models agent \
    --exp-pairs outputs/figma_agent:outputs/figma_qwen:merged \
    --output outputs/eval_editability_text
```

`eval_editability_figma.py` reads its paths from environment variables (with
sensible defaults under `outputs/`):

| Env var | Default | Meaning |
|---|---|---|
| `REDESIGN_FIGMA_DATA` | `figma_data` | dataset root |
| `REDESIGN_MATCH_ROOT` | `outputs/editability_matches` | precompute output (Step A) |
| `REDESIGN_EDIT_OUTPUT` | `outputs/eval_editability_figma` | this script's output |
| `REDESIGN_EXP_PAIRS` | `outputs/figma_agent:outputs/figma_qwen:merged` | agent/qwen output dirs |
| `REDESIGN_<MODEL>_DIR` | `outputs/baseline_<model>` | per-baseline output dirs |
| `REDESIGN_SUBSET_FILE` | `evaluation/assets/atomic_selected_subset.json` | frozen subset for reproducibility |

> Editability eval depends on LPIPS / DINO (auto-downloaded) and, for text eval,
> PaddleOCR PP-OCRv5 (auto-downloaded, GPU).
