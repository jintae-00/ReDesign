# `BASELINES/`

Baseline methods compared against the ReDesign agent in the paper. Run **from the
repository root** as modules (e.g. `python -m BASELINES.run_layerd_figma`) so the
shared packages (`REDESIGN`, `tool_learning_wo_qwen`, `modules`, `config`) resolve.

| Script | Method | Dataset |
|---|---|---|
| `run_layerd_figma.py` / `run_layerd_crello.py` | Layered front-decomposition (LaMa) | Figma / Crello |
| `run_multi_tools_figma.py` / `run_multi_tools_crello.py` | Multi-tool (GDINO+SAM2+Hi-SAM+OCR) without the agent controller | Figma / Crello |
| `run_qwen_figma.py` / `run_qwen_crello.py` | Qwen-Image-Layered direct layering | Figma / Crello |
| `run_sparse_verification_agent_figma.py` | Full agent with sparse verification (ablation) | Figma |
| `run_vtracer_baseline.py` | VTracer vectorization | Figma / Crello |

These share the agent's tool backends:

- `run_layerd_*`, `run_multi_tools_*` import `tool_learning_wo_qwen.tools.*`
  (bundled at the repo root) which wrap the same `modules/` checkpoints.
- `run_qwen_*` use `from diffusers import QwenImageLayeredPipeline` (`Qwen/Qwen-Image-Layered`).
- `run_sparse_verification_agent_figma.py` reuses `REDESIGN.episode_run` with a
  sparse-verification monkey-patch.

So the same environment (`environment.yml` + `post_install.sh`) and checkpoints
(`scripts/download_checkpoints.py`) cover the baselines too.

## Input layout

The baselines were written for the original **per-split** dataset layout
(`figma_data/process/subset/<prefix>_split_*` for Figma; `crello_splits/` for
Crello) and the original GPU/CLI flags — see each script's `--help` and the
`Configuration` block near the top. They are included **as-is** for
reproducibility; only their package imports were updated for this release
(`tool_learning` → `REDESIGN`). To run a baseline on the released merged
`figma_data`, point its data path at the dataset and adjust the split constants
in the script header as needed.

Baseline outputs feed the evaluation in `../evaluation/` (point the eval
`--*-dir` flags / `REDESIGN_<MODEL>_DIR` env vars at the baseline output dirs).
