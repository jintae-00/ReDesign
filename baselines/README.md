# `baselines/`

Baseline methods compared against the ReDesign agent in the paper. Run **from the
repository root** as modules (e.g. `python -m baselines.run_layerd_figma`) so the
shared packages (`ReDesign`, `baselines.tool_backends`, `modules`, `config`) resolve.

| Script | Method | Dataset |
|---|---|---|
| `run_layerd_figma.py` / `run_layerd_crello.py` | Layered front-decomposition (LaMa) | Figma / Crello |
| `run_multi_tools_figma.py` / `run_multi_tools_crello.py` | Multi-tool (GDINO+SAM2+Hi-SAM+OCR) without the agent controller | Figma / Crello |
| `run_qwen_figma.py` / `run_qwen_crello.py` | Qwen-Image-Layered direct layering | Figma / Crello |
| `run_sparse_verification_agent_figma.py` | Full agent with sparse verification (ablation) | Figma |
| `run_vtracer_baseline.py` | VTracer vectorization | Figma / Crello |

These share the agent's tool backends:

- `run_layerd_*`, `run_multi_tools_*` import `baselines.tool_backends.tools.*`
  (a bundled subpackage) which wrap the same `modules/` checkpoints.
- `run_qwen_*` use `from diffusers import QwenImageLayeredPipeline` (`Qwen/Qwen-Image-Layered`).
- `run_sparse_verification_agent_figma.py` reuses `ReDesign.episode_run` with a
  sparse-verification monkey-patch.

So the same environment (`environment.yml` + `post_install.sh`) and checkpoints
(`scripts/download_checkpoints.py`) cover the baselines too.

## Input / output (split-agnostic)

Like the agent runners, every baseline takes a **whole dataset directory** and an
output directory — there is no split concept. GPU ids are placeholders; the Qwen
baseline needs ~2 GPUs (see the main README compute section).

```bash
# Figma baselines: --data_dir is the merged dataset (valid_frames/ + unit_images/)
python -m baselines.run_layerd_figma \
    --data_dir figma_data --output_dir outputs/baseline_layered --gpu <GPU_IDS>

python -m baselines.run_qwen_figma \
    --data_dir figma_data --output_dir outputs/figma_qwen \
    --qwen_gpus <QWEN_GPU_IDS> --qwen_pair_size 2

# Crello baselines: --data_dir holds crello_test_*/ records (composite.png)
python -m baselines.run_layerd_crello \
    --data_dir crello_data/records --output_dir outputs/baseline_layered_crello --gpu <GPU_IDS>
```

See each script's `--help` for its full flag set. Baseline outputs feed the
evaluation in `../evaluation/` — pass them with the eval `--<model>-dir` flags /
`REDESIGN_<MODEL>_DIR` env vars (e.g. `--layered-dir outputs/baseline_layered`).
