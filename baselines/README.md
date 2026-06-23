# `baselines/`

The comparison methods from the paper. They cover the three families ReDesign is
evaluated against: **layered image decomposition** (Qwen-Image-Layered, LayerD —
RGBA layers post-processed with OCR + vectorization to become editable), a
**linear tool-using agent** (ReAct-style: same tools/VLM, one fixed tool sequence
with a single end-stage validation, i.e. no per-step verification), and **image
vectorization** (VTracer). Each produces an editable output in the same format so
the same accuracy and editability metrics apply.

Run **from the repository root** as modules (e.g. `python -m baselines.run_layerd_figma`)
so the shared packages (`ReDesign`, `baselines.tool_backends`, `modules`, `config`)
resolve.

| Script | Method | Dataset |
|---|---|---|
| `run_layerd_figma.py` / `run_layerd_crello.py` | LayerD-style front-layer decomposition | Figma / Crello |
| `run_multi_tools_figma.py` / `run_multi_tools_crello.py` | Linear multi-tool pipeline (GDINO+SAM2+Hi-SAM+OCR), no controller | Figma / Crello |
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
output directory (no per-split handling). GPU ids are placeholders; the Qwen
baseline needs ≈55 GB of GPU memory (set `--qwen_pair_size` to your GPU count —
see the main README compute section).

```bash
# Figma baselines: --data_dir is the merged dataset (valid_frames/ + unit_images/)
python -m baselines.run_layerd_figma \
    --data_dir figma_data --output_dir outputs/baseline_layered --gpu <GPU_IDS>

python -m baselines.run_qwen_figma \
    --data_dir figma_data --output_dir outputs/baseline_qwen \
    --qwen_gpus <QWEN_GPU_IDS> --qwen_pair_size <N>

# Crello baselines: --data_dir holds crello_test_*/ records (composite.png)
python -m baselines.run_layerd_crello \
    --data_dir crello_data/records --output_dir outputs/baseline_layered_crello --gpu <GPU_IDS>
```

See each script's `--help` for its full flag set. Baseline outputs feed the
evaluation in `../evaluation/` — pass them with the eval `--<model>-dir` flags /
`REDESIGN_<MODEL>_DIR` env vars (e.g. `--layered-dir outputs/baseline_layered`).
