# `ReDesign/` — the agent

The agent that recovers an editable design from a raster image. It **grows a layer
hierarchy**: starting from the whole image as the root, a **VLM controller**
expands the tree breadth-first (coarse → fine), and a **modular verifier** checks
each expansion (accept / prune / retry) so the tree converges to atomic, editable
leaves. The result is exported as an editable JSON hierarchy (`parse.json`).

At each node the controller picks one tool-backed action:
**extract text** (OCR + font + Hi-SAM + LaMa), **fork layers** (Qwen-Image-Layered),
**split** by connected components, **detect & segment** (GroundingDINO + SAM 2 +
inpaint), or **vectorize/finalize** (VTracer; photos stay raster).

Run entrypoints **from the repository root** (the package imports as `ReDesign`;
the per-episode worker is launched as `python -m ReDesign.episode_run`).

## Entry points

| File | Purpose |
|---|---|
| `run_single_image.py` | Run the agent on a single image |
| `run_agent_figma.py` | Run on a Figma dataset directory (every episode) |
| `run_agent_crello.py` | Run on a Crello dataset directory (every `crello_test_*` record) |
| `episode_run.py` | Per-episode driver (launched as a worker subprocess) |

```bash
python -m ReDesign.run_agent_figma \
    --data_dir figma_data --output_dir outputs/figma_agent \
    --qwen_gpus <QWEN_GPU_IDS> --qwen_pair_size <N> --tool_gpus <TOOL_GPU_IDS>
```

Output per episode: `outputs/<...>/episodes/<episode_id>/` with `parse.json`
(editable hierarchy), `history_tree.json` (the decomposition tree),
reconstruction images, and logs. Completed episodes are skipped on re-run.

## Components

- `build_graph.py`, `state.py`, `reducers.py`, `registry.py` — the controller graph, tree state, and tool registry
- `nodes/` — controller/verifier nodes (router, detect, segment, inpaint, ocr, fontstyle, qwen_layered, cca, finalize, …)
- `tools/` — wrappers around the `../modules/` backends (GDINO, SAM 2, Hi-SAM, LaMa, ObjectClear, OCR, VTracer) and Qwen-Image-Layered
- `qwen_pool.py`, `qwen_worker.py`, `tool_gpu_manager.py`, `tool_gpu_config.py` — multi-GPU pooling for the Qwen model and the tools
- `reconstruction.py`, `visualizer.py`, `prompts.py`, `prompt_builders.py` — JSON export, tree visualization, controller prompts

## Requirements

- Environment: `../environment.yml` + `../post_install.sh`
- Checkpoints: `python ../scripts/download_checkpoints.py`
- `../.env` with `OPENAI_API_KEY` (the VLM controller) and, optionally, `GEMINI_API_KEY`
- GPUs: set Qwen vs. tool GPUs via `--qwen_gpus` / `--qwen_pair_size` / `--tool_gpus`
  (or `URLD_QWEN_GPUS` / `URLD_TOOL_GPUS`). The Qwen model needs ≈55 GB — see the
  top-level README "Compute & API configuration".
