# `ReDesign/` — the agent

The ReDesign agent: a recursive, queue-based agentic pipeline that decomposes a
design image into editable layers/elements by orchestrating vision tools
(detection, segmentation, inpainting, OCR, layered generation) under a VLM
controller.

Run entrypoints **from the repository root** (the package is imported as
`ReDesign`, and the per-episode worker is launched as
`python -m ReDesign.episode_run`).

## Entry points

| File | Purpose |
|---|---|
| `run_agent_figma.py` | Run the agent on a Figma dataset directory (all episodes; split-agnostic) |
| `run_agent_crello.py` | Run the agent on a Crello dataset directory (all `crello_test_*` records) |
| `run_single_image.py` | Run the agent on a single image |
| `episode_run.py` | Per-episode driver (invoked as a worker subprocess) |

```bash
python -m ReDesign.run_agent_figma \
    --data_dir figma_data --output_dir outputs/figma_agent \
    --qwen_gpus <QWEN_GPU_IDS> --qwen_pair_size 2 --tool_gpus <TOOL_GPU_IDS>
```

Output per episode: `outputs/<...>/episodes/<episode_id>/parse.json`,
`history_tree.json`, reconstruction images, and logs. Completed episodes are
skipped on re-run (resume support).

## Components

- `nodes/` — graph nodes (detect/segment/inpaint/ocr/fontstyle/qwen_layered/…)
- `tools/` — tool wrappers around `modules/` backends (GDINO, SAM2, Hi-SAM, LaMa, ObjectClear, OCR, vtracer, Qwen-Image-Layered)
- `build_graph.py`, `state.py`, `reducers.py`, `registry.py` — pipeline graph, state, tool registry
- `reconstruction.py`, `visualizer.py`, `prompts.py`, `prompt_builders.py`
- `qwen_pool.py`, `qwen_worker.py`, `tool_gpu_manager.py`, `tool_gpu_config.py` — multi-GPU pooling for the Qwen layered model and the vision tools

## Requirements

- Environment from `../environment.yml` + `../post_install.sh`
- Checkpoints from `python ../scripts/download_checkpoints.py`
- `../.env` with `OPENAI_API_KEY` (VLM router) and, optionally, `GEMINI_API_KEY`
- GPUs: configure Qwen vs. tool GPUs via `--qwen_gpus` / `--tool_gpus` or the
  `URLD_QWEN_GPUS` / `URLD_TOOL_GPUS` environment variables.
