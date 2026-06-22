# Crello Dataset

ReDesign uses the **Crello dataset** from CyberAgent AI Lab for the Crello design
benchmark. We do **not** redistribute it here — please download it from the
official source.

## 1. Download the raw Crello dataset

Follow the official CyberAgent canvas-vae instructions:

> https://github.com/CyberAgentAILab/canvas-vae/blob/main/docs/crello-dataset.md

The dataset is published on the HuggingFace Hub as
[`cyberagent/crello`](https://huggingface.co/datasets/cyberagent/crello)
(Parquet shards: `train-*`, `validation-*`, `test-*`).

Quick download:

```bash
# Option A — HuggingFace CLI
hf download cyberagent/crello --repo-type dataset --local-dir crello_data

# Option B — Python
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="cyberagent/crello", repo_type="dataset",
                  local_dir="crello_data")
PY
```

Please cite the Crello dataset and follow its license/terms as described on the
pages above.

## 2. Render records for the agent

The ReDesign agent consumes one rendered canvas per design as
`crello_test_<id>/composite.png`. Build these records from the downloaded
Parquet shards with the bundled script:

```bash
# Render every test record (composite.png + elements/ + gt_metadata.json)
python scripts/prepare_crello_records.py \
    --parquet-glob "crello_data/data/test-*.parquet" \
    --output-dir crello_data/records

# Or a quick smoke test (first 3 records)
python scripts/prepare_crello_records.py \
    --parquet-glob "crello_data/data/test-*.parquet" \
    --output-dir crello_data/records --limit 3
```

This produces, per design:

```
crello_data/records/
  crello_test_0000/
    composite.png        # full composited canvas (agent input)
    elements/            # per-element GT RGBA images
    gt_metadata.json     # GT metadata (used by the Crello evaluation)
  ...
```

(z-ordered alpha blending of element images onto the canvas at the stored
sizes/positions, following the canvas-vae compositing convention.)

Then run the agent (replace `<QWEN_GPU_IDS>` / `<TOOL_GPU_IDS>` with your own
comma-separated GPU ids, e.g. `0,1`):

```bash
python -m ReDesign.run_agent_crello \
    --data_dir crello_data/records \
    --output_dir outputs/crello_agent \
    --qwen_gpus <QWEN_GPU_IDS> --qwen_pair_size 2 --tool_gpus <TOOL_GPU_IDS>
```
