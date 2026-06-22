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
`crello_test_<id>/composite.png`:

```
crello_data/records/
  crello_test_0001/composite.png
  crello_test_0002/composite.png
  ...
```

The Crello Parquet records store each design as layered elements; render the
composited canvas to `composite.png` per record (and keep the GT element images
+ metadata alongside if you intend to run the Crello *evaluation*). Then run:

```bash
python -m REDESIGN.run_agent_crello \
    --data_dir crello_data/records \
    --output_dir outputs/crello_agent \
    --qwen_gpus 2,3,4,5 --qwen_pair_size 2 --tool_gpus 6,7
```

> The original Crello → `composite.png` rendering follows the canvas-vae element
> compositing convention (z-ordered alpha blending of element images onto the
> canvas at the stored sizes/positions).
