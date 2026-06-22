# `editability_utils/`

Support library for the editability evaluation (used by
`evaluation/eval_editability_figma.py`, `eval_editability_text_figma.py`, and
`before_eval_editability_precompute_matches.py`). Imported as
`evaluation.editability_utils.*`.

## Layout

Core utility modules (imported by the evaluation scripts) live at the top level;
standalone helper/analysis scripts are grouped under `dev/`.

```
editability_utils/
├── common_utils.py        # JSON / text / IO helpers (load_json, save_json, normalize_text, ...)
├── loaders.py             # attach GT / prediction element metadata
├── matching_core.py       # GT<->prediction element matching (greedy_match_gt_to_pred, MatchConfig)
├── matching_visuals.py    # match visualization helpers
├── match_runner.py        # batch matching driver
├── task_common.py         # shared task helpers
├── task_sampling.py       # episode/task sampling
├── subset_manifest.py     # subset bookkeeping
├── joint_match_filter.py  # joint GT/pred match filtering
├── run_atomic_edit.py     # atomic-edit application (shared by subtasks/dev)
├── run_atomic_edit_radnom.py
├── subtasks/              # per atomic-edit subtask logic (delete, opacity, recolor, ...)
├── tasks/                 # higher-level editability task definitions
└── dev/                   # standalone analysis / visualization / alternative runners
                           # (not part of the deployed evaluation pipeline)
```

## Role in the pipeline

1. `before_eval_editability_precompute_matches.py` uses `matching_core` + `loaders`
   to precompute GT<->prediction element matches per model.
2. `eval_editability_figma.py` uses `subtasks/` (via `eval_editability_baselines.py`)
   to apply atomic edits and score them against those matches.

See [`../README.md`](../README.md) for how to run the full editability evaluation.
