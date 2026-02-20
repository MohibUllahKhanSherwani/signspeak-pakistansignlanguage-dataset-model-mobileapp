# Archive Cleanup (2026-02-20)

This folder stores non-essential files moved out of the active ML workflow.

## Why archived

- Keep `ml_pipeline_data_collection` focused on active scripts and artifacts:
  - data collection (`collect_data_gui.py`)
  - training (`train_model_with_augmentation.py`, optional `train_model.py`)
  - API (`api_server.py`)
  - minimal real-time testing (`realtime_inference_minimal.py`)

## Archived groups

- `comparison/`
  - model comparison script/reports/history logs
- `legacy_inference/`
  - older inference UIs not used in current workflow
- `debug/`
  - temporary debug dumps/images/npy files
- `old_data/`
  - old 30fps dataset backup (`MP_Data_Old_30fps`)
- `generic_models/`
  - generic model artifacts not used by baseline/augmented API path

## Restore

Move files/folders back to `ml_pipeline_data_collection` root if needed.
