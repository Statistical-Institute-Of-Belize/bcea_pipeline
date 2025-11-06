# Configuration Guide

Configuration lives in `config.yaml`; CLI flags override values at runtime through `_normalize_config` in `src/utils.py`. Key sections are summarised below.

## `data`
- `raw_dir`: Folder containing the incoming labeled CSVs (default `data/raw/`).
- `processed_dir`: Destination for `train.csv`, `val.csv`, `test.csv`, and prediction exports.
- `mappings_dir`: Stores `bcea_mappings.json` (`label_to_id`/`id_to_label`).
- `review_dir`: Receives `unknown_*.csv` files when predictions fall outside the known label space.
- `reference_dir`: Holds lookup tables used to enrich predictions with industry descriptions.
- `corrections_dir`: Optional folder of reviewer corrections combined during fine-tuning.
- `max_samples`: Legacy preprocessing cap; trims rows before splitting.
- `use_business_name` / `text_separator`: Control how business names and descriptions are merged.

## `model`
- `name`: Hugging Face checkpoint (e.g., `distilroberta-base`).
- `max_seq_length`: Token length for both training and inference.
- `device`: `auto`, `cpu`, `cuda`, or `mps`.

## `training`
- Core knobs: `batch_size`, `epochs`, `learning_rate`, `gradient_accumulation_steps`, `mixed_precision`.
- Early stopping via `early_stopping_patience` (set to `0` to disable).
- Subsampling controls:
  - `subset_threshold`: Only sample when the dataset exceeds this size.
  - `subset_size`: Requested sample size; when set via CLI (`--subset-size`) the run sets `subset_force: true` to guarantee sampling.
  - `subset_seed`: Seed for deterministic draws.
- `enable_optimizations`: Turns on mixed precision and dataloader tweaks from the CLI (`--enable-optimizations`).
- `force_update_best`: Promote the latest run even if metrics degrade; also toggled via `--force-update-best`.
- `memory.*`: Runtime guards for minimum available RAM and cleanup cadence.
- `dataloader_opts.*`: Threading and prefetch hints passed to PyTorch `DataLoader`.

## `logging`
- `dir`: Where log files and HTML reports are saved.
- `level`: `INFO` by default; adjust to `DEBUG` for deeper tracing.
- `style`: `rich` enables the colourised console reporter.
- `log_batch_frequency`: Logging cadence (steps) during training.

## `output`
- `model_dir`: Parent folder for timestamped training runs.
- `best_model_dir`: Promoted weights consumed by prediction and API services.

## `prediction`
- `explain`: Store explanation artefacts when supported; also toggled with `--explain`.
- `confidence_threshold`: Minimum probability for an answer to be considered confident; low scores mark predictions as fallback in CSV outputs.

## `evaluation`
- `auto_evaluate`: Runs immediately after training unless `--skip-evaluation` is passed.
- `save_confusion_matrix`, `save_class_performance`, `save_misclassifications`: Persist supplemental reports to the run directory and `logs/`.
- `metrics_to_track`: Metrics that determine best-model promotion and appear in console summaries.
