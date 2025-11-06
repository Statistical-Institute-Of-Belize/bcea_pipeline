# Developer Guide

## Architecture Snapshot
- **`main.py`** orchestrates preprocessing, training, fine-tuning, prediction, and evaluation stages driven by CLI flags to stay aligned with the ISCO pipeline.
- **`src/preprocess.py`** cleans text, joins business names and descriptions, and prepares train/val/test splits with consistent label IDs.
- **`src/model.py`** wraps Hugging Face transformers for training; it reads `training.*` options (batch size, optimizations, subset sampling) from the normalized config.
- **`src/predict.py`** handles batch inference, confidence grading, and alternative prediction ranking for both CLI and API calls.
- **`api/`** exposes the FastAPI surface (`routers/predict.py`) and shares the same artifacts (`models/best_model/`) as the CLI.

## Repository Layout
```
├─ src/              # Core pipeline modules
├─ api/              # FastAPI app, routers, pydantic models
├─ data/             # raw/ processed/ mappings/ review/ reference/
├─ models/           # Timestamped runs and best_model/
├─ logs/             # Rich console and trainer logs
├─ docs/             # MkDocs content (this site)
├─ config.yaml       # Central configuration
├─ requirements.txt  # Runtime dependencies
└─ requirements-docs.txt  # MkDocs extras
```

## Coding Guidelines
- Follow PEP 8 with four-space indentation and descriptive `snake_case` identifiers; keep classes in `PascalCase`.
- Mirror existing docstrings (triple double-quotes + Args/Returns) and use `logging` helpers from `src/utils.py` for consistency.
- Store tunables in `config.yaml`; CLI flags override them through `_normalize_config` in `src/utils.py`.
- Prefer pure functions and explicit dependency injection—pass configs and paths instead of relying on globals.

## Working With Data & Models
- Preprocessing writes `train.csv`, `val.csv`, and `test.csv` under `data/processed/`; each row carries `combined_text`, `bcea_code`, and `label_id`.
- The best model promotion logic compares metrics against `models/best_model/metrics.json` unless `--force-update-best` is set.
- Subset sampling is controlled via `training.subset_size`/`subset_threshold`; the CLI flag `--subset-size` sets `subset_force` so the trainer always samples the requested size.

## Testing & Validation
- Run lightweight smoke checks with `python -m compileall main.py src api` (mirrors the existing CI sanity check).
- Add formal tests under a future `tests/` directory using `pytest`; stub out small CSV fixtures to reach preprocessing and prediction branches.
- Before promoting a build, capture metrics with `python main.py --skip-training --evaluate` and archive the generated HTML report in `logs/`.

## Extending the API
- Pydantic models live in `api/models.py`; augment them when adding fields so both CLI and API responses stay in sync.
- Shared helpers (e.g., confidence grading) should remain in `src/` and be imported by the FastAPI layer—avoid duplicating business logic inside `api/`.
- Expose new routes by wiring routers in `api/main.py` and documenting them in `docs/api.md`.
