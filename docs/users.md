# User Guide

## Environment Setup
- Install Python 3.10+ and git.
- Create a virtual environment and install dependencies:
  ```bash
  python -m venv venv && source venv/bin/activate
  pip install -r requirements.txt
  ```
- If your data lives elsewhere, update the `data.*` paths inside `config.yaml`.

## Data Requirements
- Training CSV must include `bus_name`, `description`, and `bcea_code` columns.
- Place raw files in `data/raw/`; preprocessing writes cleaned splits into `data/processed/` (`train.csv`, `val.csv`, `test.csv`).
- Reference lookups (industry descriptions, mappings) live in `data/reference/` and `data/mappings/`; keep both synced with the latest BCEA catalogue.

## Running the CLI
- Full workflow (preprocess → train → evaluate):
  ```bash
  python main.py --config config.yaml
  ```
- Train on a manageable sample when the dataset is huge:
  ```bash
  python main.py --config config.yaml --subset-size 50000
  ```
  The subset is always enforced when the flag is supplied, regardless of `training.subset_threshold`.
- Fine-tune with reviewer corrections (CSV files in `data/corrections/`):
  ```bash
  python main.py --fine-tune --corrections-dir data/corrections
  ```
- Force evaluation of the current promoted model without retraining:
  ```bash
  python main.py --skip-training --evaluate
  ```

## Generating Predictions
- Score a new file without retraining:
  ```bash
  python main.py --skip-training --input data/new_businesses.csv --skip-evaluation
  ```
- Output columns include `predicted_code`, `industry_description`, `confidence`, `confidence_grade`, `is_fallback`, and two `alt_*` alternatives. Results are timestamped and saved under `data/processed/`.
- Enable explanations (if supported by the model checkpoint) with `--explain`.

## FastAPI Service
- Start the server with `python api_server.py`; open `http://localhost:8000/docs` for Swagger UI.
- Live endpoints mirror the CLI outputs: `/predict/business`, `/predict/batch`, `/predict/csv` (multipart upload).
- Responses contain the original input, predicted code, industry description, probability, graded confidence, and ranked alternatives.

## Troubleshooting
- If `models/best_model/` is missing, run a training job before serving predictions.
- Memory warnings during training usually indicate the batch size is too high—lower `training.batch_size` or increase `gradient_accumulation_steps`.
- When the CLI appears to ignore your subset request, ensure you passed `--subset-size`; setting only `training.subset_size` in the config respects the `subset_threshold`.
