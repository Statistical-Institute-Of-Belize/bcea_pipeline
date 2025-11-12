# BCEA Classification Pipeline

Fast-track BCEA code prediction for Belizean business registrations using a fine-tuned transformer model. The project mirrors the ISCO pipeline interface so teams can swap between them with minimal changes.
[Full docs](https://statistical-institute-of-belize.github.io/bcea_pipeline/)

## Quick Start (CLI)
- Create an isolated environment and install dependencies:
  ```bash
  python -m venv venv && source venv/bin/activate
  pip install -r requirements.txt
  ```
- Place your labeled CSV (`bus_name`, `description`, `bcea_code`) under `data/raw/` and confirm paths in `config.yaml`.
- Train, evaluate, and promote a model in one step:
  ```bash
  python main.py --config config.yaml
  ```
- Fine-tune with reviewer corrections or sample a large dataset:
  ```bash
  python main.py --fine-tune --corrections-dir data/corrections
  python main.py --config config.yaml --subset-size 50000
  ```
  Passing `--subset-size` forces a random draw of that many records regardless of the configured threshold.
- Reuse the best checkpoint for scoring only:
  ```bash
  python main.py --skip-training --input data/new_businesses.csv --skip-evaluation
  ```
  A timestamped CSV with prediction details is written to `data/processed/`.

## Quick Start (API)
- Launch the FastAPI service (loads `models/best_model/` by default):
  ```bash
  python api_server.py
  ```
- Hit the live docs at `http://localhost:8000/docs` or issue a direct call:
  ```bash
  curl -X POST http://localhost:8000/predict/business \
       -H 'Content-Type: application/json' \
       -d '{"bus_name": "Acme Ltd", "description": "Fresh seafood wholesale"}'
  ```
  Responses include the predicted code, confidence grade, and two ranked alternatives.

## Frequently Used Flags
- `--skip-training` / `--evaluate` / `--skip-evaluation` – control which stages run.
- `--subset-size N` – down-sample the training set to `N` rows (always applied when provided).
- `--enable-optimizations` – opt into mixed precision and training speed-ups.
- `--force-update-best` – promote the latest run even if metrics regress.
- `--fine-tune` and `--corrections-dir` – merge correction CSVs into a follow-up training pass.
- `--explain` – request explanation artifacts when available.

## Documentation
Browse Markdown sources under `docs/` or serve the full site with MkDocs:
```bash
pip install -r requirements-docs.txt
mkdocs serve
```
Static HTML builds land under `site/` when you run `mkdocs build`.
