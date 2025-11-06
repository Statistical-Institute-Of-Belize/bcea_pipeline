# Deployment Guide

## CLI Automation
- Wrap training commands in a scheduler (cron, Airflow, GitHub Actions) using the same flags you run locally:
  ```bash
  python main.py --config config.yaml --subset-size 75000 --skip-evaluation
  python main.py --skip-training --evaluate
  ```
- Export `PYTHONPATH=$(pwd)` before invoking scheduled tasks to ensure `src/` modules resolve correctly.
- Archive the generated run directory under `models/run-*/` alongside the HTML evaluation report for traceability.

## Promoting Models
- Successful runs copy metrics and artifacts into `models/best_model/` unless you disable promotion.
- Use `--force-update-best` in automation only when you have upstream regression guards.
- Record the promoted commit hash, config, and metrics in your release notes (consider storing in `logs/promotion_history.json`).

## Serving the API
- Run the packaged server directly:
  ```bash
  python api_server.py --port 8000
  ```
  or with Uvicorn for live reload in development:
  ```bash
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
  ```
- Containerise the service by copying `requirements.txt`, `api/`, `src/`, `config.yaml`, and `models/best_model/`. Expose port 8000 and mount `logs/` for rotating access logs.

## Monitoring & Logging
- Access CLI and trainer logs under `logs/`; rotate or export them to your observability stack.
- API responses include confidence scores—aggregate them to monitor prediction drift. Low average confidence suggests retraining or adjusting the subset size.
- Watch the `data/review/unknown_*.csv` folder for codes outside the known mapping; feed them back into the correction workflow.

## Continuous Documentation
- Rebuild and publish the MkDocs site whenever documentation changes:
  ```bash
  pip install -r requirements-docs.txt
  mkdocs build
  ```
- Commit the contents of the `site/` folder to a `gh-pages` branch or enable GitHub Pages with an Actions workflow similar to the ISCO pipeline repository.

## Security Considerations
- Treat `config.yaml` as non-secret but avoid embedding credentials; use environment variables for buckets or databases.
- Sanitise datasets before copying models across networks; BCEA codes may fall under regulatory review.
- Review dependencies via `pip install -r requirements.txt --require-hashes` when locking down production builds.
