# API Reference

Base URL: `http://localhost:8000`

The FastAPI service loads `models/best_model/` at startup and returns the same prediction schema as the CLI, including confidence grades and alternative codes.

## Authentication
No authentication is enforced by default. Restrict access at the ingress layer or wrap the app with your preferred auth middleware before production use.

## Endpoints

### `GET /health`
- Returns service status, model path, and model version metadata.
- Example response:
  ```json
  {
    "status": "ok",
    "model_loaded": true,
    "model_dir": "models/best_model"
  }
  ```

### `POST /predict/business`
- Predicts a BCEA code for a single record.
- Request body:
  ```json
  {
    "bus_name": "Acme Widget Manufacturing",
    "description": "Manufacturing of metal widgets and gadgets for industrial use"
  }
  ```
- Response body:
  ```json
  {
    "bus_name": "Acme Widget Manufacturing",
    "description": "Manufacturing of metal widgets and gadgets for industrial use",
    "predicted_code": "2599.0",
    "industry_description": "Manufacture of other fabricated metal products n.e.c.",
    "confidence": 0.87,
    "confidence_grade": "high",
    "alternatives": [
      {"code": "2593.0", "description": "Manufacture of wire products, chain and springs", "confidence": 0.08},
      {"code": "2591.0", "description": "Manufacture of steel drums and similar containers", "confidence": 0.03}
    ]
  }
  ```

### `POST /predict/batch`
- Accepts an array of `businesses` and returns `predictions` in the same format as the single endpoint.
- Ideal for synchronous batch jobs with <1k rows; for larger workloads, prefer streaming CSV uploads or the CLI.

### `POST /predict/csv`
- Multipart upload that ingests a CSV with `bus_name` and `description` columns.
- Returns a CSV file containing the original columns plus prediction outputs (`bcea_code`, `industry_description`, `confidence`, `confidence_grade`, `alt_*`).
- Sample request:
  ```bash
  curl -X POST "http://localhost:8000/predict/csv" \
       -H "accept: text/csv" \
        -F "file=@businesses.csv;type=text/csv" \
        -o predictions.csv
  ```

## Error Handling
- Missing columns, unreadable files, or unsupported content types trigger `400 Bad Request` responses with explanatory messages.
- Server-side failures (e.g., model unavailable) return `500 Internal Server Error`; check `logs/api.log` or the server console for stack traces.

## Versioning Strategy
- Tag releases when updating the prediction schema and publish artefact hashes in `docs/deployment.md`.
- If you introduce breaking changes, bump the API version in `api/main.py` and document the migration path here.
