# BCEA Pipeline Documentation

## Overview
The BCEA Classification Pipeline predicts 5-digit Belize Classification of Economic Activities (BCEA) codes from business descriptions. It reuses the ISCO pipeline workflow—preprocessing, transformer training, evaluation, and FastAPI serving—while adding BCEA-specific datasets and label mappings.

## What You Get
- Unified CLI with preprocessing, training, fine-tuning, evaluation, and prediction stages.
- Configurable sub-sampling (`--subset-size`) so large training corpora remain tractable.
- Shared artifacts: processed splits under `data/processed/`, promoted weights in `models/best_model/`, rich logs in `logs/`.
- FastAPI service mirroring CLI outputs, including confidence grades and alternative codes.

## Start Here
1. Review the **User Guide** for environment setup, CLI recipes, and prediction exports.
2. Visit the **Developer Guide** to understand the module layout, coding guidelines, and how to extend the pipeline safely.
3. Consult the **API Reference** for request/response schemas when integrating external systems.

## Helpful Commands
```bash
python main.py --config config.yaml          # full training run
python main.py --skip-training --input ...   # batch scoring
python main.py --fine-tune --corrections-dir data/corrections
python api_server.py                         # start REST API
mkdocs serve                                 # preview this site locally
```

## Additional Resources
- Configuration options: `config.yaml` and the [Configuration guide](configuration.md).
- Deployment tips for automating training and hosting: [Deployment](deployment.md).
- Existing API payload examples: [users.md](users.md) and [api.md](api.md).
