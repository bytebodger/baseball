# baseball

A machine learning project that predicts MLB game outcomes from pitch-level history. Raw Statcast/pitch-by-pitch data (via `pybaseball`) is aggregated into game- and at-bat-level features, used to train models (scikit-learn and PyTorch) that estimate win probability and other outcome distributions. The goal is an end-to-end pipeline: ingest pitch data, build training sets, train and evaluate models, and serve predictions for upcoming games.

## Project layout

- `data/raw/` — untouched source data pulled from external APIs
- `data/processed/` — cleaned/feature-engineered datasets
- `src/data/` — data ingestion and preprocessing code
- `src/models/` — model definitions
- `src/training/` — training loops and experiment scripts
- `src/inference/` — prediction/serving code
- `configs/` — YAML configuration files
- `notebooks/` — exploratory analysis
- `tests/` — unit tests

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate  # or `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
pip install -e .
```
