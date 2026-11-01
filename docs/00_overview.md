# Fraud Detection Service: Project Log

This folder documents each step: what was done, why, and what the results mean.

| Step | Document | Status |
|---|---|---|
| 1a. Exploratory data analysis | [01_eda.md](01_eda.md), notebook [01_eda.ipynb](../notebooks/01_eda.ipynb) | done |
| 1b. Baselines (majority class, logistic regression) | [02_baselines.md](02_baselines.md), notebook [02_baselines.ipynb](../notebooks/02_baselines.ipynb) | done |
| 2. Imbalance handling | [03_imbalance.md](03_imbalance.md), notebook [03_imbalance.ipynb](../notebooks/03_imbalance.ipynb) | done |
| 3. Gradient-boosted trees + cost-based threshold | [04_gbt_cost.md](04_gbt_cost.md), notebook [04_gbt_cost.ipynb](../notebooks/04_gbt_cost.ipynb) | done |
| 4. MLflow tracking | [05_mlflow.md](05_mlflow.md), notebook [05_mlflow.ipynb](../notebooks/05_mlflow.ipynb) | done |
| 5. FastAPI service + tests | [06_serving_deploy.md](06_serving_deploy.md) | done |
| 6-7. Docker/CI/deploy, Streamlit demo | [06_serving_deploy.md](06_serving_deploy.md) | configured |

## How to reproduce steps 1 to 4

```bash
pip install -r requirements.txt
pip install -e .                  # makes the `fraud` package importable (needed to load logged models)
# Download creditcard.csv from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# and place it at data/creditcard.csv
jupyter lab notebooks/           # run 01_eda, 02_baselines, 03_imbalance, 04_gbt_cost, 05_mlflow, each top to bottom
pytest -q tests
```

The notebooks are the main deliverable: code, plots and written analysis in one place. They import
the reusable code from `src/fraud/`, so the same preprocessing and metrics will be used by the training
script and the API later. Figures are saved to `docs/figures/` and the baseline results to `docs/results/`.

Prefer the command line? `python scripts/eda.py` and `python scripts/run_baselines.py` `python scripts/run_imbalance.py` and `python scripts/run_gbt.py` produce the same
tables and figures without Jupyter.

## Ground rules used throughout

1. **Time-based split.** Train on the oldest 70%, validate on the next 15%, test on the newest 15%.
2. **Intended protocol:** decisions (thresholds, model choice) use validation only, and test is for final reporting. The notebooks report several models and rules on the same test window; retrospective selection from those comparisons requires a fresh holdout.
3. **Preprocessing is fit on train only** and bundled into a single sklearn `Pipeline`.
4. **No accuracy.** With 99.8% legitimate transactions, a model that never flags anything scores 99.8% accuracy.

## Tracking everything with MLflow (step 4)

```bash
python scripts/run_tracked.py
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
