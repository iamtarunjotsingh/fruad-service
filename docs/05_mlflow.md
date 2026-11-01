# Step 4: Experiment Tracking with MLflow

**Goal:** record every experiment so any result can be found, compared and reproduced.
**Run:** `python scripts/run_tracked.py` or [notebooks/05_mlflow.ipynb](../notebooks/05_mlflow.ipynb).
**View:** `mlflow ui --backend-store-uri sqlite:///mlflow.db`, then open http://127.0.0.1:5000
(if `mlflow` is not found, use `python -m mlflow ui --backend-store-uri sqlite:///mlflow.db`).

## 1. The problem it solves

After steps 1 to 3 there are dozens of numbers across notebooks. Without tracking you cannot answer:
*Which model was best? With which settings? On which data and code version? Can I get it back?*
MLflow stores every run's settings, results and files in one searchable place.

## 2. Vocabulary

| Term | Meaning | Here |
|---|---|---|
| Experiment | Named group of runs | `fraud-detection` |
| Run | One recorded execution | majority class, each imbalance strategy, each LightGBM trial, the final model |
| Parameters | Settings you chose | model, strategy, seed, `num_leaves`, false-alarm cost |
| Metrics | Numbers you measured | `test_pr_auc`, `test_recall_at_90_precision`, `cost_savings_pct` |
| Tags | Context labels | git commit, data fingerprint, library versions |
| Artifacts | Files attached to a run | figures, result tables, model, calibrator |
| Nested run | Child run inside a parent | the search trials under "step3: lightgbm search" |

## 3. What gets logged

| Run | Parameters | Key metrics | Artifacts |
|---|---|---|---|
| `majority-class` | model, split, seed, row and fraud counts | test PR-AUC (equals fraud rate) | none |
| `logreg \| <strategy>` (4 runs) | model, strategy, split, seed | test PR-AUC with bootstrap CI, recall at 90% precision, Brier, mean score, tuned-threshold precision / recall / F1 / TP / FP / FN | none |
| `trial 1..N` (children of "lightgbm search") | the sampled hyperparameters | `val_pr_auc`, best iteration | none |
| `step3: lightgbm final` | best hyperparameters, false-alarm cost, search size | test PR-AUC with CI and difference vs logistic regression, alert-budget recall, cost, savings %, thresholds | figures, result CSVs, `config.json`, calibrator, **the model pipeline** |

**Naming convention:** `test_` metrics come from the test set, `val_` from validation. Test numbers are logged once per
final model and are never used to make choices.

## 4. Reproducibility tags

Every run carries:
- `git_commit` and `git_dirty`: which code made it. `git_dirty = True` means uncommitted changes existed, so the commit alone is not the full story. Commit before important runs.
- `dataset`: an md5 fingerprint of `creditcard.csv`, so you can tell if the data changed. Synthetic runs are labelled **SYNTHETIC** and go to a separate database (`mlflow_synthetic.db`).
- library versions (Python, scikit-learn, LightGBM, MLflow) and the random `seed`.

The recipe to reproduce a run: check out the commit, confirm the data fingerprint, use the logged params and seed.

## 5. Where the data is stored

- `mlflow.db`: a SQLite file with runs, params, metrics and tags.
- `mlartifacts/`: the files (figures, tables, models).

Both are in `.gitignore`. They are local and need no server. Back them up like any other results folder.
MLflow 3 recommends a database backend over the older plain-folder `mlruns/` store, which is why SQLite is used.

## 6. Using the UI

1. Open the `fraud-detection` experiment.
2. **Sort** the table by `test_pr_auc` or `cost_savings_pct`.
3. **Compare:** tick several runs, click Compare, and use the parallel-coordinates plot to see how settings relate to the metric.
4. **Drill down:** open the final run and its Artifacts tab for the PR curve, cost curve and CSV tables.
5. **Search:** use the search box, for example `metrics.test_pr_auc > 0.6 and params.model = "lightgbm"`.

## 7. The model artifact and loading it

The logged model is one pipeline (feature engineering + LightGBM), with an input example and an inferred input schema.

```python
from fraud.tracking import load_logged_model
model = load_logged_model("<run_id>")
model.predict_proba(raw_transactions)[:, 1]
```

Notes:
- **Run `pip install -e .` once** from the project root so the `fraud` package is importable. The pipeline references our feature function by name.
- **Serialization is `cloudpickle`.** MLflow 3's safer default (`skops`) refuses custom types such as our feature function unless each is listed as trusted, and that list changes with library versions. Pickle-style loading runs code from the file, so **only load models you created**.
- The **calibrator** is stored separately (`calibrator/platt_calibrator.joblib`). The API in step 5 needs both.
- `--register` registers the model in MLflow's Model Registry as `fraud-lightgbm` and creates a new version each run. Use it when you want named versions (for example "Production"); skip it while experimenting.

## 8. Good habits

- One change per run, so differences have one explanation.
- Always log the seed and the data fingerprint.
- Log test metrics only for finished candidates, not while tuning.
- Give runs meaningful names. Delete or ignore failed or synthetic runs.
- Commit before important runs so `git_dirty` is `False`.

## 9. Common problems

| Problem | Fix |
|---|---|
| `mlflow` command not found | `python -m mlflow ui --backend-store-uri sqlite:///mlflow.db` |
| Port 5000 in use | add `--port 5001` |
| UI shows no experiment | run `mlflow ui` from the project root, or pass the full `sqlite:///.../mlflow.db` path |
| `ModuleNotFoundError: fraud` when loading a model | `pip install -e .` from the project root |
| Runs look duplicated | each execution of the script is a new run. The leaderboard shows the latest run per name |

## Recorded findings

The saved notebook leaderboard identifies final LightGBM run `9be470fe02624bd0b43bb3db884b646e` with test PR-AUC 0.7514 and 59.1109% illustrative cost savings. Its paired PR-AUC difference interval versus unweighted logistic regression is +0.052 to +0.167 in the recorded experiment. The leaderboard's test ranking is retrospective, not a valid basis for selecting a model without a fresh independent evaluation.

Inspect the run's tags for its commit, dirty state and dataset fingerprint before claiming reproducibility; a run ID alone does not verify provenance or establish that the deployed bundle is the same artifact. Results remain limited to a short, historical dataset and assumed costs.
