# Credit Card Fraud Detection Service

An end-to-end fraud detection demonstration, from exploratory analysis and model evaluation to a FastAPI scoring service and Streamlit dashboard. The project uses the anonymized [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), chronological splits, LightGBM, cost-sensitive thresholds, and MLflow experiment tracking.

**This is an interview and research demo, not a validated production fraud decision system.** The input features are dataset-specific PCA components; real transactions cannot be scored without a compatible feature transformation or a retrained model.

## Results

The recorded experiment evaluated 42,722 transactions, including 52 frauds, in a short historical test window:

| Model | Test PR-AUC | Recall at 90% precision |
|---|---:|---:|
| Logistic regression | 0.644 | 0.423 |
| LightGBM | 0.751 | 0.712 |

Under illustrative costs, the validation-selected LightGBM threshold saved 59.1% versus flagging nothing. These figures describe a recorded experiment, not necessarily the deployed bundle; the small test window and reused validation period limit generalization. See the [model report](docs/04_gbt_cost.md) and [recorded results](docs/results/gbt.md).

## Quick start: Docker

Install and start Docker Desktop, and ensure a **trusted** `models/bundle.joblib` exists. If needed, follow the [model export instructions](docs/06_serving_deploy.md#export-a-trusted-model-bundle). Joblib files can execute code when loaded; never use an untrusted bundle.

From the project root in PowerShell:

```powershell
docker build -t fraud-service .
docker run -d --name fraud-service -p 127.0.0.1:8000:8000 fraud-service
Invoke-RestMethod http://127.0.0.1:8000/health
```

Open <http://127.0.0.1:8000/docs> for interactive API documentation. The default local API is unauthenticated; do not expose it publicly. For a sample prediction, authentication, troubleshooting, and cleanup, see [local Docker instructions](docs/local_docker.md). The container starts the API, not the dashboard.

## Run without Docker

Python 3.10 or newer and a trusted model bundle are required. From the project root:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
uvicorn fraud.api:app --reload
```

In a separate terminal, start the dashboard:

```powershell
streamlit run streamlit_app.py
```

The dashboard connects to `http://127.0.0.1:8000` by default; set `FRAUD_API_URL` to change it. It does not send an API key and therefore cannot score against the authenticated production-mode API.

## API

- `GET /health` checks whether the model loaded; it does not test scoring.
- `POST /predict` returns a calibrated probability, raw score, threshold, and decision.
- `POST /explain` returns feature contributions, not causal explanations.

Requests use `Time`, `Amount`, and `V1`–`V28`. See the [serving guide](docs/06_serving_deploy.md) for scoring semantics and configuration.

## Reproduce the analysis

Download `creditcard.csv` from the [dataset page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) into `data/creditcard.csv`, install dependencies as above, and run the notebooks in `notebooks/` in order. The [project overview](docs/00_overview.md) links the analysis, scripts, and MLflow instructions.

Run the tests with:

```powershell
python -m pytest -q tests
```

## Deployment and limitations

`render.yaml` configures an API-only Render deployment. Production mode requires `FRAUD_ENV=production`, a secret `FRAUD_API_KEY`, and the independently verified `MODEL_SHA256`; scoring requests use the `X-API-Key` header. See the [deployment guide](docs/06_serving_deploy.md#render-deployment) for setup and [production launch gates](docs/06_serving_deploy.md#production-launch-gates). The dashboard is not included in the blueprint.

Before any real-world use, validate the deployed artifact on representative data, establish the feature transformation, and add appropriate access controls, monitoring, and rollback procedures. Use synthetic data for demonstrations.

## Project layout

- `notebooks/` and `docs/` — analysis, results, and guides.
- `src/fraud/` — data preparation, modeling, evaluation, and API.
- `scripts/` — analysis, training, tracking, and model export.
- `tests/` — automated tests.
- `Dockerfile` and `render.yaml` — container and cloud deployment configuration.