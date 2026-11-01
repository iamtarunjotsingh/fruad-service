# Step 5–7: API, Dashboard, and Deployment

## Export a trusted model bundle

The API loads one joblib bundle containing the trained feature pipeline, calibrator, cost-tuned threshold, and metadata.
Train with real data first, then export the artifacts from the training or MLflow run:

```powershell
python scripts/run_gbt.py
python scripts/export_model.py --source files --rule cost
```

Alternatively, export a tracked run with `python scripts/export_model.py --source mlflow --rule cost`.
Synthetic runs are not exportable. The default output path is `models/bundle.joblib`.

**Security:** joblib uses pickle-based loading and can execute code. Only export, copy, and deploy a bundle you created
and trust. It contains model behavior, not the original dataset. Re-export and test the bundle when changing the Python,
scikit-learn, or LightGBM versions; its metadata records the LightGBM and scikit-learn versions used at export.
Compute its SHA-256 with `Get-FileHash models/bundle.joblib -Algorithm SHA256` and provide that digest through a
trusted deployment configuration independent of the bundle. A hash cannot establish that the original bundle is safe.

## Run locally

Install project dependencies and make the package importable:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

Start the API and dashboard in separate terminals:

```powershell
uvicorn fraud.api:app --reload
streamlit run streamlit_app.py
```

The dashboard defaults to `http://127.0.0.1:8000`; set `FRAUD_API_URL` to a different API URL if needed.
The API offers `GET /health`, `POST /predict`, and `POST /explain`. Requests require `Time`, `Amount`, and `V1`–`V28`;
unknown fields, missing fields, out-of-range time/amount values, and non-finite values are rejected.

Local development defaults to unauthenticated access. In production, set `FRAUD_ENV=production`, a strong secret
`FRAUD_API_KEY`, and the independently verified `MODEL_SHA256`. Send the secret in the `X-API-Key` header for
`/predict` and `/explain`; `/health` is public. Missing credentials, digest, or an invalid bundle leaves the service
unready. Terminate TLS and enforce request size limits, rate limits and authorization at a trusted gateway; the API key
alone does not provide per-user identity or distributed rate limiting. Rotate secrets through the deployment platform.
The demo dashboard does not send an API key and must not be publicly deployed as a privileged proxy.

The Streamlit session dashboard displays API health, scored-transaction count, flag rate, mean fraud probability, and
the per-session probability trend. These counters are in-memory and are not durable production monitoring or drift
detection. Raw transaction inputs are not retained in the dashboard's session history.

## Explanations and scoring semantics

`/predict` returns the calibrated fraud probability and a decision based on the cost-tuned threshold applied to the
uncalibrated model score. `/explain` returns LightGBM Tree SHAP contributions for the engineered model features.
Those contributions add to the raw LightGBM margin (log-odds), not the calibrated probability; they describe model
behavior and are not causal explanations. `V1`–`V28` are anonymized PCA components, so their meanings are unknown.

The current cost model assumes a missed fraud costs its transaction amount and a false alarm costs the configured
flat amount (default `$5`); the resulting threshold is selected on validation data.

The checked-in [GBDT report](results/gbt.md) records the evaluated run: 42,722 test transactions with 52 frauds,
LightGBM PR-AUC `0.751` versus `0.644` for logistic regression, and recall `0.712` at 90% precision. Under the stated
cost assumption, its validation-selected threshold was `0.0971`; on test it saved `59.1%` versus flagging nothing
(39 true positives, 30 false positives, 13 false negatives). These figures describe that recorded run; confirm that
the bundle selected for deployment came from the same run before treating them as deployed-model metrics.

## Docker

The Docker image intentionally requires a trusted exported `models/bundle.joblib` in its build context:

```powershell
docker build -t fraud-service .
docker run --rm -p 8000:8000 fraud-service
```

The container starts the API and reads `MODEL_PATH` (default `/app/models/bundle.joblib`). The same image can run the
dashboard with `python start_dashboard.py`; it uses the hosting platform's `PORT` and `FRAUD_API_URL` environment
variables. Only the serving dependencies are installed in the image. Direct serving dependencies are pinned in
`requirements-serving.txt`; verify them against each exported bundle and test the exact image. Transitive dependencies
and the base-image digest are not locked, so builds are not fully reproducible.

## Render deployment

`render.yaml` defines an API service in production mode. Push the code and trusted exported bundle to the connected
repository, then create a Render Blueprint and supply `FRAUD_API_KEY` and `MODEL_SHA256` as secret environment values.
Render credentials and account setup are external; no live service is created by this configuration. The API health
check returns `503` until the model bundle is loaded, but does not exercise scoring. The demo dashboard is not included.

Model artifacts are normally ignored except `models/bundle.joblib`, which is allowed for deployment packaging. Review
the file's provenance before adding it to a public repository; use a private repository if that better fits the
model's access requirements.

## CI and tests

The GitHub Actions workflow runs `pytest -q tests` and builds and smoke-tests the serving image for each push and pull request. To run tests locally:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q tests
```

## Production launch gates

This repository is not evidence that the model is suitable for production fraud decisions. Before launch:

- Establish a verified transaction-to-`V1`–`V28` transformation and correct time semantics, or retrain on available production features. Do not substitute zeroes or assume dataset-relative time is local time.
- Establish the deployed bundle's provenance and evaluate it on independent, representative recent data, including calibration, fraud recall, precision, costs and subgroup performance. Approve decision thresholds and a human-review process.
- Provide authenticated TLS ingress, centralized rate and request-size limits, secret rotation, and restricted dashboard access. Do not put real transaction details or secrets in logs.
- Add durable metrics and alerts for errors, latency, traffic, score distributions, drift and delayed outcomes; define incident response and rollback procedures.
- Exercise scoring and explanations in the exact deployment image with the actual bundle; implement a representative scoring readiness probe. Test backup and rollback before enabling decisions.