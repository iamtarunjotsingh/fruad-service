# Run locally with Docker

These instructions run the API in a local Docker container using PowerShell. The image does not start the Streamlit dashboard by default.

## Prerequisites

- Install Docker Desktop and start its Docker engine.
- Open PowerShell in the project root, where the `Dockerfile` is located.
- Ensure `models\bundle.joblib` exists and is a trusted exported model. The image build copies this file; if it is missing, see [model export instructions](06_serving_deploy.md#export-a-trusted-model-bundle). Joblib files can execute code when loaded, so never use an untrusted bundle.

## Build and run

```powershell
docker build -t fraud-service .
docker run -d --name fraud-service -p 127.0.0.1:8000:8000 fraud-service
```

The port binding limits access to this computer. This default local configuration does **not** require an API key; do not expose it publicly. If port 8000 is in use, use `-p 127.0.0.1:8001:8000` and replace `8000` in the URLs below with `8001`.

## Verify the API

Allow a few moments for the model to load, then check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

A ready service returns `status: ok` and `model_loaded: true`. The health check verifies loading, not scoring. To exercise scoring with artificial values:

```powershell
$transaction = @{ Time = 0; Amount = 10 }
1..28 | ForEach-Object { $transaction["V$_"] = 0 }
$body = $transaction | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType 'application/json' -Body $body
```

The response includes `fraud_probability`, `raw_score`, `is_fraud`, `threshold`, and `model_version`. The zero-valued PCA inputs are for a connectivity demonstration only, not realistic transactions or a basis for real fraud decisions. Interactive API documentation is available at <http://127.0.0.1:8000/docs>.

## Optional: authenticated local run

To exercise production-mode checks, first stop and remove the existing container, then set a strong secret and the SHA-256 digest of the trusted bundle used in the build. Verify the digest independently; a hash does not make an untrusted pickle safe.

```powershell
docker stop fraud-service
docker rm fraud-service
$apiKey = 'replace-with-a-long-random-secret'
$digest = (Get-FileHash models\bundle.joblib -Algorithm SHA256).Hash
docker run -d --name fraud-service -p 127.0.0.1:8000:8000 -e FRAUD_ENV=production -e "FRAUD_API_KEY=$apiKey" -e "MODEL_SHA256=$digest" fraud-service
```

Rebuild the image if the bundle has changed since the build; otherwise the digest will not match the embedded file. Repeat the scoring request above with the key:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType 'application/json' -Headers @{ 'X-API-Key' = $apiKey } -Body $body
```

`/health` remains public. The dashboard currently does not send an API key, so it cannot score against this authenticated API without changes. API keys and hashes alone are not sufficient for public production deployment; see [deployment safeguards](06_serving_deploy.md#production-launch-gates).

## Logs and cleanup

```powershell
docker logs fraud-service
docker stop fraud-service
docker rm fraud-service
```

If the build cannot find the bundle, export it before building. If `/health` returns 503, inspect the logs and check the bundle, digest, and production configuration. If the container name is already in use, stop and remove the earlier container or choose another name.