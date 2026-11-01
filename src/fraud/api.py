"""FastAPI service.  Run:  uvicorn fraud.api:app --reload"""
import hmac
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .schemas import Explanation, Prediction, Transaction
from .serving import DEFAULT_BUNDLE, ModelBundle, load_bundle

logger = logging.getLogger(__name__)


def create_app(bundle: ModelBundle | None = None) -> FastAPI:
    """Pass a bundle to inject a model (tests); otherwise MODEL_PATH (default models/bundle.joblib) is loaded."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        production = os.getenv("FRAUD_ENV", "").strip().lower() == "production"
        key = os.getenv("FRAUD_API_KEY", "")
        digest = os.getenv("MODEL_SHA256", "").strip()
        app.state.api_key = key if key.strip() else None
        app.state.bundle = None
        if production and not app.state.api_key:
            logger.error("Production requires FRAUD_API_KEY")
        elif bundle is not None:
            app.state.bundle = bundle
        elif production and not digest:
            logger.error("Production requires MODEL_SHA256")
        else:
            path = Path(os.getenv("MODEL_PATH", DEFAULT_BUNDLE))
            try:
                if path.exists():
                    app.state.bundle = (load_bundle(path, expected_sha256=digest) if digest
                                        else load_bundle(path))
            except Exception:
                logger.error("Model bundle could not be loaded or verified")
        yield

    app = FastAPI(title="Transaction Fraud Detection", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def authenticate_scoring(request, call_next):
        if request.url.path in ("/predict", "/explain"):
            if app.state.bundle is None:
                return JSONResponse(status_code=503, content={"detail": "Model not loaded"})
            key = app.state.api_key
            supplied = request.headers.get("X-API-Key", "")
            if key and not hmac.compare_digest(supplied.encode("utf-8"), key.encode("utf-8")):
                return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(request, exc: RequestValidationError):
        # Avoid echoing invalid values (including NaN/Infinity) into JSON error bodies.
        detail = [{"loc": error["loc"], "msg": error["msg"], "type": error["type"]}
                  for error in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": detail})

    @app.get("/health")
    def health():
        b = app.state.bundle
        if b is None:   # 503 so a platform health check keeps a model-less container out of rotation
            return JSONResponse(status_code=503, content={"status": "unavailable", "model_loaded": False})
        return {"status": "ok", "model_loaded": True, "model_version": b.metadata.get("run_id", "unknown")}

    @app.post("/predict", response_model=Prediction)
    def predict(tx: Transaction):
        b = app.state.bundle
        if b is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        out = b.score(pd.DataFrame([tx.model_dump()])).iloc[0]
        return Prediction(fraud_probability=float(out["fraud_probability"]), raw_score=float(out["raw_score"]),
                          is_fraud=bool(out["is_fraud"]), threshold=b.threshold,
                          model_version=str(b.metadata.get("run_id", "unknown")))

    @app.post("/explain", response_model=Explanation)
    def explain(tx: Transaction):
        b = app.state.bundle
        if b is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        out = b.explain(pd.DataFrame([tx.model_dump()]))
        return Explanation(**out, model_version=str(b.metadata.get("run_id", "unknown")))

    return app


app = create_app()