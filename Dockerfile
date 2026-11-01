FROM python:3.14.5-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/models/bundle.joblib

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app

COPY requirements-serving.txt pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements-serving.txt \
    && pip install --no-cache-dir --no-deps .

COPY streamlit_app.py ./streamlit_app.py
COPY start_dashboard.py ./start_dashboard.py
COPY models/bundle.joblib ./models/bundle.joblib

USER app
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn fraud.api:app --host 0.0.0.0 --port ${PORT:-8000}"]