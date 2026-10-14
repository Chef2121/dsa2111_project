# Radview - one image, three roles (selected by the container command):
#   pipeline : python -m radview.pipeline [--mode full|rescore]
#   api      : uvicorn radview.api.main:app --host 0.0.0.0 --port 8000
#   dashboard: streamlit run dashboard/app.py --server.port 8501
#
# Multi-stage: `test` runs the pytest suite and the runtime image copies a marker from it,
# so an image can only be built from code that passes its tests.

# ---------------------------------------------------------------- base: deps + code
FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    RADVIEW_DATA_DIR=/data

WORKDIR /app

# libgomp is needed by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# PyPI stays the primary index (torch's own deps resolve there); "+cpu" pins the CPU wheel,
# which also satisfies the plain torch==2.7.0 pin in requirements.txt.
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.7.0+cpu \
 && pip install -r requirements.txt

# Bake the DistilBERT base weights into the image so pods start without internet access.
RUN python -c "from transformers import AutoModel, AutoTokenizer; \
AutoTokenizer.from_pretrained('distilbert-base-uncased'); AutoModel.from_pretrained('distilbert-base-uncased')"

COPY radview/ radview/
COPY dashboard/ dashboard/
COPY .streamlit/ .streamlit/

# ---------------------------------------------------------------- test: the CI gate
FROM base AS test
COPY tests/ tests/
COPY pytest.ini .
RUN pip install pytest==8.3.5 httpx==0.28.1 \
 && RADVIEW_DATA_DIR=/tmp/radview-test python -m pytest -q \
 && touch /tmp/tests-passed

# ---------------------------------------------------------------- runtime
FROM base AS runtime
COPY --from=test /tmp/tests-passed /tmp/tests-passed
# Run as an unprivileged user; /data is the shared volume (PVC in Kubernetes).
RUN useradd --create-home --uid 1000 radview && mkdir -p /data && chown -R radview /data /app /opt/hf-cache
USER radview
VOLUME ["/data"]
EXPOSE 8000 8501

CMD ["uvicorn", "radview.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
