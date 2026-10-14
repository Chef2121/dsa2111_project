# Radview — agentic early warning for restaurant reputation

**DSA2111 Group 10** · Yeo Zhang Han · Daniel Leong · Peh Xue Ying · Yang Jiajun Joven · Wong Wei Xuen · Vijayanand Jeeva

Restaurant owners usually find out about a reputation problem *after* their star rating has dropped.
Radview reads every review, works out **which aspect** (food, service, cleanliness, price, ambience) customers
are starting to complain about, **forecasts** whether the rating will fall over the next three months, and has an
agent write a **ranked action plan in which every claim is backed by a quoted review** — verified automatically
before the owner sees it.

```
              ┌─────────────────────────── Kubernetes (Docker containers) ───────────────────────────┐
 Yelp Open    │  M1 Ingestion ──► M2 Aspect classifier ──► M3 Trend detector ──► M4 Decline forecaster│
 Dataset  ───►│  stream 5 GB,     DistilBERT (PyTorch),    EWMA + robust z      GRU (PyTorch) + SHAP  │
 SemEval-2014 │  clean, dedupe    weak supervision          per aspect           calibrated risk       │
              │                          │                                             │               │
              │                          └──── evidence ───►  M5 Agent (LangGraph) ◄───┘               │
              │                                               gather → prioritise → retrieve →         │
              │  Weekly CronJob ── rescore ──►                draft (Claude / rules) → verify → plan   │
              │                                                         │                              │
              │                                      M6 FastAPI ◄───────┘──► Streamlit dashboard        │
              └──────────────────────────────────────────────────────────────────────────────────────┘
```

| Module | Implementation | Key file |
|---|---|---|
| **M1** Ingestion | Streams the Yelp JSON-lines files line by line, keeps restaurants, cleans text, removes duplicates and the partial last month, writes parquet | [`radview/m1_ingest/ingest.py`](radview/m1_ingest/ingest.py) |
| **M2** Aspect sentiment | Seed-lexicon + star-rating **weak supervision** → TF-IDF/LogReg baseline and a **fine-tuned DistilBERT** (PyTorch, one 3-way head per aspect, int8-quantised for CPU inference); evaluated on the human-labelled **SemEval-2014** restaurant set | [`radview/m2_aspects/`](radview/m2_aspects/) |
| **M3** Complaint trends | Monthly complaint rate per aspect, EWMA-smoothed, robust z-score against the restaurant's *own* trailing 12 months | [`radview/m3_trends/detector.py`](radview/m3_trends/detector.py) |
| **M4** Decline forecast | Target: rating over the next 3 months falls ≥ 0.3★ below the trailing 6 months **and** the drop is statistically significant. **PyTorch GRU ensemble** over 12-month feature sequences, Platt-calibrated, explained with **SHAP** (GradientExplainer); compared with LogReg / LightGBM baselines on an out-of-time split | [`radview/m4_forecast/`](radview/m4_forecast/) |
| **M5** Agent | **LangGraph** workflow calling tools (`query_metrics`, `run_forecast`, `get_evidence_reviews`, `compare_peers`); drafts with **Claude** when `ANTHROPIC_API_KEY` is set, otherwise a deterministic planner; a **verifier** rejects any claim whose quote is not verbatim in a cited review, is not itself a complaint about that aspect, or whose numbers don't match the metrics | [`radview/m5_agent/`](radview/m5_agent/) |
| **M6** Serving | FastAPI REST API + Streamlit dashboard, one Docker image, Kubernetes Job / CronJob / Deployments | [`radview/api/`](radview/api/), [`dashboard/`](dashboard/), [`k8s/`](k8s/) |

Results from the latest run are in **[docs/RESULTS.md](docs/RESULTS.md)** (regenerate with `python scripts/make_results.py`).

---

## Quick start (local)

Requires Python 3.10+. A GPU is optional — everything runs on CPU.

```bash
pip install -r requirements-dev.txt

# 1. data: Yelp (~5.9 GB from Kaggle) + SemEval-2014 benchmark
python scripts/download_data.py

# 2. full pipeline: ingest → train/evaluate all models → score → action plans
python -m radview.pipeline            # ~2 h on CPU with DistilBERT; add --fast for TF-IDF only (~5 min)

# 3. dashboard (reads the outputs in-process) …
streamlit run dashboard/app.py        # http://localhost:8501
# … and/or the REST API
uvicorn radview.api.main:app --port 8000   # http://localhost:8000/docs
```

**No Yelp download?** `RADVIEW_SOURCE=synthetic python -m radview.pipeline --fast` generates a Yelp-format
synthetic dataset (220 restaurants, with injected deterioration episodes and known ground truth) and runs the whole
system on it in a few minutes. Outputs for each source live in `data/work/<source>/` so they never overwrite each other.

**LLM-drafted plans:** export `ANTHROPIC_API_KEY=…` (default model `claude-opus-5-5`, override with
`RADVIEW_LLM_MODEL`). Without a key the agent uses its deterministic planner; the verifier runs either way.

### Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `RADVIEW_SOURCE` | `yelp` if `data/yelp/` has the files, else `synthetic` | which dataset to use |
| `RADVIEW_DATA_DIR` | `./data` | root for raw data, models and outputs (`/data` in containers) |
| `RADVIEW_CITY` / `RADVIEW_MAX_RESTAURANTS` | `Philadelphia` / `400` | Yelp slice: the N most-reviewed restaurants in a city |
| `RADVIEW_START_DATE` | `2016-01-01` | ignore older reviews |
| `ANTHROPIC_API_KEY`, `RADVIEW_LLM_MODEL` | – / `claude-opus-5-5` | enable Claude in M5 (and the zero-shot M2 baseline) |
| `RADVIEW_API_URL` | – | dashboard reads from this API instead of in-process |

Model and threshold settings (decline threshold, horizon, spike z, time split…) are documented in
[`radview/config.py`](radview/config.py).

---

## Docker & Kubernetes

```bash
docker build -t radview:latest .

# docker compose: build data/models once, then serve
docker compose --profile jobs run --rm pipeline
docker compose up -d api dashboard          # API :8000 · dashboard :8501
```

Kubernetes (any cluster — Docker Desktop, minikube, kind, EKS/GKE/AKS):

```bash
kubectl apply -f k8s/00-namespace-config.yaml
# optional - Claude drafting:  kubectl -n radview create secret generic radview-secrets --from-literal=ANTHROPIC_API_KEY=...
# real data: copy data/yelp + data/semeval onto the PVC and set RADVIEW_SOURCE=yelp in the ConfigMap
kubectl apply -f k8s/10-pipeline-jobs.yaml   # Job radview-train + weekly CronJob radview-rescore
kubectl apply -f k8s/20-services.yaml        # API + dashboard Deployments/Services
kubectl -n radview port-forward svc/radview-dashboard 8501:80
```

| Object | Role |
|---|---|
| `Job/radview-train` | `python -m radview.pipeline --mode full` — trains and evaluates every model |
| `CronJob/radview-rescore` | Mondays 03:00: ingest new reviews, re-score with saved models, regenerate plans for at-risk restaurants |
| `Deployment/radview-api` | FastAPI, health-checked, reads the shared volume |
| `Deployment/radview-dashboard` | Streamlit, stateless (talks to the API), 2 replicas behind a LoadBalancer |
| `PersistentVolumeClaim/radview-data` | shared data/models/outputs volume |

---

## REST API

| Method & path | Returns |
|---|---|
| `GET /restaurants` | all monitored restaurants ranked by decline risk (filter `risk_level=high`) |
| `GET /restaurants/{id}` | metrics, forecast + SHAP attribution, peer comparison |
| `GET /restaurants/{id}/timeline` | monthly ratings, complaint rates, spikes, risk |
| `GET /restaurants/{id}/reviews?aspect=service&sentiment=negative` | review explorer |
| `GET /restaurants/{id}/plan?refresh=true` | the agent's verified action plan |
| `GET /evaluation`, `GET /evaluation/m4/curves` | M2–M5 evaluation results, PR/ROC curves |
| `POST /feedback` | owner usefulness rating (1–5) for a plan |

Interactive docs at `/docs`.

---

## Evaluation design

* **M2** — macro-F1 over {not mentioned, positive, negative} per aspect, plus complaint-F1 (negative class),
  on SemEval-2014 Task 4 restaurants (800 human-labelled sentences; food/service/price/ambience). Baselines: the
  lexicon rule itself, TF-IDF + logistic regression, and zero-shot Claude (when a key is set).
  `data/labels/yelp/to_label.csv` is exported for an in-domain hand-labelled set covering cleanliness as well.
* **M3** — precision/lift of complaint spikes for a decline in the next 3 months, share of decline onsets
  preceded by a spike, and lead time; compared with an equivalent alarm on the star rating itself.
* **M4** — AUROC, PR-AUC (the honest metric for a rare outcome), precision/recall in the top-10% riskiest
  restaurant-months, and early warning at a fixed 15% alert budget; strict time-based split, features use past
  data only (unit-tested).
* **M5** — groundedness = share of drafted claims that pass the verifier; owner usefulness ratings are
  collected from the dashboard (`POST /feedback`).

## Tests

```bash
pytest            # unit tests + API smoke tests (API tests need a pipeline run)
```

## Repository layout

```
radview/
  config.py            paths, constants, thresholds
  lexicon.py           seed lexicons for weak supervision & evidence
  panel.py             monthly restaurant panel shared by M3/M4
  m1_ingest/           ingest.py (Yelp streaming ETL), synthetic.py (Yelp-format simulator)
  m2_aspects/          weak_labels.py, models.py (lexicon, TF-IDF, DistilBERT, zero-shot LLM), train.py
  m3_trends/           detector.py (EWMA + robust z, evaluation)
  m4_forecast/         features.py (leak-free features & label), model.py (PyTorch GRU, SHAP), train.py
  m5_agent/            tools.py, agent.py (LangGraph), verifier.py, playbook.py
  service.py           read-side service layer used by API and dashboard
  api/main.py          FastAPI app
  pipeline.py          end-to-end orchestration (full / rescore)
dashboard/             Streamlit app + API client
k8s/                   Kubernetes manifests
scripts/               data download, results report
tests/                 pytest suite
docs/RESULTS.md        latest evaluation tables
```

## Data & licences

The Yelp Open Dataset is used under Yelp's Dataset User Agreement (academic use) and is **not** included in this
repository. SemEval-2014 Task 4 data is used for evaluation only.
