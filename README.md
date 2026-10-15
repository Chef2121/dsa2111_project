# Radview — agentic early warning for restaurant reputation

**DSA2111 Group 10** · Yeo Zhang Han · Daniel Leong · Peh Xue Ying · Yang Jiajun Joven · Wong Wei Xuen · Vijayanand Jeeva

Restaurant owners usually find out about a reputation problem *after* their star rating has dropped.
Radview reads every review, works out **which aspect** (food, service, cleanliness, price, ambience) customers
are starting to complain about, **forecasts** whether the rating will fall over the next three months, and has an
agent write a **ranked action plan in which every claim is backed by a quoted review** — verified mechanically
before the owner sees it.

```
              ┌─────────────────────────── Kubernetes (Docker containers) ───────────────────────────┐
 Yelp Open    │  M1 Ingestion ──► M2 Aspect classifier ──► M3 Trend detector ──► M4 Decline forecaster│
 Dataset  ───►│  stream 5 GB,     DistilBERT (PyTorch),    EWMA + robust z      GRU (PyTorch), market │
 SemEval-2014 │  clean, dedupe    weak supervision          per aspect           features, SHAP +      │
              │                          │                                       counterfactual        │
              │                          └──── evidence ───►  M5 Agent (LangGraph) ◄───┘               │
              │                                               Claude calls the tools itself → plan →   │
              │  Weekly CronJob ── rescore ──►                mechanical verifier (5 check axes)       │
              │                                                         │                              │
              │                                      M6 FastAPI ◄───────┘──► Streamlit dashboard        │
              └──────────────────────────────────────────────────────────────────────────────────────┘
```

| Module | Implementation | Key file |
|---|---|---|
| **M1** Ingestion | Streams the Yelp JSON-lines files line by line, keeps one city's restaurants, cleans text, removes duplicates and the partial last month, writes parquet. A preflight CLI suggests the city slice. | [`radview/m1_ingest/`](radview/m1_ingest/) |
| **M2** Aspect sentiment | Seed-lexicon + star-rating **weak supervision** → TF-IDF/LogReg baseline and a **fine-tuned DistilBERT** (PyTorch, one 3-way head per aspect, int8-quantised CPU inference). Aspect polarity comes **from the text**, not from the star rating. Evaluated on the human-labelled **SemEval-2014** restaurant set. | [`radview/m2_aspects/`](radview/m2_aspects/) |
| **M3** Complaint trends | Monthly complaint rate per aspect, EWMA-smoothed, robust (median/MAD) z-score against the restaurant's *own* trailing 12 months, with a minimum-mention guard. | [`radview/m3_trends/detector.py`](radview/m3_trends/detector.py) |
| **M4** Decline forecast | Target: next-3-month rating falls ≥ 0.3★ below the trailing 6 months **and** the drop is statistically significant. **PyTorch GRU ensemble** over 12-month sequences incl. **market-relative** features (vs. the city median that month), Platt-calibrated. Explained two ways — **SHAP** (GradientExplainer) and a **counterfactual** ("risk if this aspect were at the market norm") — plus a ridge **star-cost** estimate. | [`radview/m4_forecast/`](radview/m4_forecast/) |
| **M5** Agent | **LangGraph** workflow. **Claude runs its own tool-calling loop** (`query_metrics`, `run_forecast`, `get_evidence_reviews`, `compare_peers`) and submits a structured plan; a **rule-based planner** is the offline fallback *and* the baseline. A mechanical **verifier** checks every claim on five axes (citation, quotation, relevance, statistic, substance); an **adversarial planner** proves it rejects fabrications. Priority = impact (SHAP + counterfactual) and urgency, divided by the effort of the fix. | [`radview/m5_agent/`](radview/m5_agent/) |
| **M6** Serving | FastAPI REST API + Streamlit dashboard (light/dark), one Docker image with a pytest gate, Kubernetes Job / CronJob / Deployments. | [`radview/api/`](radview/api/), [`dashboard/`](dashboard/), [`k8s/`](k8s/) |

Results from the latest run are in **[docs/RESULTS.md](docs/RESULTS.md)** (regenerate with `python scripts/make_results.py`);
**[docs/TEST_OUTPUTS.md](docs/TEST_OUTPUTS.md)** collects every test output in one file (all metrics, the pre-COVID
check, training curves, planner comparison, every pytest result, raw JSON); **[docs/FINDINGS.md](docs/FINDINGS.md)** interprets them — including what did *not* work on real data.

**Headline on real Yelp data (Philadelphia, 2016–2021):** fine-tuned DistilBERT is the best aspect model on human-labelled
SemEval-2014 (macro-F1 0.737, complaint-F1 0.567); the agent's 53 plans are 100% verified while an adversarial planner is
100% rejected; but a rating-trend logistic regression (AUROC 0.738) edges out the PyTorch GRU (0.727) at forecasting
declines, and complaint spikes do not anticipate declines at monthly granularity — see FINDINGS for why.

---

## Run it with the already-trained models (group members start here)

No retraining needed — about 10 minutes. You need this repo plus the private bundle
**`Radview-trained-models-yelp.zip`** (~320 MB, on the group drive; it contains Yelp review text, so keep it
within the group and never commit it).

**1. Get the code and install** (Python 3.10/3.11; on Windows tick *"Add python.exe to PATH"* when installing Python)

```powershell
git clone https://github.com/Chef2121/dsa2111_project.git
cd dsa2111_project
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
```

**2. Unzip the bundle into the `dsa2111_project` folder**, so that `data/` sits next to this README
(`data/work/yelp/models`, `data/work/yelp/outputs`, …). Check it worked — this should print `yelp 327`:

```powershell
python -c "from radview.config import SOURCE; from radview import service; print(SOURCE, len(service.list_restaurants()))"
```

**3. Run**

| What | Command | Open |
|---|---|---|
| Dashboard | `python -m streamlit run dashboard/app.py` | http://localhost:8501 |
| REST API + docs | `python -m uvicorn radview.api.main:app --port 8000` | http://localhost:8000/docs |
| Tests (23) | `python -m pytest -q` | — |
| An action plan in the terminal | `python -m radview.m5_agent.cli` | — |

**Test the models yourself:** dashboard → **Try a review** (paste any review, see the aspect labels and the
complaint sentence), **Portfolio** → click a restaurant (risk, drivers, verified action plan), **Model
performance** (all evaluation results). Keep the terminal open while using the dashboard — closing it gives
*"site cannot be reached"*.

The full guide — macOS/Linux notes, a 5-minute presentation demo script, re-running evaluations, Claude/Docker
options and troubleshooting — is **[docs/RUN_WITH_TRAINED_MODELS.md](docs/RUN_WITH_TRAINED_MODELS.md)**.

---

## Quick start: train everything from scratch

Requires Python 3.10+. Runs on CPU (default). `RADVIEW_DEVICE=cuda` opts in to a GPU.

```bash
pip install -r requirements-dev.txt

# 1. data: Yelp (~5.9 GB from Kaggle) + SemEval-2014 benchmark
python scripts/download_data.py
python -m radview.m1_ingest.inspect_data          # preflight: which city, how big

# 2. full pipeline: ingest → train/evaluate all models → score → action plans → planner comparison
python -m radview.pipeline            # ~2-3 h on CPU with DistilBERT; add --fast for TF-IDF only (~5 min)

# 3. dashboard (reads the outputs in-process) …
streamlit run dashboard/app.py        # http://localhost:8501
# … and/or the REST API
uvicorn radview.api.main:app --port 8000   # http://localhost:8000/docs
```

**No Yelp download?** `RADVIEW_SOURCE=synthetic python -m radview.pipeline --fast` generates a Yelp-format
synthetic dataset (220 restaurants with injected deterioration episodes and known ground truth) and runs the whole
system in a few minutes. Outputs for each source live in `data/work/<source>/`, so they never overwrite each other.

**Claude-drafted plans:** export `ANTHROPIC_API_KEY=…` (default model `claude-opus-5-5`, override with
`RADVIEW_LLM_MODEL`). Without a key the rule-based planner runs and every plan says which planner produced it.

```bash
python -m radview.m5_agent.cli --business-id <id>          # one plan
python -m radview.m5_agent.cli --compare 20                # rule-based vs Claude vs adversarial
```

### Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `RADVIEW_SOURCE` | `yelp` if `data/yelp/` has the files, else `synthetic` | which dataset to use |
| `RADVIEW_DATA_DIR` | `./data` | root for raw data, models and outputs (`/data` in containers) |
| `RADVIEW_CITY` / `RADVIEW_MAX_RESTAURANTS` | `Philadelphia` / `400` | Yelp slice: the N most-reviewed restaurants in a city |
| `RADVIEW_START_DATE` | `2016-01-01` | ignore older reviews |
| `RADVIEW_DEVICE` | `cpu` | `cuda` to train/score the PyTorch models on a GPU |
| `ANTHROPIC_API_KEY`, `RADVIEW_LLM_MODEL` | – / `claude-opus-5-5` | enable the Claude planner (and the zero-shot M2 baseline) |
| `RADVIEW_API_URL` | – | dashboard reads from this API instead of in-process |

Model and threshold settings (decline threshold, horizon, spike z, time split…) are documented in
[`radview/config.py`](radview/config.py).

---

## The dashboard

`streamlit run dashboard/app.py` — the restaurant page answers an owner's questions in the order they ask them:

| Question | What the page shows |
|---|---|
| Am I in trouble? | Risk badge (colour **and** label), a plain-English verdict sentence, tiles: chance the rating falls, 3-month rating vs 6-month, review volume, live alerts |
| Is my rating moving? | Monthly average rating, with gaps where nobody reviewed instead of interpolation |
| What are people complaining about? | Complaint rate per aspect with a tick for the **market median** |
| Is it getting worse? | Pick an issue → its monthly complaint rate vs the market, with M3 spike markers |
| What do I do first? | Ranked action cards: "Do this", estimated star cost, risk reduction if fixed, effort, verified review quotes, and a verifier log of rejected claims |
| Why? | SHAP and counterfactual explanations side by side; risk history; peers; review explorer |

Other pages: **Portfolio** (all restaurants ranked, for chains/analysts), **Alerts** (live spikes across the
market), **Try a review** (paste reviews → aspect sentiment with the complaint sentence highlighted), **Model
performance** (every evaluation table and curve), **About**. Every chart has a **Table** toggle; light and dark
palettes follow the viewer's theme; aspects keep one colour everywhere and status colours always carry a label.

---

## Docker & Kubernetes

```bash
docker build -t radview:latest .            # the build runs the pytest suite; a failing test fails the build

# docker compose: build data/models once, then serve
docker compose --profile jobs run --rm pipeline
docker compose up -d api dashboard          # API :8000 · dashboard :8501
```

**On the group's RONIN machine** (trained models, Docker Compose, SSH tunnel): follow
[docs/DEPLOY_RONIN.md](docs/DEPLOY_RONIN.md) — one command, `bash scripts/deploy_ronin.sh`.

Kubernetes (Docker Desktop, minikube, kind, EKS/GKE/AKS):

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
| `Deployment/radview-api` | FastAPI, readiness/liveness probes, reads the shared volume |
| `Deployment/radview-dashboard` | Streamlit, stateless (talks to the API), 2 replicas behind a LoadBalancer |
| `PersistentVolumeClaim/radview-data` | shared data/models/outputs volume |

---

## REST API

| Method & path | Returns |
|---|---|
| `GET /restaurants` | all monitored restaurants ranked by decline risk (filter `risk_level=high`) |
| `GET /restaurants/{id}` | metrics vs market, forecast with SHAP / counterfactual / star cost, peer comparison |
| `GET /restaurants/{id}/timeline` | monthly ratings, complaint rates vs market, spikes, risk |
| `GET /restaurants/{id}/reviews?aspect=service&sentiment=negative` | review explorer |
| `GET /restaurants/{id}/plan?refresh=true` | the agent's verified action plan (with tool calls and rejected claims) |
| `GET /alerts` | live complaint spikes across all restaurants, most severe first |
| `POST /analyse` | aspect sentiment + complaint sentences for raw review texts |
| `GET /evaluation`, `GET /evaluation/m4/curves` | M2–M5 evaluation results, PR/ROC curves |
| `POST /feedback` | owner usefulness rating (1–5) for a plan |

Interactive docs at `/docs`.

---

## Evaluation design

* **M2** — macro-F1 over {not mentioned, positive, negative} per aspect, plus complaint-F1, on SemEval-2014 Task 4
  restaurants (800 human-labelled sentences; food/service/price/ambience). Baselines: the lexicon rule, TF-IDF +
  logistic regression, zero-shot Claude (with a key). `data/labels/yelp/to_label.csv` is exported for an in-domain
  hand-labelled set that also covers cleanliness.
* **M3** — precision/lift of complaint spikes for a decline in the next 3 months, share of decline onsets preceded
  by a spike and the lead time, against an equivalent alarm on the star rating itself.
* **M4** — strict chronological split; features use past data only (unit-tested). AUROC, PR-AUC, **Brier score**
  (every model Platt-calibrated on validation), precision/recall in the top 10%, early warning at a fixed 15% alert
  budget. Baselines from trivial to strong: majority, current negativity (no model), rating-trend LogReg/LightGBM,
  +overall sentiment, all features with LightGBM. On synthetic data, **injected deterioration episodes** are scored for
  detection and whether the alert fired *before* the drop was visible in the rating. SHAP and counterfactual
  attributions are compared (top-aspect agreement, rank correlation).
* **M5** — **groundedness** (share of drafted claims that pass) with failures broken down by axis; the planner
  comparison runs rule-based, adversarial (must score 0) and Claude (with a key) on the riskiest restaurants and
  reports latency, tool calls and tokens. Owner usefulness ratings are collected from the dashboard.

## Compared with the ReviewRadar reference implementation

[ReviewRadar](https://github.com/zappyhan/Nice_Or_Not-) implements the same project idea on a simulator. Radview keeps
its strongest ideas (agentic tool calling, a mechanical verifier with an adversarial test, counterfactual attribution
and star cost, market-relative features, Brier/baselines/lead-time evaluation, owner-first dashboard) and goes further:

| | ReviewRadar | Radview |
|---|---|---|
| Evaluation data | simulator only | **real Yelp** (400 Philadelphia restaurants) + simulator for injected-episode checks |
| Aspect polarity | star rating (`stars ≤ 2` → complaint) | **from the text** — fine-tuned DistilBERT |
| Aspect model evaluation | weak labels | human-labelled **SemEval-2014** |
| Forecaster | gradient boosting | **GRU sequence model** (PyTorch) + 5 baselines, calibrated |
| Explanation | counterfactual | counterfactual **and SHAP**, with their agreement reported |
| Check-ins / tips | not used | used as features |
| Verifier | citation, quotation, statistic | + **relevance** (the quoted sentence must itself be a complaint about the aspect) + retrieval-scoped citations + star figures |
| Portfolio view, usefulness ratings, live review analysis | future work | built |

## Tests

```bash
pytest            # unit, agent and API tests (agent/API tests need a pipeline run)
```

## Repository layout

```
radview/
  config.py            paths, constants, thresholds
  lexicon.py           seed lexicons for weak supervision & evidence
  panel.py             monthly restaurant panel shared by M3/M4
  m1_ingest/           ingest.py (Yelp streaming ETL), synthetic.py (simulator), inspect_data.py (preflight)
  m2_aspects/          weak_labels.py, models.py (lexicon, TF-IDF, DistilBERT, zero-shot LLM), train.py
  m3_trends/           detector.py (EWMA + robust z, evaluation)
  m4_forecast/         features.py (leak-free + market features & label), model.py (PyTorch GRU, SHAP),
                       attribution.py (counterfactual, star cost), train.py (models, baselines, evaluation)
  m5_agent/            tools.py, agent.py (LangGraph + Claude/rule-based planners), verifier.py,
                       adversarial.py, playbook.py, cli.py (planner comparison)
  service.py           read-side service layer used by API and dashboard
  api/main.py          FastAPI app
  pipeline.py          end-to-end orchestration (full / rescore)
dashboard/             Streamlit app + API client
k8s/                   Kubernetes manifests
scripts/               data download, results report
tests/                 pytest suite
docs/RESULTS.md        latest evaluation tables
```

## Known limitations

1. **Weak labels are noisy.** M2 is trained on lexicon + star labels; SemEval measures it on human labels but has no
   cleanliness category — hand-label `to_label.csv` for a full in-domain figure.
2. **Attribution is associational.** The counterfactual intervenes on the model, not the restaurant; the star cost is
   a market-wide regression. Neither proves that fixing an aspect raises the rating.
3. **The verifier checks traceability, not truth.** A plan built on a wrong forecast still passes if it cites correctly.
4. **The Claude planner needs credentials** to be evaluated; without them the comparison reports rule-based and
   adversarial only, and says so.
5. **Yelp is self-selected and ends in Jan 2022** (the test period overlaps COVID-19), so absolute lead times will
   differ from a live review feed.
6. **CPU budget:** the Yelp slice (one city, top 400 restaurants, 2016+) keeps DistilBERT scoring to ~1 hour on CPU.

## Data & licences

The Yelp Open Dataset is used under Yelp's Dataset User Agreement (academic use) and is **not** included in this
repository. SemEval-2014 Task 4 data is used for evaluation only.
