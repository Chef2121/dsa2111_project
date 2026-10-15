# Running Radview with the already-trained models

This guide is for group members who want to **use** the trained system — dashboard, API, tests, demo —
without spending ~2 hours retraining. You need two things:

1. **The code** — this GitHub repo (ask Daniel for collaborator access).
2. **The trained-models bundle** — `Radview-trained-models-yelp.zip` (~320 MB, shared privately on the
   group drive). It holds the fine-tuned DistilBERT, the GRU forecaster, all scores, plans and evaluation
   results for 400 Philadelphia restaurants (Yelp, 2016–2021).

> The bundle contains Yelp review text. Yelp's licence allows academic use within the group —
> **do not post it publicly** or commit it to git (`*.zip` is already git-ignored).

---

## 1. One-time setup (≈10 minutes)

**Requirements:** Python 3.10 or 3.11 ([python.org](https://www.python.org/downloads/) — on Windows tick
*"Add python.exe to PATH"*), Git, ~3 GB free disk space. No GPU needed.

### Windows (PowerShell)

```powershell
git clone https://github.com/Chef2121/dsa2111_project.git
cd dsa2111_project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

If activation is blocked ("running scripts is disabled"), run this once in that window and retry:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

### macOS / Linux

```bash
git clone https://github.com/Chef2121/dsa2111_project.git
cd dsa2111_project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

On **Linux**, install the small CPU build of PyTorch first to avoid a ~2 GB CUDA download:
`pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu`, then the line above.

### Add the trained models

Unzip `Radview-trained-models-yelp.zip` **into the `dsa2111_project` folder** (the one containing
`README.md`). You should end up with:

```
dsa2111_project/
  data/
    work/yelp/models/       ← DistilBERT, GRU, LightGBM, TF-IDF
    work/yelp/processed/    ← cleaned reviews, aspect labels, monthly features
    work/yelp/outputs/      ← risk scores, action plans, evaluation results
    semeval/                ← SemEval-2014 test set
    labels/yelp/            ← review sheet for hand-labelling
  radview/
  dashboard/
  README.md
```

Windows: right-click the zip → *Extract All…* → choose the `dsa2111_project` folder.
Mac/Linux: `unzip Radview-trained-models-yelp.zip -d dsa2111_project`

**Check it worked** (should print `yelp` and `327`):

```bash
python -c "from radview.config import SOURCE; from radview import service; print(SOURCE, len(service.list_restaurants()))"
```

If it prints `synthetic`, the zip was extracted one folder too deep or too shallow — `data/` must sit
next to `README.md`.

---

## 2. Run it

Activate the environment first in every new terminal (`.\.venv\Scripts\Activate.ps1` or
`source .venv/bin/activate`), from inside `dsa2111_project`.

| What | Command | Then open |
|---|---|---|
| **Dashboard** | `python -m streamlit run dashboard/app.py` | http://localhost:8501 |
| **REST API** (+ interactive docs) | `python -m uvicorn radview.api.main:app --port 8000` | http://localhost:8000/docs |
| **Tests** (23, ~30 s) | `python -m pytest -q` | — |
| **One action plan in the terminal** | `python -m radview.m5_agent.cli` | — |
| **Planner comparison** (rule-based vs adversarial) | `python -m radview.m5_agent.cli --compare 20` | — |

The first dashboard load takes 10–20 s while it reads the data and loads DistilBERT; after that it is fast.
Stop any of them with **Ctrl+C**.

---

## 3. Testing the trained models

### a) Try your own reviews (the quickest model test)

Dashboard → **Try a review**. Paste reviews, one per line, and click **Analyse**. The fine-tuned
DistilBERT labels each aspect (👎 negative / 👍 positive, with confidence) and highlights the sentence that
carries each complaint. Good ones to try:

```
The steak was perfect but we waited 50 minutes for a table.
Cheap and filling, though the bathroom was disgusting.
Lovely candle-lit room, music a bit too loud, overpriced cocktails.
Staff were friendly and the pasta was fresh.
```

The same model is available at `POST /analyse` in the API docs page (*Try it out* → send
`{"texts": ["your review here"]}`).

### b) Inspect real predictions

Dashboard → **Portfolio** → click a restaurant. Check that the risk score, the complaint chart and the
action plan make sense against the actual reviews in the **💬 Reviews** tab. The **Alerts** page lists
restaurants whose complaints are spiking this month.

### c) See how good the models are

Dashboard → **Model performance**, or read [`docs/RESULTS.md`](RESULTS.md) (tables) and
[`docs/FINDINGS.md`](FINDINGS.md) (what they mean — including what did *not* work).

| Model | Test | Result |
|---|---|---|
| Aspect classifier (DistilBERT) | SemEval-2014, 800 human-labelled sentences | macro-F1 **0.737**, complaint-F1 **0.567** |
| Decline forecaster (PyTorch GRU) | Out-of-time test, Apr 2020 – Sep 2021 | AUROC **0.727**, PR-AUC **0.171** (best: rating-trend LogReg 0.738 / 0.205) |
| Agent verifier | 20 riskiest restaurants | rule-based plans **100%** verified; fabricated plans **0%** |

### d) Re-run the evaluations yourself (optional)

These reuse the trained models and overwrite the files in `data/work/yelp/outputs/`:

```bash
python -m radview.m4_forecast.train
```
Retrains and compares all 7 forecasting models (~2 min). The models train deterministically with fixed seeds, so the numbers should match the table.

```bash
python -m radview.m2_aspects.train --reuse-transformer
```
Re-evaluates the aspect models on SemEval **and re-scores all 152k reviews with DistilBERT (~50 min on CPU)** — only run it if you need to.

```bash
python scripts/make_results.py
```
Rebuilds `docs/RESULTS.md` from the outputs.

### e) Reproduce the report's tables

Each table in the report comes from one command (fixed seeds, so the numbers match exactly):

| Report item | Command |
|---|---|
| Table VI + Fig. 5 (forecasting, COVID-era test) | `python -m radview.m4_forecast.train` |
| Table VII (pre-COVID robustness check, ~2 min) | `python -m radview.m4_forecast.train --test-end 2019-12` |
| Table VIII (planner comparison) | `python -m radview.m5_agent.cli --compare 20` |
| Tables IV–V | already in the bundle: `data/work/yelp/outputs/m2_evaluation.json`, `m3_evaluation.json` |

The report's figures and tables rebuild from the repo alone — see [`docs/report/README.md`](report/README.md).

---

## 4. Demo script for the presentation (~5 minutes)

1. **Portfolio** — "Radview monitors 327 active Philadelphia restaurants; 15 are flagged high risk —
   at least twice the typical restaurant's chance of a 0.3-star drop in the next 3 months."
2. Click the top restaurant → **Restaurant** page: read the verdict line, point at the complaint bars vs the
   market-median ticks, pick an issue to show its trend and spike markers.
3. Scroll to **What to do first**: the ranked actions, the star-cost / risk badges, the verified review
   quotes, and the **Verifier log** ("nothing reaches the owner unless it is backed by a review").
4. **🔍 Why at risk** tab: SHAP and the counterfactual side by side.
5. **Try a review**: type a fresh review live and show the model picking out the complaint.
6. **Model performance**: SemEval scores, the forecasting comparison, and the planner comparison where the
   hallucinating planner scores 0%.
7. Optional: show `http://localhost:8000/docs` to demonstrate the REST API, and the Kubernetes manifests in
   `k8s/`.

Tip: start the dashboard 2 minutes before you present and click through each page once so everything is cached.

---

## 5. Optional extras

**Claude-written plans.** With an Anthropic API key the agent uses Claude (it calls the analytics tools itself)
instead of the rule-based planner:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."      # macOS/Linux: export ANTHROPIC_API_KEY=sk-ant-...
python -m radview.m5_agent.cli --compare 20
```

In the dashboard, click **↻ Re-run** on a plan to regenerate it with Claude. Never commit your key.

**Docker / Kubernetes.** Container setup is described in the main [README](../README.md#docker--kubernetes).
To serve the trained models from Docker, mount your `data/` folder:
`docker compose up -d api dashboard` (it mounts `./data` automatically).

---

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| Dashboard says *data: synthetic* or shows ~220 fictional restaurants | `data/` is in the wrong place — it must be next to `README.md`. Or force it: `$env:RADVIEW_SOURCE = "yelp"` (PowerShell) / `export RADVIEW_SOURCE=yelp` |
| `FileNotFoundError ... risk_latest.parquet` | The zip wasn't extracted (or only partly). Re-extract into the project folder. |
| `ModuleNotFoundError: radview` | Run commands from inside `dsa2111_project`, with the virtual environment activated. |
| `Port 8501 is already in use` | Another dashboard is running — close it, or add `--server.port 8502`. |
| `site cannot be reached` | The terminal running the dashboard was closed. Start it again; keep that window open. |
| First page load is slow | Normal (loads data + DistilBERT); subsequent clicks are fast. |
| `pip install` is very slow or huge on Linux | Install the CPU PyTorch build first (see setup). |
| Tests: "10 skipped" | The agent/API tests skip when no trained outputs are found — check the `data/` location. |

Questions about the code: each module's docstring explains what it does and why; start with
[`radview/pipeline.py`](../radview/pipeline.py), which runs M1 → M5 in order.
