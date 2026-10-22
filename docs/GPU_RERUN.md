# Re-running everything on a GPU (after the review fixes)

The fixes on branch `fix/review-findings` change how labels, splits, baselines and checks are computed, so every
reported number has to be regenerated once. This guide runs the whole pipeline on a GPU machine (e.g. the
group's RONIN **g4dn.xlarge**, one NVIDIA T4) from the trained-models bundle; the raw Yelp files are **not** needed.

**Time:** about 2–2.5 h of compute (main run + selection robustness check) plus ~30 min setup. **Cost:** about
$0.55–0.75 per hour on a g4dn.xlarge, so roughly $2–5 in total. **Before you start, set an auto-stop / shutdown schedule in RONIN** (e.g. stop after 1 h
idle) so a forgotten machine cannot drain the budget.

## 0. On your laptop: put the branch where the GPU machine can get it

```bash
git push -u origin fix/review-findings      # or copy the repo folder to the machine with scp
```

## 1. Set up the machine

```bash
git clone https://github.com/Chef2121/dsa2111_project.git && cd dsa2111_project
git switch fix/review-findings
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126   # CUDA build of the pinned torch
pip install -r requirements-dev.txt
python -c "import torch; print('GPU:', torch.cuda.is_available())"           # must print True
```

## 2. Data

Copy `Radview-trained-models-yelp.zip` (main analysis, ~320 MB) and `Radview-select2016-processed.zip`
(robustness sample, ~50 MB) to the machine (e.g. `scp`) and unzip both in the repo folder, so `data/` and
`data_select2016/` sit next to `README.md`. Keep a copy of the old results, and fetch the SemEval files (the
training file is new: M2 now picks its model on a dev set taken from it, not on the test set):

```bash
unzip Radview-trained-models-yelp.zip && unzip Radview-select2016-processed.zip
cp -r data/work/yelp data/work/yelp_before_fixes
python scripts/download_data.py --skip-yelp
```

Both zips contain Yelp review text: keep them within the group and delete them from the machine when done.

## 3. Full run (≈ 45–60 min)

```bash
export RADVIEW_DEVICE=cuda RADVIEW_SOURCE=yelp
python -m radview.pipeline --from-processed            # M1 skipped (cuisine recomputed); M2 retrained at 256 tokens
python -m radview.m4_forecast.train --test-end 2019-12   # pre-COVID robustness check (report Table VII)
```

If the machine *does* have the raw Yelp files in `data/yelp/`, run `python -m radview.pipeline` instead.

## 3b. Robustness check: a different restaurant selection (≈ 45–60 min)

The main analysis keeps the 400 restaurants with the most *lifetime* reviews, which favours survivors (82% still
open, against 60% of all Philadelphia restaurants). `data_select2016/` holds the 400 with the most reviews **in
2016** instead (77% still open; 287 restaurants are in both). The same pipeline on it shows whether the
conclusions depend on the selection:

```bash
RADVIEW_DATA_DIR=data_select2016 python -m radview.pipeline --from-processed
python scripts/compare_selection.py --main data --robust data_select2016 --out docs/ROBUSTNESS_SELECTION.md
```

(`data_select2016/` was built on the laptop with `RADVIEW_SELECT_BY=early python -m radview.m1_ingest.ingest`
from the raw Yelp files.)

## 4. Free open-model baselines (optional, ≈ 20 min, uses the GPU through Ollama)

See [DEPLOY_RONIN.md](DEPLOY_RONIN.md) for installing Ollama. Then:

```bash
ollama pull qwen2.5:7b
export RADVIEW_LLM_PROVIDER=open
python -m radview.m2_aspects.train --zero-shot-only --zero-shot-n 800   # M2 zero-shot baseline (Table IV)
python -m radview.m5_agent.cli --compare 20                             # rule-based vs Qwen vs adversarial (Table VIII)
unset RADVIEW_LLM_PROVIDER
```

## 5. Synthetic episodes and tests (≈ 10 min)

```bash
RADVIEW_SOURCE=synthetic python -m radview.pipeline     # injected-episode results (report Sec. IV-D)
python -m pytest -q                                     # 43 tests; all should pass
```

## 6. Bring the results back and stop the machine

```bash
python docs/report/snapshot_inputs.py                   # copies the new metrics into docs/report/inputs/
zip -r results-after-fixes.zip data/work/yelp/outputs data/work/yelp/models/m2_selected.json \
    data/work/synthetic/outputs data_select2016/work/yelp/outputs docs/report/inputs docs/ROBUSTNESS_SELECTION.md
```

Download `results-after-fixes.zip` (and, if you want the retrained models, `data/work/yelp/models/`), then
**stop the instance**. Back on the laptop, regenerate the documents from the new numbers:

```bash
python scripts/make_results.py
python scripts/export_test_outputs.py
python docs/report/make_figures.py && python docs/report/build_report.py
```

and update the numbers quoted in README.md and docs/FINDINGS.md (or ask Claude to do it from the new files).

## What changed, so you know what to expect in the new numbers

| Module | Change | Effect on results |
|---|---|---|
| M1 | Cuisine = most common food category (venue/drink tags skipped) | Better peer groups in M5; no model change |
| M2 | Idioms ("can't wait") and "nothing but" fixed; 3★ reviews no longer count as complaints; 256 tokens; model chosen on a SemEval **dev** set | New M2 F1 scores; possibly a different selected model |
| M3 | True median/MAD; rating alarm uses the s.e. of a mean rating; random-timing baseline; "lasting drop" metric | Slightly more spikes (~+20%); far fewer rating alarms; new chance and lasting-drop columns |
| M4 | 3-month purge gap at split boundaries; early warning in calendar months with a random-alert baseline; tiers relative to the mean predicted risk this month; star cost can only be ≤ 0 | AUROC a little lower (≈ −0.005 to −0.02 on the baselines); fewer "high" restaurants; ambience star cost 0 |
| M5 | Verifier: quotes required, each figure checked against the metric its sentence names, multipliers rejected, LLM summaries/headlines checked; adversarial planner has 5 subtle attacks; impact share counts non-aspect drivers | Rule-based still 100%; adversarial still 0% (now with 13 claims per plan); LLM groundedness likely lower than before |
| M6 | API reloads data after each pipeline run; plan refreshes are rule-based unless `RADVIEW_ALLOW_LLM_REFRESH=1`; clearer spike labels; escaped HTML | No metric change |
