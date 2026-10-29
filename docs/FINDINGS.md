# What the real-data results say (and don't)

Numbers are from [RESULTS.md](RESULTS.md): Yelp, the 400 most-reviewed Philadelphia restaurants
(397 kept after cleaning), reviews 2016-01 → 2021-12, test period 2020-04 → 2021-09.
This page is the interpretation for the report; RESULTS.md is regenerated automatically.
All numbers are from the run after the code-review fixes (stricter verifier, purged time split, chance
baselines, M2 chosen on a dev set); see [GPU_RERUN.md](GPU_RERUN.md) for what changed.

## What worked

1. **Text-based aspect sentiment beats the trained alternatives.** On the human-labelled SemEval-2014 test set,
   the fine-tuned DistilBERT reaches macro-F1 **0.750** vs 0.723 (TF-IDF) and 0.661 (the lexicon rule that
   produced its training labels) — it generalises beyond its own weak labels. The gain is largest exactly where
   Radview needs it: **complaint-F1 0.580 vs 0.494** (+17% relative). Per aspect it is strongest on service
   (0.86) and weakest on food and ambience (0.68). The model was chosen on a separate SemEval *dev* set
   (DistilBERT 0.711 vs TF-IDF 0.681 there), so the test score is not also the selection score.
   **But a zero-shot LLM beats it** (next section).
2. **The verifier catches fabrications — including realistic ones.** The adversarial planner's 246 claims were
   **all** rejected, every check axis firing; they now include five realistic mistakes (a real market figure stated
   as the restaurant's own, the decline risk stated as a complaint rate, the rating stated as a star cost, a real
   citation with no quote and invented content, and an uncheckable "tripled"). The rule-based planner's 62 plans
   pass **296/296** claims, but that is 100% *by construction* (it only writes the figures and quotes the verifier
   checks); the meaningful test is an LLM planner — see item 5 below.
3. **Complaint spikes mark lasting damage.** After a spike, the next 3 months are ≥0.3★ below the restaurant's
   rating from *before* the spike **33%** of the time, against 24% across all months: **lift 1.35**, where spikes moved to
   random months reach only 1.07. Spikes do not predict a *further* drop (below), but they do flag damage that
   stays.
4. **The conclusions do not depend on which restaurants are studied.** The main sample favours survivors (82%
   still open, against 60% of all Philadelphia restaurants). Re-running everything on the 400 restaurants with the
   most reviews in 2016 (77% open; 113 restaurants swapped) gives the same picture on every module — see
   [ROBUSTNESS_SELECTION.md](ROBUSTNESS_SELECTION.md): DistilBERT 0.743 / 0.550, spike lift 0.75 (chance 1.09),
   lasting-drop lift 1.34 (chance 1.05), rating-trend LogReg again the best forecaster (AUROC 0.758).
5. **Calibration holds up.** The GRU's mean predicted risk over the test period is 6.6% against an observed 7.05%
   decline rate, so its probabilities can be read as probabilities.

## What did not work — report these honestly

1. **A zero-shot LLM labels aspects better than our trained model.** The free Qwen 2.5 7B, prompted with no
   training on one T4 GPU, reaches macro-F1 **0.842** and complaint-F1 **0.748** on the same 800 SemEval test
   sentences (DistilBERT 0.750 / 0.580), ahead on every aspect, with all replies valid JSON. It takes ~886 ms per
   sentence against ~1 ms for DistilBERT on the same GPU (~840×), so DistilBERT stays the production scorer;
   SemEval is public, so the LLM may have seen it in pre-training. Next step: train DistilBERT on LLM labels
   instead of the lexicon rule.
2. **The simplest baseline wins M4.** A logistic regression on rating-trend features alone (AUROC **0.734**,
   PR-AUC **0.201**) beats the GRU (0.708 / 0.161) and every model that adds aspect features. The GRU has the
   best PR-AUC of the multi-signal models (LightGBM ≤ 0.152) but a lower AUROC than the rating-trend and
   overall-sentiment LightGBMs (0.720, 0.721). On this data, *what* people complain about does not improve the
   3-month decline forecast beyond the rating trend itself.
3. **Early warning is at chance for most models.** With a 15% alert budget, the GRU warns of 55.5% of decline
   onsets — the same as the same number of random alerts (54.4%). Only the rating-trend logistic regression
   clearly beats chance (65.8% vs 53.5%). Reported early-warning rates mean little without this baseline.
4. **Complaint spikes do not anticipate declines (M3 lift 0.73; random timing 1.13).** Months with a spike are
   followed by a decline *less* often than average — and less often than spikes at random times. The reason is
   regression to the mean: in a spike month the 3-month rating is already ~0.16★ below the 6-month rating, and
   the label compares the next 3 months with a window that contains the drop, so the following months tend to
   recover. The early-warning hypothesis is **not supported at monthly granularity** on Yelp; the rating alarm
   (now with a proper sampling-noise term) fires rarely and does worse still (lift 0.25), although it is an even
   stronger lasting-damage marker (2.13) because it only fires once damage is visible.
5. **A small open LLM is not reliable on its own.** Running the agent with a free open-weight model (Qwen 2.5 7B
   via Ollama on one T4 GPU, `RADVIEW_LLM_PROVIDER=open`) on the 20 riskiest restaurants, only **27/46 claims
   (59%)** passed the stricter verifier, against 100% for the rule-based planner. It cited only retrieved reviews
   and quoted them verbatim this time, but **11 claims attached a figure to the wrong quantity or stated an
   uncheckable one** and 8 metric claims stated no figure at all. The earlier run's 69% was measured with the
   looser verifier, which accepted any matching number. This is the strongest evidence that the verifier is
   needed — it removed every unsupported claim before the owner saw it.
   The cost shows on five well-reviewed restaurants we checked by hand: only 5 of Qwen's 11 claims passed, two
   plans were left with no evidence, and for one restaurant Qwen produced no usable plan (the agent fell back to the
   rule-based planner). So a usefulness rating study should show rule-based plans only; owners' ratings are left
   for future work.
6. **Risk is low in absolute terms.** Declines are rare (~6% of restaurant-months) and calibrated risks top out
   around 21–23%, which is why risk tiers are defined as *lift* over the typical restaurant — the mean predicted
   risk of the active restaurants in the latest month (5.4%) — with ≥1.5× medium and ≥2× high: 41 high and 21
   medium of 327 active restaurants.

## Robustness check: is it just COVID?

Re-running M4 with all labels after Dec 2019 hidden (`python -m radview.m4_forecast.train --test-end 2019-12`;
test window Jul 2018 – Dec 2019, training 2016 – Sep 2017) gives the **same ordering**: rating-trend LogReg AUROC
0.707 / PR-AUC 0.159, GRU 0.660 / 0.116, LightGBM variants 0.590–0.625. The simple model's lead is therefore not a
COVID artefact; with ~6% positives there are too few decline events for higher-capacity models to add much. In this
window every model warns of fewer onsets than random alerts with the same budget, so early warning is not shown
here either.

## Robustness check: restaurant selection

See item 4 above and [ROBUSTNESS_SELECTION.md](ROBUSTNESS_SELECTION.md). Selecting restaurants on their 2016 review
counts (no information from after the start of the study) changes 113 of the 400 restaurants and leaves every
conclusion unchanged.

## Caveats that affect every number

- **COVID-19:** the test window (Apr 2020 – Sep 2021) is a regime the model never saw in training
  (2016 – Jun 2019) — volumes halved and ratings behaved differently. The pre-COVID re-run above shows the
  model ranking holds, but absolute scores differ between the two windows.
- **Weak labels lean on stars:** about 4 in 10 M2 training labels take their polarity from the star rating (when
  a sentence names an aspect without a sentiment word), so M2 is *mostly*, not purely, text-based.
- **Low review volume:** many restaurants have only a handful of reviews per quarter by late 2021, so a
  single review can move a complaint rate by 30 points; plans state counts ("1 of 3 reviews") and add a
  small-sample caveat below 10 reviews.
- **One city, 400 restaurants**, chosen by review count — larger, busier, better-rated restaurants than average
  (mean 3.91★ against 3.56★ for all Philadelphia restaurants).
- **Star impact is constrained:** the star-cost regression only allows costs (coefficients ≤ 0); ambience comes out
  at 0 — unconstrained it was a positive correlation artefact.
- **The verifier checks traceability, not truth**, and its relevance check uses the same M2 model that chose the
  evidence.
- **M2 errors propagate:** e.g. "coleslaw was … foul smelling" is labelled a *cleanliness* complaint (smell
  words are cleanliness cues), so a food problem can surface as a cleanliness action.

## Suggested next experiments

- Hand-label the exported `data/labels/yelp/to_label.csv` (300 reviews) for an in-domain M2 score that
  includes cleanliness.
- Train DistilBERT on Qwen's zero-shot labels instead of the lexicon rule.
- Run `python -m radview.m5_agent.cli --compare 20` with an `ANTHROPIC_API_KEY` to measure a frontier LLM
  planner (Claude) against the open model (59%) and the rule-based baseline (100%) on groundedness, latency
  and cost.
