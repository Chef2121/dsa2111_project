# What the real-data results say (and don't)

Numbers are from [RESULTS.md](RESULTS.md): Yelp, the 400 most-reviewed Philadelphia restaurants
(397 kept after cleaning), reviews 2016-01 → 2021-12, test period 2020-04 → 2021-09.
This page is the interpretation for the report; RESULTS.md is regenerated automatically.

## What worked

1. **Text-based aspect sentiment beats the alternatives.** On human-labelled SemEval-2014, the fine-tuned
   DistilBERT reaches macro-F1 **0.737** vs 0.727 (TF-IDF) and 0.660 (the lexicon rule that produced its
   training labels) — it generalises beyond its own weak labels. The gain is largest exactly where Radview
   needs it: **complaint-F1 0.567 vs 0.512** (+11% relative). Per aspect it is strongest on service (0.86)
   and weakest on ambience (0.64).
2. **The agent's plans are fully traceable.** 53 plans, **265/265** claims verified (100% groundedness);
   the adversarial planner's 160 fabricated claims were **all** rejected, every check axis firing. Note the
   rule-based planner is 100% by construction (it only states what the tools return); the meaningful
   groundedness test is an LLM planner — see item 4 below.
3. **The GRU is the best *learned* multi-signal model**: AUROC 0.727 / PR-AUC 0.171, ahead of every
   LightGBM variant (≤ 0.717 / ≤ 0.152), and 3× the test base rate in precision@10% (0.207 vs 0.070).
4. **The two explanation methods broadly agree**: SHAP and the counterfactual pick the same top aspect for
   64% of restaurants (mean rank correlation 0.64), which gives some confidence the "fix this first"
   ranking is not an artefact of one method.

## What did not work — report these honestly

1. **The simplest baseline wins M4.** A logistic regression on rating-trend features alone (AUROC **0.738**,
   PR-AUC **0.205**) beats the GRU and every model that adds aspect features. On this data, *what* people
   complain about does not improve the 3-month decline forecast beyond the rating trend itself.
2. **Complaint spikes do not anticipate declines (M3 lift 0.63 < 1).** Months with a spike are followed by a
   decline *less* often than average. The most likely reason is regression to the mean: a spike usually
   coincides with a rating that has already dropped, and the label compares the next 3 months with the past
   6 — so the following months tend to recover. The early-warning hypothesis is **not supported at monthly
   granularity** on Yelp; the rating-based alarm does even worse (lift 0.32).
3. **Risk is low in absolute terms.** Declines are rare (~6% of restaurant-months) and calibrated risks top
   out around 13–16%, which is why risk tiers are defined as *lift* over the base rate (≥1.5× medium,
   ≥2× high) rather than fixed probabilities.
4. **A small open LLM is not reliable on its own.** Running the agent with a free open-weight model (Qwen 2.5 7B
   via Ollama on one T4 GPU, `RADVIEW_LLM_PROVIDER=open`) on the 20 riskiest restaurants, only **33/48 claims
   (69%)** passed the verifier, against 100% for the rule-based planner. The model did act as an agent (4.4 tool
   calls of its own per plan, mostly fetching more evidence), but it paraphrased "quotes", wrote vague claims with
   no figure, cited invented review ids such as `r123` and once stated a statistic the tools never returned.
   12/20 plans were fully grounded, 8 needed a redraft after verifier feedback and 4 ended with nothing verified;
   owners saw 11 quote-backed items against 24 from the rule-based planner, and its actions were mostly copied
   from the playbook. This is the strongest evidence that the verifier is needed — it removed every unsupported
   claim. Details: [TEST_OUTPUTS.md §6.3](TEST_OUTPUTS.md).

## Robustness check: is it just COVID?

Re-running M4 with all labels after Dec 2019 hidden (`python -m radview.m4_forecast.train --test-end 2019-12`;
test window Jul 2018 – Dec 2019, training 2016–2017) gives the **same ordering**: rating-trend LogReg AUROC 0.711 /
PR-AUC 0.163, GRU 0.686 / 0.147, LightGBM variants 0.608–0.632. The simple model's lead is therefore not a COVID
artefact; with ~6% positives there are too few decline events for higher-capacity models to add much.

## Caveats that affect every number

- **COVID-19:** the test window (Apr 2020 – Sep 2021) is a regime the model never saw in training
  (2016 – Sep 2019) — volumes halved and ratings behaved differently. The pre-COVID re-run above shows the
  model ranking holds, but absolute scores differ between the two windows.
- **Low review volume:** many restaurants have only a handful of reviews per quarter by late 2021, so a
  single review can move a complaint rate by 30 points; plans now state counts ("1 of 3 reviews") and add a
  small-sample caveat below 10 reviews.
- **One city, 400 restaurants**, chosen by review count — larger, busier restaurants than average.
- **Star-impact signs:** the ridge model gives ambience a small *positive* coefficient (+0.37 stars per unit
  rate) — a correlation artefact (ambience complaints co-occur with otherwise well-rated venues); star cost is
  clipped at zero, so it never reports a negative cost.
- **M2 errors propagate:** e.g. "coleslaw was … foul smelling" is labelled a *cleanliness* complaint (smell
  words are cleanliness cues), so a food problem can surface as a cleanliness action.

## Suggested next experiments

- Redefine the M3 evaluation target as "rating 3 months later is below the *pre-spike* baseline" to test
  whether spikes mark lasting damage rather than predicting a further drop.
- Hand-label the exported `data/labels/yelp/to_label.csv` (300 reviews) for an in-domain M2 score that
  includes cleanliness.
- Run `python -m radview.m5_agent.cli --compare 20` with an `ANTHROPIC_API_KEY` to measure a frontier LLM
  planner (Claude) against the open model (69%) and the rule-based baseline (100%) on groundedness, latency
  and cost.
