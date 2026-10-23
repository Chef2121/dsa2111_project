# Robustness check: restaurant selection

Main analysis = the 400 Philadelphia restaurants with the most lifetime Yelp reviews; robustness = the 400
with the most reviews in 2016 (no information from after the start of the study period).

| Result | Main (lifetime top 400) | Robustness (2016 top 400) |
|---|---|---|
| Restaurants (still open) | 397 (83%) | 400 (77%) |
| Reviews | 152,649 | 151,866 |
| M2 selected model (on semeval2014_dev) | distilbert_finetuned | distilbert_finetuned |
| M2 macro-F1 / complaint-F1 (SemEval test) | 0.7496 / 0.5796 | 0.7431 / 0.5497 |
| M3 decline base rate | 5.83% | 6.15% |
| M3 spike lift (random timing) | 0.73 (1.13) | 0.75 (1.09) |
| M3 onsets detected % (random timing) | 22.3 (34.9) | 23.0 (33.9) |
| M3 lasting-drop lift (random timing) | 1.35 (1.07) | 1.34 (1.05) |
| M4 majority: AUROC / PR-AUC | 0.5 / 0.0705 | 0.5 / 0.0694 |
| M4 current_negativity: AUROC / PR-AUC | 0.5972 / 0.0921 | 0.6142 / 0.0953 |
| M4 rating_logreg: AUROC / PR-AUC | 0.7336 / 0.2006 | 0.7578 / 0.2044 |
| M4 rating_lgbm: AUROC / PR-AUC | 0.7204 / 0.1476 | 0.6971 / 0.1441 |
| M4 sentiment_lgbm: AUROC / PR-AUC | 0.7213 / 0.1516 | 0.6831 / 0.1378 |
| M4 radview_lgbm: AUROC / PR-AUC | 0.6802 / 0.1384 | 0.6811 / 0.1467 |
| M4 radview_gru: AUROC / PR-AUC | 0.7083 / 0.1607 | 0.7131 / 0.149 |
| M4 best model by PR-AUC | rating_logreg | rating_logreg |
| M4 GRU onsets warned % (random alerts) | 55.5 (54.4) | 51.4 (55.8) |
| M5 groundedness: rule_based | 1.0 (70/70) | 1.0 (81/81) |
| M5 groundedness: adversarial | 0.0 (0/246) | 0.0 (0/251) |
