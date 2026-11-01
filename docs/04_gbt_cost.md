# Step 3: Gradient-Boosted Trees and a Cost-Based Threshold

**Goal:** train LightGBM, check honestly whether it beats the logistic regression baseline, and choose the
decision threshold using money instead of F1.
**Run:** [notebooks/04_gbt_cost.ipynb](../notebooks/04_gbt_cost.ipynb) (or `python scripts/run_gbt.py`).
Results go to `results/gbt.md` and `figures/gbt_*.png`.

## 1. Why gradient-boosted trees

Logistic regression draws one straight boundary through the features. Gradient-boosted trees build many small decision
trees, each one correcting the mistakes of the previous ones. They capture non-linear patterns and feature interactions
(for example "unusual V14 *and* unusual V17"), which is why they are the usual choice for tabular fraud data.
They also need no feature scaling.

**Honesty check:** with about 50 test frauds, a better model may not show up as a *statistically clear* gain.
The notebook reports a paired bootstrap interval for the PR-AUC difference. If it includes 0, say "not proven better".

## 2. How we train without peeking

| What we decide | Data used |
|---|---|
| Fit the trees | Train |
| Number of trees (early stopping), hyperparameters | Validation |
| Calibration (Platt scaling) | Validation |
| Decision threshold | Validation |
| Final numbers | **Test, used once** |

- **Early stopping:** add trees until validation PR-AUC stops improving for 50 rounds. This prevents overfitting.
- **Random search:** 12 random hyperparameter combinations (tree size, minimum leaf size, row/column sampling, regularization).
  The best trial by validation PR-AUC wins.
- **`scale_pos_weight`:** LightGBM's version of class weights. Trials rotate through no weighting, a square-root weight and the full
  weight, so the notebook can show whether weighting helps trees as well (step 2's question, repeated).
- **One pipeline artifact:** preprocessing plus model are saved together; verify the deployed bundle's provenance before claiming it is the evaluated model.

## 3. Metrics

| Metric | What it tells you |
|---|---|
| **PR-AUC** (with bootstrap interval) | Ranking quality for the rare class |
| **Recall @ 90% precision** | Fraud caught if alerts must be 90% right |
| **Recall @ top-k alerts** | Fraud caught if analysts can review only the top 0.2% of transactions (a real *alert budget*) |
| **Total cost in dollars** | The business answer, using the cost assumptions below |

## 4. Calibration (making scores mean probabilities)

Weighting and boosting can distort scores, so "0.3" may not mean a 30% chance of fraud. **Platt scaling** fits a tiny logistic
regression from the model's score to the true outcome on validation. It changes the *scale* but not the *ranking*, so PR-AUC is unchanged.
It matters because the amount-aware rule in section 6 needs real probabilities.

## 5. The cost model (assumptions, stated on purpose)

| Event | Cost |
|---|---|
| Missed fraud (FN) | The transaction **Amount** |
| False alarm (FP) | A flat **$5** (analyst time plus customer friction). Editable |
| Caught fraud, correct pass | $0 |

**Total cost = (sum of missed fraud amounts) + (number of false alarms x $5).**
The baseline for comparison is "flag nothing": it costs the sum of *all* fraud amounts.
**Savings** = baseline minus a rule's cost, shown as a percentage.

Simplifications to keep in mind: real fraud losses include chargeback fees, investigation cost and recovery; a false alarm's
true cost varies by customer; the dataset contains only two days of data.

## 6. The decision rules compared

| Rule | Idea |
|---|---|
| Flag nothing / everything | Reference points. "Everything" shows how expensive false alarms are |
| LightGBM at 0.5 | The naive default |
| **F1-tuned threshold** | Best F1 on validation. Treats a $5 fraud and a $5,000 fraud the same |
| **Cost-tuned threshold** | One threshold chosen to minimize validation cost with real amounts |
| **Amount-aware rule** | A different bar per transaction (below) |

**Where the amount-aware rule comes from.** For a transaction with fraud probability `p` and amount `A`:
- Ignoring it costs, on average, `p x A` (the expected loss).
- Flagging it costs, on average, `(1 - p) x C_FP` (a false alarm if it was legit).

Flag when ignoring is the more expensive choice: `p x A >= (1 - p) x C_FP`, which rearranges to
**`p >= C_FP / (C_FP + A)`**. A $5 fraud needs `p >= 0.5`; a $500 one needs only `p >= 0.01`.
This only works with calibrated probabilities, so if it underperforms the single cost-tuned threshold, suspect calibration.

## 7. How to read the results

1. **Ranking table:** does the PR-AUC difference vs logistic regression exclude 0? Compare recall at 90% precision and at the alert budget.
2. **Cost table:** which rule is cheapest on test, and how much does it save vs flagging nothing? Check TP/FP/FN counts: with few frauds, one large transaction can swing the cost.
3. **Cost curve:** is the minimum wide and stable, and do validation and test agree?
4. **Sensitivity table:** does the model keep saving money if false alarms cost 1, 5, 20 or 50? If savings go negative at a plausible cost, the model is not worth deploying under that assumption.

## 8. Dataset-specific cautions

- **Many frauds in this dataset have small amounts** (check the EDA amount table). When most missed frauds cost only a few dollars,
  a $5 review cost is relatively expensive, so the best strategy may flag few transactions and the savings are driven by a handful of larger frauds.
- **Test has few frauds,** so conclusions about cost are fragile. Report counts and intervals.
- **V1 to V28 are anonymized,** which limits explanations (step 7).
- **One short test window.** Drift over time is not measured here.

## Recorded findings

LightGBM's test PR-AUC was 0.751 versus 0.644 for unweighted logistic regression; the paired bootstrap difference interval was +0.052 to +0.167. At 90% precision, recall was 0.712 versus 0.423; at the top 85 alerts, recall was 0.750 versus 0.692. The interval is conditional on this one 52-fraud test window and does not establish future performance.

Under the illustrative assumptions above, the validation cost-tuned LightGBM threshold of 0.0971 produced 39 true positives, 30 false positives and 13 misses, costing $2,522.40 versus $6,168.88 for flagging nothing (59.1% savings). The lowest observed test cost among the listed rules was $2,487.40 at threshold 0.5; choosing it because of this test result would require a fresh holdout. Across the listed $1–$50 false-alarm costs, savings were 56.1%–60.6% with thresholds retuned on validation. Calibration improved the recorded test Brier score from 0.00073 to 0.00050, but that alone does not establish reliable probabilities. See [results](results/gbt.md).

Hyperparameter search, early stopping, calibration and threshold selection reuse the same validation period, increasing overfitting risk. The two-day dataset, anonymized features, and assumed costs prevent using this as production validation.
