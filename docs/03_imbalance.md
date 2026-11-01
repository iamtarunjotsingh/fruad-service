# Step 2: Handling Class Imbalance

**Goal:** find out whether class weights, SMOTE, undersampling or threshold tuning improve fraud detection, and by how much.
**Run:** [notebooks/03_imbalance.ipynb](../notebooks/03_imbalance.ipynb) (or `python scripts/run_imbalance.py`). Results go to `results/imbalance.md` and `figures/imbalance_*.png`.

## The idea in plain language

With about 1 fraud per 600 legit transactions, a model trained naively sees fraud so rarely that it
learns to give almost everything a low score. There are two different ways to respond:

1. **Change the training** so the model pays more attention to fraud (class weights, SMOTE, undersampling).
2. **Change the decision** by moving the threshold so low-but-elevated scores still get flagged.

This step tests both, with everything else held constant.

## The strategies

| Strategy | How it works | Upside | Downside |
|---|---|---|---|
| **Unweighted** | Train as-is | Simple baseline | Calibration and threshold still need validation |
| **Class weights** | Training errors on fraud are multiplied (`class_weight="balanced"`, roughly 600x here) | Simple, no new data created | Scores become inflated |
| **SMOTE (10%)** | Creates synthetic frauds by interpolating between real neighboring frauds until fraud is 10% of legit | Gives the model more fraud examples to learn from | Synthetic points can be unrealistic; scores shift |
| **Undersampling (10%)** | Randomly drops legit rows until fraud is 10% of legit | Fast training | Throws away information |
| **Threshold tuning** | Keep the model; pick a better cut-off on validation | No retraining, keeps the model honest | Needs enough validation frauds |

"10%" is a choice (fraud ends up at 10% of the legit count), not a requirement. Fully balancing to 50/50 is the
common textbook setting and tends to distort probabilities more.

## Fair-comparison rules

- **Same model:** logistic regression for every strategy, so differences come from the strategy alone.
- **Same split and protocol:** time-based 70/15/15; thresholds tuned on validation; test used once.
- **Resampling only on train.** SMOTE and undersampling live inside the training pipeline and run only
  during `fit`. Resampling validation or test would make precision meaningless, because it changes the fraud rate.
- **Uncertainty is reported.** Test has few frauds, so PR-AUC comes with a bootstrap 95% interval, and each
  strategy is compared to the baseline with a *paired* bootstrap difference.

## How to read the results

**1. Ranking quality table (PR-AUC, recall at 90% precision).** These are threshold-free.
- If a strategy's "diff vs unweighted" range includes 0, it is not distinguishable from the baseline.
- Weighting and resampling mostly change the *scores*, not the *order* of transactions, so PR-AUC is often
  similar across strategies. A small drop is also common.

**2. Operating-point table (threshold 0.5 vs validation-tuned).**
- At 0.5, weighted or resampled models flag far more transactions: high recall, many false alarms.
- After tuning, the strategies often reach similar precision and recall. When that happens, the threshold
  mattered more than the training trick.

**3. Cost preview.** Total cost = misses x cost of a miss + false alarms x cost of a false alarm. The costs in the
notebook are **illustrative assumptions** (a miss costs 100, a false alarm costs 5). Change them to see how the
best strategy depends on your assumptions. Step 3 formalizes this with transaction amounts.

**4. Calibration check.** Compare the average predicted score with the true fraud rate. A ratio far above 1
means inflated scores. This matters because the cost rule `flag if p >= C_FP / (C_FP + C_FN)` needs calibrated
probabilities. If you keep class weights or SMOTE, calibrate on validation (isotonic or Platt) or choose the
threshold by sweeping actual costs.

## What to expect (hypotheses to check, not results)

| Hypothesis | Where to check |
|---|---|
| PR-AUC is similar across strategies | Ranking table: does the paired difference interval include zero? |
| Tuning the threshold helps the unweighted model | Operating-point table: val-tuned vs 0.5 rows |
| Class weights and SMOTE inflate probabilities | Calibration ratio |
| Strategies reach similar cost once thresholds are tuned | Cost preview |

If your numbers contradict a hypothesis, that is a finding worth writing up.

## Decision to carry into step 3

Pick the simplest strategy that holds up. A reasonable default is unweighted or class weights plus a tuned
threshold, and SMOTE only if it shows a clear, interval-backed gain. Gradient-boosted trees have their own
imbalance setting (`scale_pos_weight`), so step 3 repeats a short version of this comparison.

## Recorded findings

Undersampling had the highest observed test PR-AUC (0.703; 95% bootstrap interval 0.568–0.818), compared with 0.644 unweighted. Its paired difference interval was +0.022 to +0.102; SMOTE's was +0.008 to +0.093, while the class-weight interval included zero (−0.034 to +0.140). These intervals describe uncertainty on this test sample, not performance on future populations.

At the validation-tuned F1 threshold, class weights produced 35 true positives, one false positive and 17 misses, versus 33, 21 and 19 for unweighted logistic regression. Under the separate illustrative flat $100/miss and $5/false-alarm assumptions, those counts imply costs of $1,705 and $2,005 respectively. Mean weighted score was 0.1083 against a test fraud rate of approximately 0.0012, so the scores should not be interpreted as calibrated probabilities. The test contains only 52 frauds; selecting a strategy after looking at these results would require another independent evaluation. See [results](results/imbalance.md).
