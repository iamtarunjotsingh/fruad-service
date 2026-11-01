# Step 1b: Baselines

**Goal:** set the floor that every later model must beat, using the final evaluation protocol.
**Run:** [notebooks/02_baselines.ipynb](../notebooks/02_baselines.ipynb) (or `python scripts/run_baselines.py`). Results are saved to `results/baselines.md`; plots to `figures/`.

## Protocol (used for every model in this project)

1. Fit the model on **train** (oldest 70%).
2. Choose the decision threshold on **validation** (next 15%) by maximizing F1.
   Later steps replace this with a cost-based threshold.
3. Report on **test** (newest 15%) once, at both the default 0.5 threshold and the tuned one.

## Metrics, in plain language

| Metric | Meaning | Why we use it |
|---|---|---|
| **PR-AUC** (average precision) | How well the model ranks fraud above legit across all thresholds, focused on the rare class | Main metric. A random model scores about the fraud rate (about 0.002), so anything above that is skill |
| **Recall @ 90% precision** | Share of frauds caught if we only accept alerts that are right 9 times out of 10 | Maps to a business constraint: analysts can't review endless false alarms |
| **Precision / Recall / F1** | At one chosen threshold: how many alerts are real / how many frauds we catch | Needed for the confusion matrix |
| **TP / FP / FN** | Raw counts | The test set has few frauds, so counts keep us honest |
| ROC-AUC | Ranking quality over both classes | Reported for reference only. It looks great on imbalanced data even for mediocre models |

## Baseline 1: Majority class

**What it does:** always predicts "legit" with the same score for every row.
**What you should see:**
- PR-AUC equal to the fraud rate, recall 0, precision 0, FN equal to the total number of frauds in test.
- (Its accuracy would be about 99.8%, which is exactly why accuracy is not in our tables.)

**Purpose:** it proves the problem. Any model's usefulness is measured against "does nothing and
looks 99.8% accurate".

## Baseline 2: Logistic regression (unweighted)

**What it does:** a linear model on the scaled features, no imbalance handling at all.
**Why this one:** it is simple and fast, and it shows how far plain, untuned modeling gets before any tricks.

**How to read the results table:**
- **0.5 threshold vs val-tuned threshold.** With rare fraud, an unweighted model rarely outputs
  probabilities above 0.5 for fraud, so the default threshold tends to give high precision but
  missed frauds. If the val-tuned row has noticeably higher recall (and lower precision), that
  confirms that **the threshold matters as much as the model**. That sets up step 2.
- **PR-AUC vs the random line** in `baseline_pr_curve.png`. The gap is the model's real skill.
- **Recall @ 90% precision.** If this is much lower than the recall at the tuned threshold,
  the model cannot reach high precision at useful recall, which is a sign a stronger model is needed.
- **Unstable thresholds.** If the validation-tuned threshold performs worse on test, that is
  drift or small-sample noise (few frauds in validation). Note it rather than hiding it.

## What the baseline tells us and what comes next

| Observation to make from your table | Leads to |
|---|---|
| Default threshold misses many frauds, tuned threshold catches more | Step 2: compare class weights, SMOTE and threshold tuning properly |
| Logistic regression PR-AUC is well above random but recall @ 90% precision is limited | Step 3: gradient-boosted trees can capture non-linear interactions |
| Test frauds are few, so metrics wobble | Report counts; consider time-based cross-validation folds in step 3 |

## Recorded findings

On 42,722 test transactions (52 frauds), the majority classifier missed all 52; its average precision was approximately the 0.0012 test prevalence. Unweighted logistic regression achieved PR-AUC 0.644 and recall 0.423 at 90% precision. At threshold 0.5 it caught 30 frauds with 12 false alarms; the validation-F1 threshold of 0.202 caught 33 with 21 false alarms, reducing test F1 from 0.638 to 0.623. This is a recall/precision trade-off, not evidence that tuning improved every metric. See [results](results/baselines.md).

The boosted-model comparison uses these ranking metrics as a baseline, but differences from this single short test period should not be assumed to generalize.

## Verification note

The code path (split, preprocessing, metrics, scripts, unit tests) was verified on synthetic data
with the same schema. All real numbers come from running on `creditcard.csv`.
