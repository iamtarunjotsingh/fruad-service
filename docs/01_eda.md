# Step 1a: Exploratory Data Analysis

**Goal:** understand the data well enough to choose a split, metrics and features that won't mislead us.
**Run:** [notebooks/01_eda.ipynb](../notebooks/01_eda.ipynb) (or `python scripts/eda.py`). Plots are saved to `figures/`.

## The dataset in one paragraph

The ULB credit card dataset has two days of European card transactions. Each row has `Time`
(seconds since the first transaction), `Amount`, `Class` (1 = fraud), and 28 columns `V1..V28`
that are the output of a PCA transformation, so their original meaning is hidden for privacy.
It is publicly documented as roughly 285k rows with about 490 frauds (about 0.17%).
Check the overview table in the notebook against this.

## 1. Class balance

**What we check:** fraud rate and the legit-to-fraud ratio.
**Why it matters:** at about 1 fraud per 600 legit rows, three things change:
- Accuracy is useless (always predicting "legit" scores about 99.8%).
- Models trained with default settings tend to ignore the rare class.
- Metrics based on few positives are noisy, so we report counts, not just percentages.

**Decision:** use PR-AUC, recall at a fixed precision, and (later) a cost-based threshold.

## 2. Data quality

**What we check:** missing values and duplicate rows.
**Observed:** no missing values and 1,081 exact duplicate rows.
**Decision:** keep duplicates for this experiment; their cause cannot be determined from these features alone.
If you want to test their effect later, do it as a documented experiment, since removing rows
changes the class balance slightly.

## 3. Transaction amount

**What we check:** amount summary by class and the log-scale histogram (`figures/amount_by_class.png`).
**How to read it:** amounts are heavily right-skewed (many small purchases, a few very large).
If the fraud curve sits left of the legit curve, fraudsters favor small amounts (often card testing).
If the two curves overlap heavily, amount alone is a weak signal.
**Decision:** use `log1p(Amount)` so the model sees a well-behaved scale. Raw amounts are kept
available for the cost step, where a missed fraud costs more when the amount is larger.

## 4. Elapsed-hour cycle

**What we check:** fraud rate by elapsed-hour bucket modulo 24 (`figures/fraud_rate_by_hour.png`). This is not verified local time of day.
**How to read it:** with only two days of data, hourly rates rest on very few frauds per hour,
so spikes can be chance. Treat this as a hint, not a finding.
**Decision:** encode hour as `sin/cos` (so hour 23 is next to hour 0). Don't feed raw `Time` to the
model: it only counts seconds since the start, and a model that learns "fraud happens at second
X" would be useless on future data.

## 5. Which features separate fraud from legit?

**What we check:** the Kolmogorov-Smirnov (KS) statistic per feature, comparing the fraud and legit
distributions (0 = identical, 1 = completely separated), plus histograms of the top 3
(`figures/top_features.png`).
**How to read it:** a high KS means that single feature already separates the classes well.
On this dataset a handful of V-features (commonly V14, V17, V12, V10) stand out, so check your table.
**Caveat for the demo:** because V-features are anonymized, SHAP plots later will say "V14 pushed
the score up" and nothing more. That is a limit of this dataset, and it is worth stating in the README.

## 6. The time-based split

**What we do:** sort by `Time`, train on the first 70%, validate on the next 15%, test on the last 15%.
**Why not random:** a random split puts transactions from the same hour (and same fraud bursts)
in both train and test, which flatters the model. In production you predict the future from the
past, so the split must do the same.
**What to check in the split table:** the number of frauds in each split. The test set will have
only a few dozen frauds, so each missed fraud moves recall by a couple of percentage points.
Report TP/FP/FN counts alongside rates, and avoid reading much into small differences between models.
**Possible issue:** the fraud rate can differ between splits. That is real drift, and it is the
reason time-based evaluation is the honest choice.

## Summary of decisions

| Finding | Decision |
|---|---|
| About 0.17% fraud | PR-AUC, recall at fixed precision, cost-based threshold; no accuracy |
| Amount is right-skewed | `log1p(Amount)` feature |
| `Time` is an elapsed counter | Encode the elapsed-hour cycle; do not assume local time |
| Few frauds per split | Report counts; be careful about small metric differences |
| Data is clean (no nulls) | Minimal cleaning; scale features, fit on train only |

## Recorded findings

The saved run contains 284,807 transactions, including 492 frauds (0.1727%). Median fraud amount is $9.25 versus $22 for legitimate transactions, although fraud's mean is higher ($122.21 versus $88.29); amount distributions are skewed and overlap. V14, V10 and V12 have the highest observed KS statistics (0.843, 0.804 and 0.784), but the PCA components have no interpretable business meaning. These descriptive comparisons use the full dataset and are not independent evidence of predictive performance.

The highest elapsed-hour bucket is 2 (57 frauds in 3,328 transactions); elapsed time is not verified local clock time. The chronological test contains only 52 frauds across roughly the final six hours, making temporal and performance conclusions fragile. See [recorded tables](results/eda.md).
