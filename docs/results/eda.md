## 1. Dataset overview

| Item | Value |
|---|---|
| Rows | 284,807 |
| Columns | 31 |
| Fraud rows | 492 |
| Fraud rate | 0.1727% |
| Imbalance (legit : fraud) | 578 : 1 |
| Missing values | 0 |
| Duplicate rows | 1,081 |
| Time span (hours) | 48.0 |

## 2. Transaction amount by class

|       |   count |   mean |   50% |      max |
|:------|--------:|-------:|------:|---------:|
| legit |  284315 |  88.29 | 22    | 25691.2  |
| fraud |     492 | 122.21 |  9.25 |  2125.87 |

## 3. Time of day

Highest fraud-rate hours: hour 2 (1.71%), hour 4 (1.04%), hour 3 (0.49%)

## 4. Features that separate fraud from legit (KS statistic)

|     |    KS |
|:----|------:|
| V14 | 0.843 |
| V10 | 0.804 |
| V12 | 0.784 |
| V4  | 0.766 |
| V11 | 0.756 |
| V17 | 0.745 |
| V3  | 0.704 |
| V16 | 0.687 |
| V7  | 0.661 |
| V2  | 0.633 |

## 5. Time-based split (70 / 15 / 15)

| split      |   rows |   fraud | fraud_rate   |   start_h |   end_h |
|:-----------|-------:|--------:|:-------------|----------:|--------:|
| train      | 199364 |     384 | 0.193%       |       0   |    36.9 |
| validation |  42721 |      56 | 0.131%       |      36.9 |    42   |
| test       |  42722 |      52 | 0.122%       |      42   |    48   |
