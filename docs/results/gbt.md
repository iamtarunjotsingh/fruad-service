Test: 42,722 rows, 52 fraud. Cost assumptions: missed fraud = its Amount; false alarm = $5.

Best LightGBM params: `{'num_leaves': 63, 'min_child_samples': 20, 'subsample': 0.9, 'subsample_freq': 1, 'colsample_bytree': 0.8, 'reg_lambda': 1.0, 'scale_pos_weight': 22.763503318543332}`, best iteration 530.

### Ranking quality

| model               |   PR-AUC | 95% CI         | diff vs LR (95% CI)   |   recall@90%prec |   recall @ top 85 alerts |   precision @ top 85 alerts |   ROC-AUC |
|:--------------------|---------:|:---------------|:----------------------|-----------------:|-------------------------:|----------------------------:|----------:|
| Logistic regression |    0.644 | 0.514 to 0.767 | +0.000 to +0.000      |            0.423 |                    0.692 |                       0.424 |     0.964 |
| LightGBM            |    0.751 | 0.635 to 0.855 | +0.052 to +0.167      |            0.712 |                    0.75  |                       0.459 |     0.974 |

### Cost-based decisions on test

| rule                                     |   threshold |   flagged |   TP |    FP |   FN |   missed $ |   review $ |   total cost $ | savings vs nothing   |
|:-----------------------------------------|------------:|----------:|-----:|------:|-----:|-----------:|-----------:|---------------:|:---------------------|
| Flag nothing                             |    nan      |         0 |    0 |     0 |   52 |    6168.88 |          0 |        6168.88 | 0.0%                 |
| Flag everything                          |    nan      |     42722 |   52 | 42670 |    0 |       0    |     213350 |      213350    | -3358.5%             |
| Logistic regression + cost threshold     |      0.0612 |        94 |   38 |    56 |   14 |    2450.29 |        280 |        2730.29 | 55.7%                |
| LightGBM, threshold 0.5                  |      0.5    |        62 |   39 |    23 |   13 |    2372.4  |        115 |        2487.4  | 59.7%                |
| LightGBM, F1-tuned threshold             |      0.9861 |        37 |   36 |     1 |   16 |    2530.17 |          5 |        2535.17 | 58.9%                |
| LightGBM, cost-tuned threshold           |      0.0971 |        69 |   39 |    30 |   13 |    2372.4  |        150 |        2522.4  | 59.1%                |
| LightGBM, amount-aware rule (calibrated) |    nan      |        62 |   37 |    25 |   15 |    2372.41 |        125 |        2497.41 | 59.5%                |

### Sensitivity to false-alarm cost

|   c_fp |   threshold |   flagged |   tp |   fp |   fn |    cost |   no_model_cost |   savings_pct |
|-------:|------------:|----------:|-----:|-----:|-----:|--------:|----------------:|--------------:|
|      1 |      0.0044 |        99 |   39 |   60 |   13 | 2432.4  |         6168.88 |          60.6 |
|      2 |      0.0971 |        69 |   39 |   30 |   13 | 2432.4  |         6168.88 |          60.6 |
|      5 |      0.0971 |        69 |   39 |   30 |   13 | 2522.4  |         6168.88 |          59.1 |
|     10 |      0.0971 |        69 |   39 |   30 |   13 | 2672.4  |         6168.88 |          56.7 |
|     20 |      0.9482 |        48 |   37 |   11 |   15 | 2710.19 |         6168.88 |          56.1 |
|     50 |      0.9861 |        37 |   36 |    1 |   16 | 2580.17 |         6168.88 |          58.2 |

### Calibration

|                 |   mean score |     Brier |
|:----------------|-------------:|----------:|
| raw             |      0.00144 |   0.00073 |
| calibrated      |      0.00135 |   0.0005  |
| true fraud rate |      0.00122 | nan       |

### Top hyperparameter trials (validation)

|   num_leaves |   min_child_samples |   subsample |   subsample_freq |   colsample_bytree |   reg_lambda | spw_name      |   scale_pos_weight |   best_iter |   val_PR_AUC |
|-------------:|--------------------:|------------:|-----------------:|-------------------:|-------------:|:--------------|-------------------:|------------:|-------------:|
|           63 |                  20 |         0.9 |                1 |                0.8 |            1 | sqrt(neg/pos) |               22.8 |         530 |       0.8689 |
|           15 |                  50 |         0.7 |                1 |                1   |           10 | neg/pos       |              518.2 |         328 |       0.8523 |
|           63 |                  50 |         1   |                1 |                1   |           10 | none (1)      |                1   |           9 |       0.8509 |
|           31 |                  20 |         0.7 |                1 |                0.8 |           10 | none (1)      |                1   |         166 |       0.8474 |
|           63 |                  50 |         0.7 |                1 |                1   |            1 | none (1)      |                1   |         200 |       0.8472 |