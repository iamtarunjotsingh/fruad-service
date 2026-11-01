Test set: 42,722 rows, 52 fraud (0.122%). Random-guess PR-AUC equals the fraud rate, **0.001**.

| model                            | threshold   |   thr value |   PR-AUC |   ROC-AUC |   recall@90%prec |   precision |   recall |    F1 |   TP |   FP |   FN |
|:---------------------------------|:------------|------------:|---------:|----------:|-----------------:|------------:|---------:|------:|-----:|-----:|-----:|
| Majority class                   | 0.5         |       0.5   |    0.001 |     0.5   |            0     |       0     |    0     | 0     |    0 |    0 |   52 |
| Majority class                   | val-tuned   |       0.5   |    0.001 |     0.5   |            0     |       0     |    0     | 0     |    0 |    0 |   52 |
| Logistic regression (unweighted) | 0.5         |       0.5   |    0.644 |     0.964 |            0.423 |       0.714 |    0.577 | 0.638 |   30 |   12 |   22 |
| Logistic regression (unweighted) | val-tuned   |       0.202 |    0.644 |     0.964 |            0.423 |       0.611 |    0.635 | 0.623 |   33 |   21 |   19 |