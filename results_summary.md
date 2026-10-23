# Model comparison (out-of-sample)

IC = mean daily Spearman rank correlation between prediction and realised 20d excess return. t = Newey-West t-stat. Q5-Q1 = top-quintile minus bottom-quintile realised return, before costs. Validation numbers were used for tuning; **test is the honest number**.

| model         | mode    |   val IC |   val t |   test IC |   test t |   test ICIR |   test IC>0 days |   test Q5-Q1 /20d |   test Q5-Q1 ann. (gross) |   test dates |
|:--------------|:--------|---------:|--------:|----------:|---------:|------------:|-----------------:|------------------:|--------------------------:|-------------:|
| Random Forest | wf      |   0.0940 |  3.4036 |    0.0208 |   0.9799 |      0.5650 |           0.5653 |            0.0131 |                    0.1655 |          513 |
| XGBoost       | wf      |   0.0965 |  3.8741 |    0.0183 |   0.8330 |      0.4755 |           0.5419 |            0.0133 |                    0.1677 |          513 |
| LSTM          | wf      |   0.0968 |  3.8378 |    0.0137 |   0.7393 |      0.4142 |           0.5536 |            0.0097 |                    0.1222 |          513 |
| Ridge         | wf      |   0.1051 |  4.0066 |    0.0100 |   0.6382 |      0.3330 |           0.5478 |            0.0094 |                    0.1183 |          513 |
| RNN           | wf      |   0.1080 |  3.7053 |    0.0092 |   0.5293 |      0.2856 |           0.5341 |            0.0102 |                    0.1287 |          513 |
| ARIMA         | rolling |   0.0788 |  3.2678 |    0.0072 |   0.2861 |      0.1656 |           0.5728 |            0.0040 |                    0.0500 |          103 |
| ElasticNet    | wf      |   0.1066 |  3.7876 |    0.0038 |   0.2006 |      0.1090 |           0.5556 |            0.0073 |                    0.0921 |          513 |