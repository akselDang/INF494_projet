### mnist

| Model | Input | Parameters | Seeds | Val. accuracy (%) | Test accuracy (%) | Best epoch | Train acc., eval mode (%) | Seconds / epoch |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GCN (3 features) | graph, 3 features | 33,994 | 3 | 62.49 ± 3.30 | 63.04 ± 3.29 | 20.0 ± 0.0 | 62.66 ± 3.06 | 9 |
| GCN | graph, 5 features | 34,122 | 3 | 72.53 ± 6.44 | 72.39 ± 6.68 | 20.0 ± 0.0 | 72.21 ± 6.49 | 9 |
| GraphSAGE (raw features) | graph, 5 features, not standardised | 59,018 | 3 | 75.21 ± 2.71 | 75.76 ± 2.86 | 19.3 ± 0.6 | 75.07 ± 2.99 | 7 |
| GraphSAGE | graph, 5 features | 59,018 | 3 | 91.77 ± 0.28 | 92.29 ± 0.17 | 19.0 ± 1.7 | 92.03 ± 0.17 | 8 |
| LeNet-5 (CNN) | pixels | 61,706 | 3 | 99.07 ± 0.01 | 99.10 ± 0.10 | 17.7 ± 0.6 | 99.95 ± 0.03 | 3 |
| CNN | pixels | 421,642 | 3 | 99.16 ± 0.02 | 99.31 ± 0.08 | 15.7 ± 4.9 | 99.88 ± 0.10 | 3 |

Mean ± standard deviation over seeds. The best epoch is chosen on the validation set; the test set is evaluated once per run.
