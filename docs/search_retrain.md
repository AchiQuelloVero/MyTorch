# Hyperparameter search: best trials retrained on the whole train.txt

`scripts/search.py --seed 42 --retrain 0 11 10 28 --final-epochs 60`: best validation accuracy first.

| trial | hidden | learning_rate | batch_size | dropout | l2 | best epoch | val_accuracy | val_loss | seconds |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 256-64 | 0.001 | 64 | 0.0 | 0.0001 | 35 of 40 | 93.18% | 0.1798 | 336 |
| 11 | 512-256-64 | 0.0018 | 64 | 0.42 | 8.7e-05 | 16 of 21 | 87.37% | 0.3163 | 474 |
| 28 | 128 | 0.0023 | 32 | 0.27 | 4e-05 | 30 of 35 | 87.07% | 0.3363 | 256 |
| 10 | 128-32 | 0.00052 | 32 | 0.14 | 0.001 | 21 of 26 | 85.59% | 0.3691 | 208 |
