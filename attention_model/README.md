# CNN+LSTM with temporal attention (Model B)

`run_new_data_attention.py` trains and evaluates Model B: the baseline's frozen ResNet18 frame features and LSTM, with attention over every LSTM output in place of the final hidden state. Same frame sampling, fixed train / validation / test split, optimiser and metrics as the baseline.

It is the attention teammate's script with two changes, so that all team models run on one computer under one protocol:

1. The dataset is also looked for in `Baseline Model/cleaned_unified_deduplicated`.
2. Cross-validation keeps, in each fold, the epoch with the best validation accuracy, with the same seed as the final model. This is the same rule as Model A's cross-validation, so the models' cross-validation scores are directly comparable. The original script scored each fold at its last epoch.

Cross-validation uses `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the 144 training videos in manifest order, which gives the shared team folds in `Baseline Model/outputs/cv_folds.csv`.

## Run

From this folder:

```
python run_new_data_attention.py
```

The first run extracts the ResNet18 frame features into `outputs_attention/features/` (not committed; identical to `Baseline Model/outputs/features/`).

## Outputs

| File | Contents |
|---|---|
| `outputs_attention/cnn_lstm_attention_best.pt` | final model (epoch with the best validation accuracy) |
| `outputs_attention/run_log.txt` | full output: per-fold and mean ± std cross-validation, training curve, validation and test metrics, confusion matrices, misclassified videos, accuracy by source |
