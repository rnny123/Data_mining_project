# Report figures

Builds the Results-section charts from the outputs the models already saved. Nothing is retrained.

```
python report_figures/make_figures.py
```

Before plotting, the script checks that the error counts and confusion matrices it rebuilds match Tables 5.3 and 5.7 to 5.11 in the report, and that Model B's checkpoint reproduces its saved test probabilities. It stops if anything differs.

| File | What it shows | Data source |
|---|---|---|
| `figures/fig_error_types.png` | False alarms and missed thefts per model, test set and cross-validation | `fusion_model/results`, `fusion_attention_model/results` (nested CV weights per fold) |
| `figures/fig_confusion_A.png` to `fig_confusion_E.png` | Test confusion matrix of each model (replaces Tables 5.7 to 5.11) | same prediction files |
| `figures/fig_feature_importance.png` | Top 10 pose features by permutation importance, with Spearman ρ against tracking coverage | `pose_model/results/approach_a_coverage_check.csv` |
| `figures/fig_attention_weights.png` | Model B's attention weight on each of the 60 frames, for all 33 test videos | recomputed from `attention_model/outputs_attention/cnn_lstm_attention_best.pt` and cached features |
| `figures/attention_weights_test.csv` | The attention weights behind the last figure | as above |
