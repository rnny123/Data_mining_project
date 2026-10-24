# Late fusion: pose model + CNN+LSTM baseline

Combines the two existing shoplifting classifiers by averaging their predicted probabilities:

```
p_fused = w * p_pose + (1 - w) * p_cnn        # shoplifting if p_fused >= 0.5
```

- **Pose model**: Approach A from `pose_model/` (hand-crafted pose features + HistGradientBoosting), `pose_model/models/approach_a.joblib`
- **CNN+LSTM baseline**: from `Baseline Model/` (ResNet18 frame features + LSTM), `Baseline Model/outputs/lstm_best.pt`

## Evaluation protocol (shared with the other team models)

- Fixed **train / validation / test** split: 144 / 30 / 33 clips, from `splits/*.csv`.
- **5-fold stratified cross-validation inside the 144 training clips**, using the shared folds in `Baseline Model/outputs/cv_folds.csv` (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on `splits/train.csv`). Validation and test clips never enter the folds.

Steps:

1. **Cross-validation.** Each training clip gets an *out-of-fold* probability from each model, meaning it was predicted by a model that did not train on it.
   - The pose probabilities come from `pose_model/02_features_ml.ipynb` (`results/approach_a_cv_oof.csv`).
   - The CNN+LSTM is retrained here on each fold with `fit()` from `Baseline Model/run_new_data.py`, which keeps its checkpoint on the fixed validation set, as the baseline's own CV does.
   - Fold *k*'s fusion score uses a weight chosen on the other four folds only.
2. **Final weight.** The final weight is chosen on the out-of-fold predictions of all 144 training clips: best F1, then ROC-AUC, then the weight closest to 0.5.
3. **Held-out evaluation.** The two full models (trained on all of train) are fused with that weight and scored on validation and test. The test set is used only for this final report.

## Run

Run notebook `pose_model/02_features_ml.ipynb` first, since it writes the pose out-of-fold predictions. Then, from the repo root:

```
pose_model/.venv/Scripts/python.exe fusion_model/fusion.py
```

## Outputs

| File | Contents |
|---|---|
| `fusion_meta.json` | chosen weight and how it was picked |
| `results/cv_fold_metrics.csv` | per-fold scores of pose, CNN+LSTM and fusion |
| `results/cv_summary.csv` | mean and std across the 5 folds (std uses `ddof=0`, like `np.std`) |
| `results/cv_oof_predictions.csv` | out-of-fold p_pose, p_cnn and p_fused for the 144 training clips |
| `results/weight_sweep_cv.csv`, `results/weight_sweep_test.csv` | metrics for every weight from 0 to 1 in steps of 0.05 |
| `results/metrics.csv` | full models on validation and test |
| `results/all_predictions.csv` | full-model p_pose, p_cnn, p_fused for all 207 clips |
| `results/test_metrics_by_source.csv` | test metrics split by dataset source (staged vs DCSASS) |
| `results/test_confidence_intervals.csv` | 95% bootstrap CIs for accuracy and F1 on test |
| `figures/cv_folds.png` | F1 and ROC-AUC per fold for the three models |
| `figures/weight_sweep.png` | F1 and ROC-AUC against the weight (CV and test) |
| `figures/test_confusion_and_roc.png` | confusion matrices and ROC curves on test |
