# Late fusion: pose model + CNN+LSTM with attention (Model E)

Combines the pose model with the attention model by averaging their predicted probabilities:

```
p_fused = w * p_pose + (1 - w) * p_att        # shoplifting if p_fused >= 0.5
```

- **Pose model**: Approach A from `pose_model/` (hand-crafted pose features + HistGradientBoosting), `pose_model/models/approach_a.joblib`
- **Attention model**: Model B's architecture (frozen ResNet18 frame features + LSTM + temporal attention), copied from `run_new_data_attention.py` and trained here with the same settings: 30 epochs, Adam, learning rate 0.001, batch size 8, seed 42, keeping the epoch with the best validation accuracy. It uses the baseline's cached frame features in `Baseline Model/outputs/features/`.

## Evaluation protocol

Identical to Model D in `fusion_model/`: the weight sweep, weight choice, nested cross-validation, metrics and figures are imported from `fusion_model/fusion.py`, so Models D and E are fused the same way. Only the RGB branch differs.

1. **Cross-validation.** Each of the 144 training clips gets an out-of-fold probability from both models, using the shared folds in `Baseline Model/outputs/cv_folds.csv`. Pose probabilities come from `pose_model/results/approach_a_cv_oof.csv`; the attention model is retrained here on each fold, keeping its checkpoint on the fixed validation set like its final model. Fold *k*'s fusion score uses a weight chosen on the other four folds only.
2. **Final weight.** Chosen on the out-of-fold predictions of all 144 training clips: best F1, then ROC-AUC, then the weight closest to 0.5.
3. **Held-out evaluation.** The two full models (trained on all of train) are fused with that weight and scored on validation and test.

## Run

From the repo root:

```
python fusion_attention_model/fusion_attention.py
python fusion_attention_model/fusion_attention.py --attention-checkpoint path/to/weights.pt   # reuse saved weights
```

The same code and seed train slightly differently on different computers, so the cross-validation numbers and the chosen weight can vary between machines. The final attention weights are saved to `models/attention_lstm_best.pt`; pass them with `--attention-checkpoint` to reproduce the held-out results.

## Outputs

The same files as `fusion_model/`, written to this folder: `fusion_meta.json`, `results/` (per-fold and summary CV metrics, out-of-fold and full-model predictions, weight sweeps, validation and test metrics, test metrics by source, bootstrap CIs) and `figures/` (`cv_folds.png`, `weight_sweep.png`, `test_confusion_and_roc.png`). In the CSVs, `p_attention` is the attention model's probability.
