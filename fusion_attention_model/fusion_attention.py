"""
Model E: late fusion of the pose model and the CNN+LSTM with temporal attention.

    pose model  (Approach A: hand-crafted pose features + HistGradientBoosting)
    attention   (60 frames -> frozen ResNet18 -> LSTM -> temporal attention; Model B)

    p_fused = w * p_pose + (1 - w) * p_att          (predict shoplifting if p_fused >= 0.5)

This is Model D (fusion_model/fusion.py) with the baseline CNN+LSTM swapped for the attention model.
The fusion itself (weight sweep, weight choice, nested cross-validation, metrics, bootstrap CIs,
figures) is imported from fusion.py unchanged, so Models D and E are fused in exactly the same way.
Nothing outside this folder is modified; all outputs are written here.

Attention branch: architecture copied verbatim from Model B's script (run_new_data_attention.py),
trained with the same loop as that script's train() and the baseline's fit(): 30 epochs, Adam,
lr 0.001, batch 8, seed 42, keeping the epoch with the best validation accuracy. It reuses the
baseline's cached ResNet18 frame features (Baseline Model/outputs/features), which Model B's script
extracts with identical frame sampling and preprocessing.
    * final model: trained on all 144 training clips, or loaded from --attention-checkpoint
    * out-of-fold: retrained on 4 of the 5 shared folds to predict the 5th, with the same settings and
      validation checkpoint as the final model (as fusion.py does for the baseline)

The same code and seed train slightly differently on different computers, so the trained weights are
saved to models/attention_lstm_best.pt; pass them (or Model B's own weights) with
--attention-checkpoint to reproduce the held-out results elsewhere.

Usage (from the repo root):
    python fusion_attention_model/fusion_attention.py
    python fusion_attention_model/fusion_attention.py --attention-checkpoint path/to/weights.pt
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

E_DIR = Path(__file__).resolve().parent
sys.dont_write_bytecode = True  # importing the other models' scripts must not write cache files into their folders
sys.path.insert(0, str(E_DIR.parent / "fusion_model"))
import fusion as F  # noqa: E402  (Model D; also loads Baseline Model/run_new_data.py as F.BASE)

BASE = F.BASE
ATT_CHECKPOINT = E_DIR / "models" / "attention_lstm_best.pt"

# fusion.py's helpers read these module settings: relabel its RGB branch ("cnn") as the attention model
# and send results and figures to this folder
F.NAMES = {**F.NAMES, "cnn": "CNN+LSTM + attention"}
F.RESULTS_DIR = E_DIR / "results"
F.FIGURES_DIR = E_DIR / "figures"


class AttentionLSTMClassifier(nn.Module):
    """Model B, copied verbatim from run_new_data_attention.py."""

    def __init__(self, input_size=512):
        super().__init__()
        self.lstm = nn.LSTM(input_size, BASE.HIDDEN_SIZE, BASE.NUM_LAYERS, batch_first=True)
        self.attention_projection = nn.Linear(BASE.HIDDEN_SIZE, BASE.HIDDEN_SIZE)
        self.attention_score = nn.Linear(BASE.HIDDEN_SIZE, 1, bias=False)
        self.dropout = nn.Dropout(BASE.DROPOUT)
        self.classifier = nn.Linear(BASE.HIDDEN_SIZE, 1)

    def forward(self, features):
        sequence_outputs, _ = self.lstm(features)
        scores = self.attention_score(torch.tanh(self.attention_projection(sequence_outputs))).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(sequence_outputs * weights.unsqueeze(-1), dim=1)
        return self.classifier(self.dropout(context)).squeeze(-1)


def fit_attention(train_rows, val_rows):
    """Train for 30 epochs and return the weights with the best validation accuracy
    (same loop as Model B's train() and the baseline's fit())."""
    torch.manual_seed(BASE.SEED)
    train_loader = DataLoader(BASE.FeatureDataset(train_rows), batch_size=BASE.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(BASE.FeatureDataset(val_rows), batch_size=BASE.BATCH_SIZE)
    model = AttentionLSTMClassifier()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=BASE.LR)

    best, best_state, best_epoch = -1.0, None, 0
    for epoch in range(1, BASE.EPOCHS + 1):
        BASE.run_epoch(model, train_loader, criterion, optimizer)
        _, val_acc = BASE.run_epoch(model, val_loader, criterion)
        if val_acc > best:
            best, best_state, best_epoch = val_acc, copy.deepcopy(model.state_dict()), epoch
    return best_state, best, best_epoch


def predict_attention(state, names) -> np.ndarray:
    model = AttentionLSTMClassifier()
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        return np.array([torch.sigmoid(model(torch.from_numpy(np.load(Path(BASE.FEATURES_DIR) / f"{n}.npy"))
                                             .float().unsqueeze(0))).item() for n in names])


def attention_model(checkpoint) -> dict:
    """Final attention weights: an existing checkpoint, or trained here on all 144 training clips."""
    if checkpoint:
        print(f"Attention model: loaded {checkpoint}")
        return torch.load(checkpoint, weights_only=True)
    rows = BASE.load_manifest()
    state, best, best_epoch = fit_attention([r for r in rows if r["split"] == "train"],
                                            [r for r in rows if r["split"] == "val"])
    ATT_CHECKPOINT.parent.mkdir(exist_ok=True)
    torch.save(state, ATT_CHECKPOINT)
    print(f"Attention model: trained on all training clips, best epoch {best_epoch} "
          f"(val acc {best:.3f}), saved to {ATT_CHECKPOINT}")
    return state


def attention_out_of_fold(train: pd.DataFrame) -> np.ndarray:
    """Retrain the attention model on 4 folds, predict the 5th (same procedure as its final model)."""
    rows = {r["name"]: r for r in BASE.load_manifest()}
    val_rows = [r for r in rows.values() if r["split"] == "val"]
    oof = np.zeros(len(train))
    for k in range(1, F.N_FOLDS + 1):
        held = (train["fold"] == k).to_numpy()
        state, _, best_epoch = fit_attention([rows[n] for n in train.loc[~held, "name"]], val_rows)
        oof[held] = predict_attention(state, train.loc[held, "name"])
        print(f"  attention fold {k}: trained on {(~held).sum()}, best epoch {best_epoch}", flush=True)
    return oof


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--attention-checkpoint", type=Path,
                        help="existing attention-model weights to use as the final model instead of training one")
    args = parser.parse_args()
    F.RESULTS_DIR.mkdir(exist_ok=True)
    F.FIGURES_DIR.mkdir(exist_ok=True)
    as_saved = {"p_cnn": "p_attention"}  # "cnn" is fusion.py's name for the RGB branch

    df = F.pose_probabilities()
    df["p_cnn"] = predict_attention(attention_model(args.attention_checkpoint), df["name"])
    print(df.groupby("split").size().to_dict())

    # ---- step 1: 5-fold CV on the training set (shared team folds) ----
    folds = pd.read_csv(F.CV_FOLDS_PATH).set_index("name")["fold"]
    pose_oof = pd.read_csv(F.POSE_DIR / "results" / "approach_a_cv_oof.csv")
    pose_oof["name"] = pose_oof["relative_path"].map(lambda p: Path(p).stem)
    train = df[df.split == "train"][["name", "relative_path", "label", "source"]].reset_index(drop=True)
    train["fold"] = train["name"].map(folds)
    assert train["fold"].notna().all(), "a training clip is missing from cv_folds.csv"
    assert (train["fold"].to_numpy() == train["name"].map(pose_oof.set_index("name")["fold"]).to_numpy()).all(), \
        "pose out-of-fold predictions were made with different folds - re-run pose_model/02_features_ml.ipynb"
    train["p_pose"] = train["name"].map(pose_oof.set_index("name")["p_oof"])
    print(f"\n{F.N_FOLDS}-fold CV on the training set ({len(train)} clips); retraining the attention model per fold:")
    train["p_cnn"] = attention_out_of_fold(train)

    cv = F.cross_validate(train)
    cv.to_csv(F.RESULTS_DIR / "cv_fold_metrics.csv", index=False)
    score_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    cv_summary = cv.groupby("model", sort=False)[score_cols].agg(["mean", lambda s: s.std(ddof=0)])
    cv_summary.columns = [f"{m}_{'mean' if s == 'mean' else 'std'}" for m, s in cv_summary.columns]
    cv_summary.to_csv(F.RESULTS_DIR / "cv_summary.csv")
    print("\nPer fold:\n" + cv.round(3).to_string(index=False))
    print("\nMean ± std across folds:")
    for model, r in cv_summary.iterrows():
        print(f"  {model:26s} " + "  ".join(f"{m}={r[m + '_mean']:.3f}±{r[m + '_std']:.3f}" for m in score_cols))

    # ---- step 2: final weight from all out-of-fold predictions ----
    w = F.choose_weight(train)
    cv_sweep = F.sweep(train)
    cv_sweep.to_csv(F.RESULTS_DIR / "weight_sweep_cv.csv", index=False)
    train["p_fused"] = F.fuse(train, w)
    train.rename(columns=as_saved).to_csv(F.RESULTS_DIR / "cv_oof_predictions.csv", index=False)
    print(f"\nFinal pose weight (chosen on out-of-fold predictions of all {len(train)} training clips): "
          f"w = {w:.2f}  (attention weight = {1 - w:.2f})")
    print(cv_sweep.round(3).to_string(index=False))

    # ---- step 3: full models on validation and test ----
    val = df[df.split == "validation"].reset_index(drop=True)
    test = df[df.split == "test"].reset_index(drop=True)
    test_sweep = F.sweep(test)
    test_sweep.to_csv(F.RESULTS_DIR / "weight_sweep_test.csv", index=False)
    for d in (df, val, test):
        d["p_fused"] = F.fuse(d, w)
        d["pred_fused"] = (d["p_fused"] >= F.THRESHOLD).astype(int)
    df[["relative_path", "label", "source", "split", "p_pose", "p_cnn", "p_fused", "pred_fused"]].rename(
        columns=as_saved).to_csv(F.RESULTS_DIR / "all_predictions.csv", index=False)

    rows = []
    for split_name, d in [("validation", val), ("test", test)]:
        for k in F.MODELS:
            rows.append({"split": split_name, "model": F.NAMES[k], **F.metrics(d["label"], d[f"p_{k}"])})
        rows.append({"split": split_name, "model": "Late fusion, equal weights (w = 0.5)",
                     **F.metrics(d["label"], F.fuse(d, 0.5))})
    table = pd.DataFrame(rows)
    table.to_csv(F.RESULTS_DIR / "metrics.csv", index=False)
    print("\n" + table.round(3).to_string(index=False))

    by_source = pd.DataFrame([{"source": src, "model": F.NAMES[k], **F.metrics(g["label"], g[f"p_{k}"])}
                              for src, g in test.groupby("source") for k in F.MODELS])
    by_source.to_csv(F.RESULTS_DIR / "test_metrics_by_source.csv", index=False)
    print("\nTest set by source:\n" + by_source.round(3).to_string(index=False))

    ci = F.bootstrap_ci(test)
    ci.to_csv(F.RESULTS_DIR / "test_confidence_intervals.csv", index=False)
    print("\nTest 95% bootstrap CIs:\n" + ci.round(3).to_string(index=False))

    is_wrong = {k: (test[f"p_{k}"] >= F.THRESHOLD).astype(int) != test["label"] for k in F.MODELS}
    wrong = test[is_wrong["pose"] | is_wrong["cnn"] | is_wrong["fused"]]
    print("\nTest clips at least one model gets wrong:")
    print(wrong[["name", "source", "label", "p_pose", "p_cnn", "p_fused"]].rename(columns=as_saved)
          .round(3).to_string(index=False))

    F.plot_weight_sweep(cv_sweep, test_sweep, w)
    F.plot_cv_folds(cv)
    F.plot_test_results(test)

    (E_DIR / "fusion_meta.json").write_text(json.dumps({
        "method": "late fusion: weighted average of P(shoplifting)",
        "formula": "p_fused = w_pose * p_pose + (1 - w_pose) * p_attention",
        "w_pose": w, "w_attention": round(1 - w, 2), "threshold": F.THRESHOLD,
        "chosen_by": "out-of-fold predictions on the 144 training clips (shared 5-fold CV): "
                     "F1, then ROC-AUC, then closest to 0.5",
        "cv_folds": "Baseline Model/outputs/cv_folds.csv",
        "pose_model": "pose_model/models/approach_a.joblib",
        "attention_model": (args.attention_checkpoint or ATT_CHECKPOINT.relative_to(F.ROOT)).as_posix(),
    }, indent=2))
    print(f"\nSaved results to {F.RESULTS_DIR} and figures to {F.FIGURES_DIR}")


if __name__ == "__main__":
    main()
