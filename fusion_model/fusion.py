"""
Late fusion of the two shoplifting classifiers:

    pose model  (Approach A: hand-crafted pose features + HistGradientBoosting)
    baseline    (60 frames -> frozen ResNet18 -> LSTM)

Each model outputs P(shoplifting) for a clip and the fused probability is a weighted average:

    p_fused = w * p_pose + (1 - w) * p_cnn          (predict shoplifting if p_fused >= 0.5)

Evaluation follows the team protocol:
  * fixed train / validation / test split (144 / 30 / 33 clips, from splits/*.csv)
  * 5-fold stratified cross-validation inside the 144 training clips, with the shared folds
    in Baseline Model/outputs/cv_folds.csv (StratifiedKFold(n_splits=5, shuffle=True, random_state=42))

Step 1 - cross-validation on the training set.
    Out-of-fold probabilities for both models: each training clip is predicted by a model that
    did not train on it. Pose: from pose_model/02_features_ml.ipynb (results/approach_a_cv_oof.csv).
    CNN+LSTM: retrained here on each fold with Baseline Model/run_new_data.py's fit(), picking its
    checkpoint on the fixed validation set exactly like that script's own cross-validation.
    The fused score of fold k uses a weight chosen on the other four folds only (no peeking).

Step 2 - final weight and held-out evaluation.
    The final weight is chosen on the out-of-fold predictions of all 144 training clips (best F1,
    then ROC-AUC, then closest to 0.5) and applied to the two full models (trained on all of train)
    on the validation and test sets. The test set is only used for this final report.

Usage (from the repo root, with the pose_model venv):
    pose_model/.venv/Scripts/python.exe fusion_model/fusion.py
"""
import importlib.util
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

FUSION_DIR = Path(__file__).resolve().parent
ROOT = FUSION_DIR.parent
POSE_DIR = ROOT / "pose_model"
BASELINE_SCRIPT = ROOT / "Baseline Model" / "run_new_data.py"
CV_FOLDS_PATH = ROOT / "Baseline Model" / "outputs" / "cv_folds.csv"
RESULTS_DIR = FUSION_DIR / "results"
FIGURES_DIR = FUSION_DIR / "figures"

WEIGHTS = np.round(np.arange(0, 1.0001, 0.05), 2)  # candidate pose weights
THRESHOLD = 0.5
N_FOLDS = 5
SEED = 42

MODELS = ["pose", "cnn", "fused"]
# categorical slots 1-3 of the reference palette: pose, CNN+LSTM, fused
COLORS = {"pose": "#2a78d6", "cnn": "#eb6834", "fused": "#1baf7a"}
NAMES = {"pose": "Pose (gradient boosting)", "cnn": "CNN+LSTM baseline", "fused": "Late fusion"}


def load_baseline_module():
    """Import Baseline Model/run_new_data.py (its folder name has a space, so no plain import)."""
    spec = importlib.util.spec_from_file_location("run_new_data", BASELINE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_baseline_module()


# ---------- full models (trained on all 144 training clips): P(shoplifting) for every clip ----------
def pose_probabilities() -> pd.DataFrame:
    meta = json.loads((POSE_DIR / "models" / "approach_a_meta.json").read_text())
    model = joblib.load(POSE_DIR / "models" / "approach_a.joblib")
    feats = pd.read_csv(POSE_DIR / "results" / "clip_features.csv")
    out = feats[["relative_path", "label", "source", "split"]].copy()
    out["name"] = out["relative_path"].map(lambda p: Path(p).stem)
    out["p_pose"] = model.predict_proba(feats[meta["feature_names"]])[:, 1]
    return out


def cnn_probabilities(names) -> np.ndarray:
    rows = [{"name": n, "label": 0} for n in names]  # label is unused here
    _, probs = BASE.predict(torch.load(BASE.CHECKPOINT_PATH, weights_only=True), rows)
    return np.array(probs)


# ---------- out-of-fold probabilities on the training set ----------
def cnn_out_of_fold(train: pd.DataFrame) -> np.ndarray:
    """Retrain the CNN+LSTM on 4 folds, predict the 5th (same procedure as the baseline's own CV)."""
    rows = {r["name"]: r for r in BASE.load_manifest()}
    val_rows = [r for r in rows.values() if r["split"] == "val"]
    oof = np.zeros(len(train))
    for k in range(1, N_FOLDS + 1):
        held = (train["fold"] == k).to_numpy()
        state, _, best_epoch = BASE.fit([rows[n] for n in train.loc[~held, "name"]], val_rows, verbose=False)
        _, oof[held] = BASE.predict(state, [rows[n] for n in train.loc[held, "name"]])
        print(f"  CNN+LSTM fold {k}: trained on {(~held).sum()}, best epoch {best_epoch}", flush=True)
    return oof


# ---------- metrics and weight selection ----------
def metrics(y, p) -> dict:
    yhat = (np.asarray(p) >= THRESHOLD).astype(int)
    return {"n": len(y), "accuracy": accuracy_score(y, yhat),
            "precision": precision_score(y, yhat, zero_division=0),
            "recall": recall_score(y, yhat, zero_division=0),
            "f1": f1_score(y, yhat, zero_division=0),
            "roc_auc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan}


def fuse(df: pd.DataFrame, w: float) -> np.ndarray:
    return w * df["p_pose"].to_numpy() + (1 - w) * df["p_cnn"].to_numpy()


def sweep(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{"w_pose": w, **metrics(df["label"], fuse(df, w))} for w in WEIGHTS])


def choose_weight(df: pd.DataFrame) -> float:
    ranked = sweep(df)
    ranked["dist_from_half"] = (ranked["w_pose"] - 0.5).abs()
    ranked = ranked.sort_values(["f1", "roc_auc", "dist_from_half"], ascending=[False, False, True])
    return float(ranked.iloc[0]["w_pose"])


def cross_validate(train: pd.DataFrame) -> pd.DataFrame:
    """Per-fold scores of pose, CNN+LSTM and fusion; fold k's weight is picked on the other folds."""
    rows = []
    for k in range(1, N_FOLDS + 1):
        held, rest = train[train["fold"] == k], train[train["fold"] != k]
        w_k = choose_weight(rest)
        probs = {"pose": held["p_pose"], "cnn": held["p_cnn"], "fused": fuse(held, w_k)}
        for m in MODELS:
            rows.append({"fold": k, "model": NAMES[m], "w_pose": w_k if m == "fused" else np.nan,
                         **metrics(held["label"], probs[m])})
    return pd.DataFrame(rows)


# ---------- figures ----------
def plot_weight_sweep(cv_sweep, test_sweep, w_best):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, sw, title in [(axes[0], cv_sweep, "Training-set CV, out-of-fold (used to pick the weight)"),
                          (axes[1], test_sweep, "Test (for reference only)")]:
        ax.plot(sw["w_pose"], sw["f1"], color=COLORS["pose"], lw=2, marker="o", ms=4, label="F1")
        ax.plot(sw["w_pose"], sw["roc_auc"], color=COLORS["cnn"], lw=2, marker="o", ms=4, label="ROC-AUC")
        ax.axvline(w_best, ls="--", c="grey", lw=1)
        ax.text(w_best, 0.02, f" chosen w = {w_best:.2f}", color="#444", fontsize=9, va="bottom")
        ax.set(title=title, xlabel="weight on pose model  (0 = CNN+LSTM only, 1 = pose only)", ylim=(0, 1.05))
        ax.title.set_fontsize(10)
        ax.grid(alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("score")
    axes[0].legend(loc="lower left", frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "weight_sweep.png", dpi=150)
    plt.close(fig)


def plot_cv_folds(cv):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    x = np.arange(1, N_FOLDS + 1)
    for ax, metric in zip(axes, ["f1", "roc_auc"]):
        for i, m in enumerate(MODELS):
            vals = cv[cv["model"] == NAMES[m]].sort_values("fold")[metric]
            ax.bar(x + (i - 1) * 0.26, vals, width=0.24, color=COLORS[m], label=NAMES[m])
        ax.set(title=f"{'F1' if metric == 'f1' else 'ROC-AUC'} per fold (training-set CV)", xlabel="fold",
               xticks=x, ylim=(0, 1.05))
        ax.title.set_fontsize(10)
        ax.grid(axis="y", alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9, frameon=False)
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    plt.savefig(FIGURES_DIR / "cv_folds.png", dpi=150)
    plt.close(fig)


def plot_test_results(test):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, k in zip(axes[:3], MODELS):
        cm = confusion_matrix(test["label"], (test[f"p_{k}"] >= THRESHOLD).astype(int), labels=[0, 1])
        ax.imshow(cm, cmap="Blues", vmin=0)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, v, ha="center", va="center", fontsize=13,
                    color="white" if v > cm.max() / 2 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["normal", "shoplifting"],
               yticklabels=["normal", "shoplifting"], xlabel="predicted", ylabel="actual")
        ax.set_title(NAMES[k], fontsize=10)
    ax = axes[3]
    for k in MODELS:
        fpr, tpr, _ = roc_curve(test["label"], test[f"p_{k}"])
        ax.plot(fpr, tpr, color=COLORS[k], lw=2, ls="--" if k == "fused" else "-",
                label=f"{NAMES[k]} (AUC {roc_auc_score(test['label'], test[f'p_{k}']):.2f})")
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    ax.set(title="ROC curve (test)", xlabel="false positive rate", ylabel="true positive rate")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "test_confusion_and_roc.png", dpi=150)
    plt.close(fig)


def bootstrap_ci(test, n_boot=10_000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    y = test["label"].to_numpy()
    rows = []
    for k in MODELS:
        p = test[f"p_{k}"].to_numpy()
        accs, f1s = [], []
        for _ in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            accs.append(accuracy_score(y[i], p[i] >= THRESHOLD))
            f1s.append(f1_score(y[i], p[i] >= THRESHOLD, zero_division=0))
        for metric, vals, point in [("accuracy", accs, accuracy_score(y, p >= THRESHOLD)),
                                    ("f1", f1s, f1_score(y, p >= THRESHOLD))]:
            rows.append({"model": NAMES[k], "metric": metric, "value": point,
                         "ci_low": np.percentile(vals, 2.5), "ci_high": np.percentile(vals, 97.5)})
    return pd.DataFrame(rows)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    df = pose_probabilities()
    df["p_cnn"] = cnn_probabilities(df["name"])
    print(df.groupby("split").size().to_dict())

    # ---- step 1: 5-fold CV on the training set (shared team folds) ----
    folds = pd.read_csv(CV_FOLDS_PATH).set_index("name")["fold"]
    pose_oof = pd.read_csv(POSE_DIR / "results" / "approach_a_cv_oof.csv")
    pose_oof["name"] = pose_oof["relative_path"].map(lambda p: Path(p).stem)
    train = df[df.split == "train"][["name", "relative_path", "label", "source"]].reset_index(drop=True)
    train["fold"] = train["name"].map(folds)
    assert train["fold"].notna().all(), "a training clip is missing from cv_folds.csv"
    assert (train["fold"].to_numpy() == train["name"].map(pose_oof.set_index("name")["fold"]).to_numpy()).all(), \
        "pose out-of-fold predictions were made with different folds - re-run pose_model/02_features_ml.ipynb"
    train["p_pose"] = train["name"].map(pose_oof.set_index("name")["p_oof"])
    print(f"\n{N_FOLDS}-fold CV on the training set ({len(train)} clips); retraining CNN+LSTM per fold:")
    train["p_cnn"] = cnn_out_of_fold(train)

    cv = cross_validate(train)
    cv.to_csv(RESULTS_DIR / "cv_fold_metrics.csv", index=False)
    score_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    cv_summary = cv.groupby("model", sort=False)[score_cols].agg(["mean", lambda s: s.std(ddof=0)])
    cv_summary.columns = [f"{m}_{'mean' if s == 'mean' else 'std'}" for m, s in cv_summary.columns]
    cv_summary.to_csv(RESULTS_DIR / "cv_summary.csv")
    print("\nPer fold:\n" + cv.round(3).to_string(index=False))
    print("\nMean ± std across folds:")
    for model, r in cv_summary.iterrows():
        print(f"  {model:26s} " + "  ".join(f"{m}={r[m + '_mean']:.3f}±{r[m + '_std']:.3f}" for m in score_cols))

    # ---- step 2: final weight from all out-of-fold predictions ----
    w = choose_weight(train)
    cv_sweep = sweep(train)
    cv_sweep.to_csv(RESULTS_DIR / "weight_sweep_cv.csv", index=False)
    train["p_fused"] = fuse(train, w)
    train.to_csv(RESULTS_DIR / "cv_oof_predictions.csv", index=False)
    print(f"\nFinal pose weight (chosen on out-of-fold predictions of all {len(train)} training clips): "
          f"w = {w:.2f}  (CNN+LSTM weight = {1 - w:.2f})")
    print(cv_sweep.round(3).to_string(index=False))

    # ---- step 3: full models on validation and test ----
    val = df[df.split == "validation"].reset_index(drop=True)
    test = df[df.split == "test"].reset_index(drop=True)
    test_sweep = sweep(test)
    test_sweep.to_csv(RESULTS_DIR / "weight_sweep_test.csv", index=False)
    for d in (df, val, test):
        d["p_fused"] = fuse(d, w)
        d["pred_fused"] = (d["p_fused"] >= THRESHOLD).astype(int)
    df[["relative_path", "label", "source", "split", "p_pose", "p_cnn", "p_fused", "pred_fused"]].to_csv(
        RESULTS_DIR / "all_predictions.csv", index=False)

    rows = []
    for split_name, d in [("validation", val), ("test", test)]:
        for k in MODELS:
            rows.append({"split": split_name, "model": NAMES[k], **metrics(d["label"], d[f"p_{k}"])})
        rows.append({"split": split_name, "model": "Late fusion, equal weights (w = 0.5)",
                     **metrics(d["label"], fuse(d, 0.5))})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    print("\n" + table.round(3).to_string(index=False))

    by_source = pd.DataFrame([{"source": src, "model": NAMES[k], **metrics(g["label"], g[f"p_{k}"])}
                              for src, g in test.groupby("source") for k in MODELS])
    by_source.to_csv(RESULTS_DIR / "test_metrics_by_source.csv", index=False)
    print("\nTest set by source:\n" + by_source.round(3).to_string(index=False))

    ci = bootstrap_ci(test)
    ci.to_csv(RESULTS_DIR / "test_confidence_intervals.csv", index=False)
    print("\nTest 95% bootstrap CIs:\n" + ci.round(3).to_string(index=False))

    is_wrong = {k: (test[f"p_{k}"] >= THRESHOLD).astype(int) != test["label"] for k in MODELS}
    wrong = test[is_wrong["pose"] | is_wrong["cnn"] | is_wrong["fused"]]
    print("\nTest clips at least one model gets wrong:")
    print(wrong[["name", "source", "label", "p_pose", "p_cnn", "p_fused"]].round(3).to_string(index=False))

    plot_weight_sweep(cv_sweep, test_sweep, w)
    plot_cv_folds(cv)
    plot_test_results(test)

    (FUSION_DIR / "fusion_meta.json").write_text(json.dumps({
        "method": "late fusion: weighted average of P(shoplifting)",
        "formula": "p_fused = w_pose * p_pose + (1 - w_pose) * p_cnn",
        "w_pose": w, "w_cnn": round(1 - w, 2), "threshold": THRESHOLD,
        "chosen_by": "out-of-fold predictions on the 144 training clips (shared 5-fold CV): "
                     "F1, then ROC-AUC, then closest to 0.5",
        "cv_folds": "Baseline Model/outputs/cv_folds.csv",
        "pose_model": "pose_model/models/approach_a.joblib",
        "cnn_model": "Baseline Model/outputs/lstm_best.pt",
    }, indent=2))
    print(f"\nSaved results to {RESULTS_DIR} and figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
