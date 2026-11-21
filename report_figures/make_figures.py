"""Build the Results-section figures for the report from the saved model outputs.

Every number is read from the files the models wrote, and checked against the
tables in the report before plotting. Nothing is retrained.

Run from anywhere:  python report_figures/make_figures.py
Outputs go to report_figures/figures/ (300 dpi PNG).
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "report_figures", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

FALSE_ALARM = "#E69F00"  # Okabe-Ito orange, colour-blind safe
MISSED = "#0072B2"  # Okabe-Ito blue
POSE_BAND = "#EEF0EC"
TEXT = "#222222"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

MODELS = ["A", "B", "C", "D", "E"]
MODEL_LABELS = {
    "A": "A\nBaseline",
    "B": "B\n+ Attention",
    "C": "C\nPose only",
    "D": "D\n+ Pose",
    "E": "E\n+ Attention\n+ Pose",
}
MODEL_TITLES = {
    "A": "A: CNN-LSTM baseline",
    "B": "B: CNN-LSTM + attention",
    "C": "C: Pose only",
    "D": "D: CNN-LSTM + pose",
    "E": "E: CNN-LSTM + attention + pose",
}

# Values printed in the report, used only to check the data before plotting.
REPORT_TABLE_5_3 = {  # model: (test FA, test missed, CV FA, CV missed)
    "A": (6, 1, 34, 7),
    "B": (5, 2, 22, 10),
    "C": (2, 3, 17, 16),
    "D": (2, 2, 24, 13),
    "E": (2, 3, 22, 13),
}
REPORT_CONFUSION = {  # model: [[TN, FP], [FN, TP]], Tables 5.7 to 5.11
    "A": [[9, 6], [1, 17]],
    "B": [[10, 5], [2, 16]],
    "C": [[13, 2], [3, 15]],
    "D": [[13, 2], [2, 16]],
    "E": [[13, 2], [3, 15]],
}


def load_predictions():
    """Return test and cross-validation predictions (0/1) for every model."""
    fusion = pd.read_csv(os.path.join(ROOT, "fusion_model/results/all_predictions.csv"))
    fusion_att = pd.read_csv(os.path.join(ROOT, "fusion_attention_model/results/all_predictions.csv"))
    fusion = fusion[fusion.split == "test"].set_index("relative_path")
    fusion_att = fusion_att[fusion_att.split == "test"].set_index("relative_path").loc[fusion.index]

    test = pd.DataFrame({
        "label": fusion.label,
        "A": (fusion.p_cnn >= 0.5).astype(int),
        "B": (fusion_att.p_attention >= 0.5).astype(int),
        "C": (fusion.p_pose >= 0.5).astype(int),
        "D": fusion.pred_fused,
        "E": fusion_att.pred_fused,
    })

    cv = {}
    for folder, rgb, rgb_model, fused_model in [
        ("fusion_model", "p_cnn", "A", "D"),
        ("fusion_attention_model", "p_attention", "B", "E"),
    ]:
        oof = pd.read_csv(os.path.join(ROOT, folder, "results/cv_oof_predictions.csv"))
        folds = pd.read_csv(os.path.join(ROOT, folder, "results/cv_fold_metrics.csv"))
        # Nested CV: each fold uses the weight chosen on the other four folds.
        weights = folds[folds.model == "Late fusion"][["fold", "w_pose"]]
        oof = oof.merge(weights, on="fold")
        fused = oof.w_pose * oof.p_pose + (1 - oof.w_pose) * oof[rgb]
        cv["label"] = oof.label.values
        cv["C"] = (oof.p_pose >= 0.5).astype(int).values
        cv[rgb_model] = (oof[rgb] >= 0.5).astype(int).values
        cv[fused_model] = (fused >= 0.5).astype(int).values
    return test, pd.DataFrame(cv)


def error_counts(labels, preds):
    false_alarms = int(((preds == 1) & (labels == 0)).sum())
    missed = int(((preds == 0) & (labels == 1)).sum())
    return false_alarms, missed


def check_against_report(test, cv):
    for m in MODELS:
        counts = error_counts(test.label, test[m]) + error_counts(cv.label, cv[m])
        assert counts == REPORT_TABLE_5_3[m], f"Model {m}: {counts} != Table 5.3 {REPORT_TABLE_5_3[m]}"
        tn = int(((test[m] == 0) & (test.label == 0)).sum())
        tp = int(((test[m] == 1) & (test.label == 1)).sum())
        cm = [[tn, counts[0]], [counts[1], tp]]
        assert cm == REPORT_CONFUSION[m], f"Model {m}: {cm} != report {REPORT_CONFUSION[m]}"
    print("Checked: error counts and confusion matrices match Tables 5.3 and 5.7-5.11.")


def figure_error_types(test, cv):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
    panels = [
        (axes[0], test, "Test set (33 videos)"),
        (axes[1], cv, "Cross-validation, summed over 5 folds (144 videos)"),
    ]
    x = np.arange(len(MODELS))
    width = 0.38
    for ax, data, title in panels:
        counts = [error_counts(data.label, data[m]) for m in MODELS]
        fa = [c[0] for c in counts]
        mi = [c[1] for c in counts]
        top = max(f + m for f, m in zip(fa, mi))
        ax.axvspan(1.5, 4.5, color=POSE_BAND, zorder=0)
        ax.text(3, top * 1.18, "Models that use pose", ha="center", va="bottom",
                fontsize=10, color="#555555")
        bars_fa = ax.bar(x - width / 2, fa, width, color=FALSE_ALARM, label="False alarms", zorder=2)
        bars_mi = ax.bar(x + width / 2, mi, width, color=MISSED, label="Missed thefts", zorder=2)
        for bars in (bars_fa, bars_mi):
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + top * 0.015,
                        str(int(bar.get_height())), ha="center", va="bottom", fontsize=10, color=TEXT)
        for i, (f, m) in enumerate(zip(fa, mi)):
            ax.text(i, -top * 0.30, f"Total {f + m}", ha="center", va="top",
                    fontsize=10, fontweight="bold", color=TEXT)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=9.5)
        ax.set_ylim(0, top * 1.32)
        ax.set_ylabel("Number of videos")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.yaxis.grid(True, color="#E5E5E5", zorder=0)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(w_pad=3)
    fig.subplots_adjust(bottom=0.27, top=0.86)
    path = os.path.join(OUT_DIR, "fig_error_types.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def figures_confusion_matrices():
    """One confusion matrix per model (sklearn display), to replace Tables 5.7 to 5.11."""
    from sklearn.metrics import ConfusionMatrixDisplay

    # Colour by outcome, not by count: green cells are correct, red cells are errors.
    outcome_cmap = ListedColormap(["#C62828", "#66BB6A"])
    is_correct = np.array([[1, 0], [0, 1]])
    paths = []
    for m in MODELS:
        fig, ax = plt.subplots(figsize=(4.6, 4))
        display = ConfusionMatrixDisplay(
            confusion_matrix=np.array(REPORT_CONFUSION[m]),
            display_labels=["Normal", "Shoplifting"],
        )
        display.plot(ax=ax, cmap=outcome_cmap, colorbar=False, im_kw={"vmin": 0, "vmax": 1})
        display.im_.set_data(is_correct)
        for (r, c), text in np.ndenumerate(display.text_):
            text.set_color("black" if is_correct[r, c] else "white")
            text.set_fontsize(14)
        for side in ax.spines.values():
            side.set_visible(True)
        ax.set_title(MODEL_TITLES[m])
        fig.tight_layout()
        path = os.path.join(OUT_DIR, f"fig_confusion_{m}.png")
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


FEATURE_PARTS = {
    "rwrist": "Right wrist", "lwrist": "Left wrist", "relbow": "Right elbow", "lelbow": "Left elbow",
    "body": "Body", "crouch": "Crouch",
}
STAT_PARTS = {"mean": "mean", "std": "std", "p90": "90th pct", "p10": "10th pct",
              "max": "max", "min": "min"}


def readable_feature(name):
    parts = name.split("_")
    stat = STAT_PARTS.get(parts[-1], parts[-1])
    body = parts[:-1]
    joint = FEATURE_PARTS.get(body[0], body[0])
    rest = " ".join(body[1:])
    rest = (rest.replace("midhip", "to hip-midpoint distance").replace("rhip", "to right-hip distance")
            .replace("angle", "angle").replace("speed", "speed").replace("height", "height"))
    return f"{joint} {rest} ({stat})".replace("  ", " ").strip()


def figure_feature_importance():
    imp = pd.read_csv(os.path.join(ROOT, "pose_model/results/approach_a_coverage_check.csv"))
    imp = imp.sort_values("importance").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    top5 = set(imp.nlargest(5, "importance").feature)
    colours = ["#2E5E58" if f in top5 else "#A9C4BF" for f in imp.feature]
    ax.barh([readable_feature(f) for f in imp.feature], imp.importance, color=colours, zorder=2)
    xmax = imp.importance.max()
    for i, row in imp.iterrows():
        ax.text(row.importance + xmax * 0.01, i, f"{row.importance:.3f}", va="center", fontsize=9)
        ax.text(xmax * 1.30, i, f"{row.spearman_rho_with_coverage:+.2f}", va="center", ha="right",
                fontsize=9, color="#555555")
    ax.text(xmax * 1.30, len(imp) - 0.35, "ρ with\ntracking", ha="right", va="bottom", fontsize=9,
            color="#555555", fontweight="bold")
    ax.set_xlim(0, xmax * 1.32)
    ax.set_xlabel("Drop in validation ROC-AUC when the feature is shuffled (mean of 30 repeats)")
    ax.xaxis.grid(True, color="#E5E5E5", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(color="#2E5E58", label="Top 5 (named in Section 5.1.2)"),
                       Patch(color="#A9C4BF", label="Ranks 6 to 10")],
              frameon=False, loc="lower right", bbox_to_anchor=(0.86, 0.0), fontsize=9)
    ax.set_title("Pose features the model relies on most (Model C)", loc="left", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig_feature_importance.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def attention_weights():
    """Recompute Model B's attention weights for every test video from its saved checkpoint."""
    sys.path.insert(0, os.path.join(ROOT, "attention_model"))
    import run_new_data_attention as att  # noqa: E402

    model = att.AttentionLSTMClassifier()
    model.load_state_dict(torch.load(att.CHECKPOINT_PATH, weights_only=True))
    model.eval()
    records = []
    for row in att.load_manifest():
        if row["split"] != "test":
            continue
        x = torch.from_numpy(np.load(att.feature_path(row))).float()[None]
        with torch.no_grad():
            outputs, _ = model.lstm(x)
            scores = model.attention_score(torch.tanh(model.attention_projection(outputs))).squeeze(-1)
            weights = torch.softmax(scores, dim=1)[0].numpy()
            prob = torch.sigmoid(model(x)).item()
        records.append({"name": row["name"], "relative_path": row["relative_path"],
                        "label": row["label"], "p_attention": prob, "weights": weights})

    # Check the checkpoint reproduces the probabilities used in the report.
    saved = pd.read_csv(os.path.join(ROOT, "fusion_attention_model/results/all_predictions.csv"))
    saved = saved.set_index("relative_path").p_attention
    for r in records:
        assert abs(saved[r["relative_path"]] - r["p_attention"]) < 1e-4, r["name"]
    return records


def figure_attention(records):
    n_frames = len(records[0]["weights"])
    uniform = 1 / n_frames
    order = sorted(records, key=lambda r: (r["label"], r["p_attention"]))
    matrix = np.array([r["weights"] / uniform for r in order])

    entropy = np.array([-(r["weights"] * np.log(r["weights"])).sum() / np.log(n_frames) for r in records])
    top5 = np.array([np.sort(r["weights"])[-5:].sum() for r in records])

    pd.DataFrame(
        [{"name": r["name"], "label": r["label"], "p_attention": r["p_attention"],
          **{f"w{t + 1}": w for t, w in enumerate(r["weights"])}} for r in records]
    ).to_csv(os.path.join(OUT_DIR, "attention_weights_test.csv"), index=False)

    fig = plt.figure(figsize=(10, 7.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[2.3, 1], width_ratios=[40, 1], hspace=0.45, wspace=0.03)
    ax = fig.add_subplot(grid[0, 0])
    cax = fig.add_subplot(grid[0, 1])
    norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=max(2.0, matrix.max()))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest")
    n_normal = sum(r["label"] == 0 for r in order)
    ax.axhline(n_normal - 0.5, color="black", lw=1.2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{r['name'].replace('_x264', '')}  ({r['p_attention']:.2f})" for r in order],
                       fontsize=6.5)
    ax.set_xticks([0, 9, 19, 29, 39, 49, 59])
    ax.set_xticklabels([1, 10, 20, 30, 40, 50, 60])
    ax.set_xlabel("Sampled frame (60 per video, evenly spaced)")
    ax.text(-0.155, 1 - n_normal / len(order) / 2, "Normal", transform=ax.transAxes, rotation=90,
            ha="right", va="center", fontweight="bold", fontsize=10)
    ax.text(-0.155, (len(order) - n_normal) / len(order) / 2, "Shoplifting", transform=ax.transAxes,
            rotation=90, ha="right", va="center", fontweight="bold", fontsize=10)
    ax.yaxis.set_label_coords(-0.27, 0.5)
    ax.set_title("(a) Attention weight of each frame for all 33 test videos "
                 "(sorted by Model B's probability, shown in brackets)", loc="left", fontsize=10.5,
                 fontweight="bold")
    bar = fig.colorbar(im, cax=cax)
    bar.set_label("Weight relative to an even spread\n(1 = every frame weighted equally)", fontsize=9)

    ax2 = fig.add_subplot(grid[1, 0])
    peaked = max(records, key=lambda r: r["weights"].max())
    typical = next(r for r in records if r["name"] == "shoplifting-27")
    frames = np.arange(1, n_frames + 1)
    ax2.axhline(1, color="#888888", lw=1, ls="--", label="Even spread (1/60 per frame)")
    ax2.plot(frames, typical["weights"] / uniform, color=MISSED, lw=1.8,
             label=f"{typical['name']}: typical video (top 5 frames = "
                   f"{np.sort(typical['weights'])[-5:].sum():.0%} of weight)")
    ax2.plot(frames, peaked["weights"] / uniform, color=FALSE_ALARM, lw=1.8,
             label=f"{peaked['name']}: most focused video (top 5 frames = "
                   f"{np.sort(peaked['weights'])[-5:].sum():.0%} of weight)")
    ax2.set_xlim(1, n_frames)
    ax2.set_ylim(0, max(2.0, peaked["weights"].max() / uniform * 1.1))
    ax2.set_xlabel("Sampled frame")
    ax2.set_ylabel("Relative weight")
    ax2.set_title("(b) Two examples", loc="left", fontsize=10.5, fontweight="bold")
    ax2.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax2.yaxis.grid(True, color="#E5E5E5")

    path = os.path.join(OUT_DIR, "fig_attention_weights.png")
    fig.savefig(path)
    plt.close(fig)
    summary = {
        "normalised_entropy_mean": float(entropy.mean()),
        "normalised_entropy_min": float(entropy.min()),
        "top5_share_median": float(np.median(top5)),
        "top5_share_max": float(top5.max()),
        "top5_share_if_uniform": 5 / n_frames,
        "max_single_weight": float(max(r["weights"].max() for r in records)),
        "most_focused_video": peaked["name"],
    }
    return path, summary


def main():
    test, cv = load_predictions()
    check_against_report(test, cv)
    print("Wrote", figure_error_types(test, cv))
    for path in figures_confusion_matrices():
        print("Wrote", path)
    print("Wrote", figure_feature_importance())
    path, summary = figure_attention(attention_weights())
    print("Wrote", path)
    for key, value in summary.items():
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")


if __name__ == "__main__":
    main()
