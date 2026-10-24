"""
Run the existing CNN+LSTM pipeline end-to-end on cleaned_unified_deduplicated.

Same method as scripts 01-05 (60 evenly-spaced frames -> 224x224 -> frozen
ImageNet ResNet18 -> 512-dim features -> 1-layer LSTM(128) -> sigmoid), but
reads videos and labels from the new dataset's manifest.csv and uses the
train/validation/test split already provided there (no re-splitting).

Everything is written under outputs/ next to this script, so the old
frames/, features/, splits.csv and lstm_best.pt in the parent folder are
left untouched.

After the main model is trained and evaluated, 5-fold stratified
cross-validation is run inside the training set only (validation and test
videos never enter the folds). The fold assignment is written to
outputs/cv_folds.csv so other models can use the same folds.

Usage:  python run_new_data.py
"""
import os
import csv
import copy
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # this experiment folder
# sets BASE_DIR to be new_data_experiment/
DATA_DIR = os.path.join(BASE_DIR, "cleaned_unified_deduplicated")
# sets DATA_DIR to be BASE_DIR/cleaned_unified_deduplicated

MANIFEST = os.path.join(DATA_DIR, "manifest.csv")
# tells us for each video, are they used for training, validation, or testing
# provide labels to the video (0 or 1)

OUT_DIR = os.path.join(BASE_DIR, "outputs")

FEATURES_DIR = os.path.join(OUT_DIR, "features")
CHECKPOINT_PATH = os.path.join(OUT_DIR, "lstm_best.pt")
CV_FOLDS_PATH = os.path.join(OUT_DIR, "cv_folds.csv")

N_FRAMES = 60
OUT_SIZE = (224, 224)

HIDDEN_SIZE = 128
NUM_LAYERS = 1
DROPOUT = 0.3
LR = 1e-3
EPOCHS = 30
BATCH_SIZE = 8
SEED = 42
N_FOLDS = 5

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_manifest():
    with open(MANIFEST, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["name"] = os.path.splitext(os.path.basename(r["relative_path"]))[0]
        r["label"] = int(r["label"])
        r["split"] = {"validation": "val"}.get(r["split"], r["split"])
    return rows


# ---------- step 1+2: frames -> ResNet18 features ----------
def sample_frames(video_path):
    """60 evenly spaced frames, resized to 224x224 (same logic as script 01).
    Frames take a JPEG round trip so features match the old on-disk pipeline."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    wanted = sorted(set(int(i * (total - 1) / (N_FRAMES - 1)) for i in range(N_FRAMES)))
    frames, pos, frame_no = [], 0, 0
    while pos < len(wanted):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_no == wanted[pos]:
            small = cv2.resize(frame, OUT_SIZE, interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", small)
            small = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            pos += 1
        frame_no += 1
    cap.release()
    return frames


def extract_all_features(rows):
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    resnet.fc = nn.Identity()
    resnet.eval()
    for p in resnet.parameters():
        p.requires_grad = False

    os.makedirs(FEATURES_DIR, exist_ok=True)
    for i, r in enumerate(rows, 1):
        out = os.path.join(FEATURES_DIR, f"{r['name']}.npy")
        if os.path.exists(out):
            continue  # resume support
        frames = sample_frames(os.path.join(DATA_DIR, r["relative_path"]))
        if len(frames) != N_FRAMES:
            print(f"  WARNING: {r['name']} only got {len(frames)}/{N_FRAMES} frames")
            if not frames:
                continue
        batch = torch.stack([preprocess(f) for f in frames])
        with torch.no_grad():
            feats = resnet(batch).numpy()
        np.save(out, feats)
        if i % 10 == 0 or i == len(rows):
            print(f"  features {i}/{len(rows)}", flush=True)


# ---------- step 4: LSTM ----------
class FeatureDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        feats = np.load(os.path.join(FEATURES_DIR, f"{r['name']}.npy"))
        return torch.from_numpy(feats).float(), torch.tensor(r["label"], dtype=torch.float32)


class LSTMClassifier(nn.Module):
    def __init__(self, input_size=512):
        super().__init__()
        self.lstm = nn.LSTM(input_size, HIDDEN_SIZE, NUM_LAYERS, batch_first=True)
        self.dropout = nn.Dropout(DROPOUT)
        self.classifier = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.classifier(self.dropout(h_n[-1])).squeeze(-1)


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            if is_train:
                optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += ((torch.sigmoid(logits) >= 0.5).float() == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total


def fit(train_rows, val_rows, verbose=True):
    """Train for EPOCHS and return the weights with the best validation accuracy."""
    torch.manual_seed(SEED)
    train_loader = DataLoader(FeatureDataset(train_rows), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(FeatureDataset(val_rows), batch_size=BATCH_SIZE)
    model = LSTMClassifier()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best, best_state, best_epoch = -1.0, None, 0
    for epoch in range(1, EPOCHS + 1):
        tl, ta = run_epoch(model, train_loader, criterion, optimizer)
        vl, va = run_epoch(model, val_loader, criterion)
        mark = ""
        if va > best:
            best, best_epoch = va, epoch
            best_state = copy.deepcopy(model.state_dict())
            mark = "  <- saved (best val acc)"
        if verbose:
            print(f"epoch {epoch:2d}/{EPOCHS}  train_loss={tl:.4f} train_acc={ta:.3f}  "
                  f"val_loss={vl:.4f} val_acc={va:.3f}{mark}", flush=True)
    return best_state, best, best_epoch


def train(splits):
    state, best, _ = fit(splits["train"], splits["val"])
    torch.save(state, CHECKPOINT_PATH)
    print(f"\nBest val accuracy: {best:.3f}")


# ---------- step 5: evaluation ----------
def predict(state, rows):
    """Return (labels, shoplifting probabilities) for rows under the given weights."""
    model = LSTMClassifier()
    model.load_state_dict(state)
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for r in rows:
            feats = np.load(os.path.join(FEATURES_DIR, f"{r['name']}.npy"))
            y_prob.append(torch.sigmoid(model(torch.from_numpy(feats).float().unsqueeze(0))).item())
            y_true.append(r["label"])
    return y_true, y_prob


def evaluate(rows, title):
    y_true, y_prob = predict(torch.load(CHECKPOINT_PATH, weights_only=True), rows)
    y_pred = [int(p >= 0.5) for p in y_prob]

    print(f"\n=== {title} ({len(rows)} videos) ===")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision (shoplifting): {precision_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"Recall    (shoplifting): {recall_score(y_true, y_pred):.3f}")
    print(f"F1        (shoplifting): {f1_score(y_true, y_pred):.3f}")
    print(f"AUC-ROC:   {roc_auc_score(y_true, y_prob):.3f}")
    print("Confusion matrix (rows=actual, cols=pred, [normal, shoplifting]):")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
    print(classification_report(y_true, y_pred, target_names=["normal", "shoplifting"], zero_division=0))
    print("Misclassified:")
    for r, t, p, pr in zip(rows, y_true, y_pred, y_prob):
        if t != p:
            print(f"  {r['dataset']:20s} {r['name']:28s} true={t} pred={p} p={pr:.2f}")

    # per-source breakdown: staged clips vs. the DCSASS ones
    print("\nAccuracy by source dataset:")
    for ds in sorted(set(r["dataset"] for r in rows)):
        idx = [i for i, r in enumerate(rows) if r["dataset"] == ds]
        acc = np.mean([y_true[i] == y_pred[i] for i in idx])
        print(f"  {ds:22s} n={len(idx):3d}  acc={acc:.3f}")


# ---------- step 6: 5-fold cross-validation inside the training set ----------
def cross_validate(splits):
    """Each fold trains on 4/5 of the training videos, picks its checkpoint on the
    fixed validation set (same rule as the main model) and is scored on the
    held-out fifth. Validation and test videos never enter the folds."""
    rows = splits["train"]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "auc": []}
    assignment = []

    print(f"\n=== {N_FOLDS}-fold cross-validation on the training set ({len(rows)} videos) ===")
    for k, (tr_idx, ho_idx) in enumerate(skf.split(rows, [r["label"] for r in rows]), 1):
        held_out = [rows[i] for i in ho_idx]
        state, _, best_epoch = fit([rows[i] for i in tr_idx], splits["val"], verbose=False)
        y_true, y_prob = predict(state, held_out)
        y_pred = [int(p >= 0.5) for p in y_prob]
        fold = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "auc": roc_auc_score(y_true, y_prob),
        }
        for m, v in fold.items():
            metrics[m].append(v)
        assignment += [(r["name"], r["label"], k) for r in held_out]
        print(f"fold {k}: n={len(held_out)} best_epoch={best_epoch:2d}  "
              + "  ".join(f"{m}={v:.3f}" for m, v in fold.items()), flush=True)

    print("mean ± std across folds:")
    for m, vals in metrics.items():
        print(f"  {m:9s} {np.mean(vals):.3f} ± {np.std(vals):.3f}")

    with open(CV_FOLDS_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "label", "fold"])
        writer.writerows(sorted(assignment))
    print(f"Fold assignment written to {CV_FOLDS_PATH}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load_manifest()
    splits = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    print({s: len(v) for s, v in splits.items()})
    extract_all_features(rows)
    train(splits)
    evaluate(splits["val"], "Validation set (used for checkpoint selection)")
    evaluate(splits["test"], "TEST set")
    cross_validate(splits)


if __name__ == "__main__":
    main()
