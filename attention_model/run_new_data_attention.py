"""
Train and evaluate the CNN-LSTM with temporal attention on the unified video
dataset. This keeps the baseline's frame sampling, frozen ResNet18 features,
fixed data splits, optimizer, and metrics, while replacing the final-hidden-
state classifier with attention over every LSTM output.

Usage: python run_new_data_attention.py
Outputs are isolated in outputs_attention/ beside this script.

Changes from the original script, so all models run on one computer under one protocol:
  * the dataset is also looked for in ../Baseline Model/cleaned_unified_deduplicated
  * cross-validation keeps, in each fold, the epoch with the best validation accuracy (same rule
    and seed as the final model and as Model A's cross-validation) instead of the last epoch
"""
import copy
import csv
import hashlib
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR_CANDIDATES = (
    os.path.join(BASE_DIR, "cleaned_unified_deduplicated"),
    os.path.join(BASE_DIR, "data cleaning", "cleaned_data", "unified_video_deduplicated"),
    os.path.join(
        os.path.dirname(BASE_DIR),
        "data cleaning",
        "cleaned_data",
        "unified_video_deduplicated",
    ),
    os.path.join(
        os.path.dirname(BASE_DIR),
        "new_data_experiment",
        "new_data_experiment",
        "cleaned_unified_deduplicated",
    ),
    os.path.join(os.path.dirname(BASE_DIR), "Baseline Model", "cleaned_unified_deduplicated"),
)
DATA_DIR = next(
    (
        path
        for path in DATA_DIR_CANDIDATES
        if os.path.isfile(os.path.join(path, "manifest.csv"))
        and os.path.isdir(os.path.join(path, "videos"))
    ),
    None,
)
if DATA_DIR is None:
    candidates = "\n".join(f"  - {path}" for path in DATA_DIR_CANDIDATES)
    raise FileNotFoundError(
        "Could not find a dataset with manifest.csv and a videos/ directory. "
        "Checked:\n"
        f"{candidates}"
    )

MANIFEST = os.path.join(DATA_DIR, "manifest.csv")
OUT_DIR = os.path.join(BASE_DIR, "outputs_attention")
FEATURES_DIR = os.path.join(OUT_DIR, "features")
CHECKPOINT_PATH = os.path.join(OUT_DIR, "cnn_lstm_attention_best.pt")

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

preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def load_manifest():
    with open(MANIFEST, newline="") as manifest_file:
        rows = list(csv.DictReader(manifest_file))
    for row in rows:
        row["name"] = os.path.splitext(os.path.basename(row["relative_path"]))[0]
        row["label"] = int(row["label"])
        row["split"] = {"validation": "val"}.get(row["split"], row["split"])
    return rows


def feature_path(row):
    cache_key = hashlib.sha256(row["relative_path"].encode("utf-8")).hexdigest()
    return os.path.join(FEATURES_DIR, f"{cache_key}.npy")


def sample_frames(video_path):
    """Sample 60 evenly spaced frames and match the baseline preprocessing."""
    capture = cv2.VideoCapture(video_path)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        capture.release()
        return []

    wanted = sorted(
        set(int(i * (total - 1) / (N_FRAMES - 1)) for i in range(N_FRAMES))
    )
    frames, position, frame_number = [], 0, 0
    while position < len(wanted):
        success, frame = capture.read()
        if not success:
            break
        if frame_number == wanted[position]:
            small = cv2.resize(frame, OUT_SIZE, interpolation=cv2.INTER_AREA)
            success, buffer = cv2.imencode(".jpg", small)
            if not success:
                capture.release()
                raise RuntimeError(f"Could not encode sampled frame from {video_path}")
            small = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            position += 1
        frame_number += 1
    capture.release()
    return frames


def extract_all_features(rows):
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    resnet.fc = nn.Identity()
    resnet.eval()
    for parameter in resnet.parameters():
        parameter.requires_grad = False

    os.makedirs(FEATURES_DIR, exist_ok=True)
    for index, row in enumerate(rows, 1):
        output_path = feature_path(row)
        if os.path.exists(output_path):
            continue

        video_path = os.path.join(DATA_DIR, row["relative_path"])
        frames = sample_frames(video_path)
        if len(frames) != N_FRAMES:
            raise RuntimeError(
                f"{row['name']} produced {len(frames)}/{N_FRAMES} frames; "
                "check the video file and decoder."
            )
        batch = torch.stack([preprocess(frame) for frame in frames])
        with torch.no_grad():
            features = resnet(batch).cpu().numpy()
        np.save(output_path, features)
        if index % 10 == 0 or index == len(rows):
            print(f"  features {index}/{len(rows)}", flush=True)


class FeatureDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        features = np.load(feature_path(row))
        return (
            torch.from_numpy(features).float(),
            torch.tensor(row["label"], dtype=torch.float32),
        )


class AttentionLSTMClassifier(nn.Module):
    def __init__(self, input_size=512):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            HIDDEN_SIZE,
            NUM_LAYERS,
            batch_first=True,
        )
        self.attention_projection = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.attention_score = nn.Linear(HIDDEN_SIZE, 1, bias=False)
        self.dropout = nn.Dropout(DROPOUT)
        self.classifier = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, features):
        sequence_outputs, _ = self.lstm(features)
        scores = self.attention_score(
            torch.tanh(self.attention_projection(sequence_outputs))
        ).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(sequence_outputs * weights.unsqueeze(-1), dim=1)
        return self.classifier(self.dropout(context)).squeeze(-1)


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for features, labels in loader:
            if is_train:
                optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * features.size(0)
            correct += (
                ((torch.sigmoid(logits) >= 0.5).float() == labels).sum().item()
            )
            total += features.size(0)
    return total_loss / total, correct / total


def predict_probabilities(model, rows):
    model.eval()
    probabilities = []
    with torch.no_grad():
        for row in rows:
            features = np.load(feature_path(row))
            probability = torch.sigmoid(
                model(torch.from_numpy(features).float().unsqueeze(0))
            ).item()
            probabilities.append(probability)
    return probabilities


def fit(train_rows, val_rows, verbose=True):
    """Train for EPOCHS and return the weights with the best validation accuracy."""
    torch.manual_seed(SEED)
    train_loader = DataLoader(
        FeatureDataset(train_rows),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        FeatureDataset(val_rows),
        batch_size=BATCH_SIZE,
    )
    model = AttentionLSTMClassifier()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_validation_accuracy, best_state, best_epoch = -1.0, None, 0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, optimizer
        )
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion)
        marker = ""
        if val_accuracy > best_validation_accuracy:
            best_validation_accuracy, best_epoch = val_accuracy, epoch
            best_state = copy.deepcopy(model.state_dict())
            marker = "  <- saved (best val acc)"
        if verbose:
            print(
                f"epoch {epoch:2d}/{EPOCHS}  "
                f"train_loss={train_loss:.4f} train_acc={train_accuracy:.3f}  "
                f"val_loss={val_loss:.4f} val_acc={val_accuracy:.3f}{marker}",
                flush=True,
            )
    return best_state, best_validation_accuracy, best_epoch


def cross_validate(training_rows, validation_rows):
    labels = np.array([row["label"] for row in training_rows])
    splitter = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED,
    )
    fold_metrics = []

    print(f"\n=== {N_FOLDS}-fold cross-validation on training split ===")
    for fold, (train_indices, held_out_indices) in enumerate(
        splitter.split(np.zeros(len(training_rows)), labels),
        start=1,
    ):
        fold_train_rows = [training_rows[index] for index in train_indices]
        fold_held_out_rows = [training_rows[index] for index in held_out_indices]

        state, _, best_epoch = fit(fold_train_rows, validation_rows, verbose=False)
        model = AttentionLSTMClassifier()
        model.load_state_dict(state)

        y_true = [row["label"] for row in fold_held_out_rows]
        y_prob = predict_probabilities(model, fold_held_out_rows)
        y_pred = [int(probability >= 0.5) for probability in y_prob]
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "auc": roc_auc_score(y_true, y_prob),
        }
        fold_metrics.append(metrics)
        print(
            f"Fold {fold} held-out ({len(fold_held_out_rows)} videos, best epoch {best_epoch}): "
            f"accuracy={metrics['accuracy']:.3f} "
            f"precision={metrics['precision']:.3f} "
            f"recall={metrics['recall']:.3f} "
            f"f1={metrics['f1']:.3f} "
            f"auc={metrics['auc']:.3f}",
            flush=True,
        )

    print("\nCross-validation mean +/- standard deviation:")
    for metric_name in ("accuracy", "precision", "recall", "f1", "auc"):
        values = [metrics[metric_name] for metrics in fold_metrics]
        print(f"  {metric_name}: {np.mean(values):.3f} +/- {np.std(values):.3f}")


def train(splits):
    state, best_validation_accuracy, _ = fit(splits["train"], splits["val"])
    torch.save(state, CHECKPOINT_PATH)
    print(f"\nBest val accuracy: {best_validation_accuracy:.3f}")


def evaluate(rows, title):
    model = AttentionLSTMClassifier()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for row in rows:
            features = np.load(feature_path(row))
            probability = torch.sigmoid(
                model(torch.from_numpy(features).float().unsqueeze(0))
            ).item()
            y_true.append(row["label"])
            y_pred.append(int(probability >= 0.5))
            y_prob.append(probability)

    print(f"\n=== {title} ({len(rows)} videos) ===")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.3f}")
    print(
        "Precision (shoplifting): "
        f"{precision_score(y_true, y_pred, zero_division=0):.3f}"
    )
    print(f"Recall    (shoplifting): {recall_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"F1        (shoplifting): {f1_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"AUC-ROC:   {roc_auc_score(y_true, y_prob):.3f}")
    print("Confusion matrix (rows=actual, cols=pred, [normal, shoplifting]):")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["normal", "shoplifting"],
            zero_division=0,
        )
    )
    print("Misclassified:")
    for row, true_label, predicted_label, probability in zip(
        rows, y_true, y_pred, y_prob
    ):
        if true_label != predicted_label:
            print(
                f"  {row['dataset']:20s} {row['name']:28s} "
                f"true={true_label} pred={predicted_label} p={probability:.2f}"
            )

    print("\nAccuracy by source dataset:")
    for dataset in sorted(set(row["dataset"] for row in rows)):
        indices = [
            index
            for index, row in enumerate(rows)
            if row["dataset"] == dataset
        ]
        source_accuracy = np.mean(
            [y_true[index] == y_pred[index] for index in indices]
        )
        print(f"  {dataset:22s} n={len(indices):3d}  acc={source_accuracy:.3f}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load_manifest()
    splits = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "val", "test")
    }
    print(f"Dataset: {DATA_DIR}")
    print({split: len(split_rows) for split, split_rows in splits.items()})
    extract_all_features(rows)
    cross_validate(splits["train"], splits["val"])
    train(splits)
    evaluate(splits["val"], "Validation set (used for checkpoint selection)")
    evaluate(splits["test"], "TEST set")


if __name__ == "__main__":
    main()
