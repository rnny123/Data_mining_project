"""Shared code for the pose-based shoplifting detection notebooks.

Every notebook (and predict_clip) imports from here, so a new clip goes through
exactly the same preprocessing as the training clips did.

Pipeline:  video --YOLO11-pose--> keypoints per person per frame
                 --primary person + normalisation--> body-relative pose sequence
                 --> Approach A: clip-level features --> classic ML
                 --> Approach B: 3.2 s windows --> BiGRU --> aggregated clip probability
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

# ---------------------------------------------------------------- paths
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR.parent / "cleaned_unified_deduplicated"
CACHE_DIR = PROJECT_DIR / "cache" / "keypoints"
MODELS_DIR = PROJECT_DIR / "models"
FIGURES_DIR = PROJECT_DIR / "figures"
RESULTS_DIR = PROJECT_DIR / "results"

# ---------------------------------------------------------------- settings
POSE_WEIGHTS = "yolo11n-pose.pt"
TARGET_FPS = 10        # frames per second we sample from each video
MAX_PERSONS = 5        # people kept per frame
CONF_THRESH = 0.3      # keypoints below this confidence are treated as missing
WINDOW = 32            # sequence-model window length in sampled frames (3.2 s)
EVAL_STRIDE = 8        # step between windows at evaluation time
SEED = 42
NEAR_HIP = 0.5         # "hand near hip/pocket" threshold, in torso lengths
N_FEATURES = 85        # per-frame sequence input: 34 coords + 34 velocities + 17 confidences
LABEL_NAMES = {0: "normal", 1: "shoplifting"}

# COCO-17 keypoints, in the order YOLO-pose outputs them
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
K = {name: i for i, name in enumerate(KEYPOINT_NAMES)}
FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]  # left <-> right
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]


# ---------------------------------------------------------------- dataset
def cache_path_for(relative_path: str) -> Path:
    """videos/<label>/<name>.mp4  ->  cache/keypoints/<label>/<name>.npz"""
    parts = Path(relative_path).with_suffix(".npz").parts
    if parts[0] == "videos":
        parts = parts[1:]
    return CACHE_DIR.joinpath(*parts)


def load_manifest() -> pd.DataFrame:
    """manifest.csv plus `source` (staged / dcsass), absolute `video_path` and `cache_path`."""
    df = pd.read_csv(DATA_DIR / "manifest.csv")
    df["source"] = np.where(df["dataset"].str.startswith("dcsass"), "dcsass", "staged")
    df["video_path"] = [str(DATA_DIR / p) for p in df["relative_path"]]
    df["cache_path"] = [str(cache_path_for(p)) for p in df["relative_path"]]
    return df


# ---------------------------------------------------------------- primary person + normalisation
def select_primary_track(track_ids: np.ndarray, boxes: np.ndarray) -> int | None:
    """Track ID seen in the most frames (ties -> larger mean box). None if nobody was tracked."""
    valid = track_ids >= 0
    if not valid.any():
        return None
    ids, counts = np.unique(track_ids[valid], return_counts=True)
    candidates = ids[counts == counts.max()]
    if len(candidates) == 1:
        return int(candidates[0])
    areas = (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])
    mean_area = [areas[track_ids == c].mean() for c in candidates]
    return int(candidates[int(np.argmax(mean_area))])


def primary_sequence(ext: dict) -> tuple[np.ndarray, np.ndarray, bool]:
    """Keypoints (T,17,3) and box heights (T,) of the primary person; NaN where they are not visible."""
    kpts_all, boxes_all, ids = ext["kpts"], ext["boxes"], ext["track_ids"]
    T = kpts_all.shape[0]
    kpts = np.full((T, 17, 3), np.nan, np.float32)
    box_h = np.full(T, np.nan, np.float32)
    primary = select_primary_track(ids, boxes_all)
    if primary is not None:
        t_idx, p_idx = np.nonzero(ids == primary)
    else:
        # The tracker never confirmed anyone: use the most confident detection in each frame.
        t_idx = np.nonzero(boxes_all[:, 0, 2] > boxes_all[:, 0, 0])[0]
        p_idx = np.zeros_like(t_idx)
    kpts[t_idx] = kpts_all[t_idx, p_idx]
    box_h[t_idx] = boxes_all[t_idx, p_idx, 3] - boxes_all[t_idx, p_idx, 1]
    return kpts, box_h, bool(len(t_idx) > 0)


def fill_gaps(arr: np.ndarray) -> np.ndarray:
    """Fill NaNs along time (axis 0): linear inside, nearest value at the edges, 0 if never seen."""
    arr = np.asarray(arr, dtype=np.float32)
    flat = arr.reshape(arr.shape[0], -1).copy()
    t = np.arange(flat.shape[0])
    for j in range(flat.shape[1]):
        ok = ~np.isnan(flat[:, j])
        if ok.all():
            continue
        flat[:, j] = np.interp(t, t[ok], flat[ok, j]) if ok.any() else 0.0
    return flat.reshape(arr.shape)


def _body_centre(xy: np.ndarray, ever_seen: np.ndarray) -> np.ndarray:
    """Per-frame origin (T, 2) from keypoints that were actually seen: hips, else shoulders, else any."""
    for group in ((K["left_hip"], K["right_hip"]), (K["left_shoulder"], K["right_shoulder"])):
        idx = [i for i in group if ever_seen[i]]
        if idx:
            return xy[:, idx].mean(axis=1)
    idx = np.nonzero(ever_seen)[0]
    return xy[:, idx].mean(axis=1) if len(idx) else np.zeros((len(xy), 2), np.float32)


def normalize_pose(kpts: np.ndarray, box_h: np.ndarray) -> dict:
    """Pixel keypoints -> body-relative coordinates: mid-hip at the origin, 1 unit = torso length.

    Removes camera resolution, zoom and position, so a 320x240 CCTV clip and a
    640x480 staged clip of the same movement give the same numbers. Keypoints that
    are never confidently seen in the clip are set to 0 (the body centre) afterwards,
    so they carry no information about where the person stood in the image.
    """
    xy = kpts[..., :2].astype(np.float32).copy()
    conf = np.nan_to_num(kpts[..., 2], nan=0.0).astype(np.float32)
    xy[conf < CONF_THRESH] = np.nan
    seen = ~np.isnan(xy[..., 0])
    ever_seen = seen.any(axis=0)
    torso_seen = seen[:, [K["left_shoulder"], K["right_shoulder"], K["left_hip"], K["right_hip"]]].all(axis=1)
    xy = fill_gaps(xy)  # never-seen keypoints become pixel (0, 0) here; they are excluded and zeroed below

    centre = _body_centre(xy, ever_seen)
    mid_hip = (xy[:, K["left_hip"]] + xy[:, K["right_hip"]]) / 2
    mid_sh = (xy[:, K["left_shoulder"]] + xy[:, K["right_shoulder"]]) / 2
    torso = np.linalg.norm(mid_sh - mid_hip, axis=1)
    if torso_seen.any() and np.median(torso[torso_seen]) > 1e-3:
        scale = float(np.median(torso[torso_seen]))
    elif np.isfinite(box_h).any():
        scale = float(np.nanmedian(box_h)) * 0.3  # torso is roughly 30% of body height
    else:
        scale = 1.0
    scale = max(scale, 1e-3)

    seq = (xy - centre[:, None, :]) / scale
    vel = np.zeros_like(xy)
    vel[1:] = (xy[1:] - xy[:-1]) / scale  # keeps whole-body motion (walking away, etc.)
    seq[:, ~ever_seen] = 0.0
    vel[:, ~ever_seen] = 0.0
    return {"seq": seq.astype(np.float32), "vel": vel.astype(np.float32), "conf": conf, "scale": scale}


def clip_to_sequence(ext: dict) -> dict:
    """Extraction dict -> normalised pose sequence of the primary person."""
    kpts, box_h, has_person = primary_sequence(ext)
    out = normalize_pose(kpts, box_h)
    out["n_people"] = np.asarray(ext["n_people"], dtype=np.float32)
    out["has_person"] = has_person and bool((out["conf"] >= CONF_THRESH).any())  # a box alone is not a pose
    return out


# ---------------------------------------------------------------- pose extraction
def compute_stride(fps: float, target_fps: int = TARGET_FPS) -> int:
    """Keep every n-th frame so we sample ~target_fps. Broken/unknown fps -> assume 30 fps."""
    if not fps or not math.isfinite(fps) or fps <= 0:
        return max(1, round(30 / target_fps))
    return max(1, int(round(fps / target_fps)))


def load_pose_model(weights: str | Path = POSE_WEIGHTS):
    """YOLO11-pose model; the weights file is downloaded into models/ on first use."""
    from ultralytics import YOLO

    weights = Path(weights)
    if not weights.is_absolute():
        weights = MODELS_DIR / weights
    weights.parent.mkdir(parents=True, exist_ok=True)
    return YOLO(str(weights))


def extract_video(model, video_path, target_fps: int = TARGET_FPS, max_persons: int = MAX_PERSONS) -> dict:
    """Run YOLO-pose + ByteTrack over a clip at ~target_fps.

    Returns an extraction dict: kpts (T,P,17,3) [x, y, conf] in pixels, boxes (T,P,4) xyxy,
    track_ids (T,P) (-1 = empty/untracked), n_people (T,), width, height, fps_sampled, stride.
    Person slots in each frame are sorted by detection confidence.
    """
    import cv2

    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    opened, fps = cap.isOpened(), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not opened:
        raise ValueError(f"OpenCV cannot open video: {video_path}")
    stride = compute_stride(fps, target_fps)

    kpts, boxes, ids, n_people = [], [], [], []
    width = height = 0
    results = model.track(source=str(video_path), stream=True, persist=False, vid_stride=stride,
                          tracker="bytetrack.yaml", conf=0.25, imgsz=640, device="cpu", verbose=False)
    for r in results:
        height, width = r.orig_shape
        k = np.zeros((max_persons, 17, 3), np.float32)
        b = np.zeros((max_persons, 4), np.float32)
        t = np.full(max_persons, -1, np.int32)
        n = 0
        if r.boxes is not None and len(r.boxes) > 0 and r.keypoints is not None:
            n = len(r.boxes)
            order = np.argsort(-r.boxes.conf.cpu().numpy())[:max_persons]
            m = len(order)
            k[:m] = r.keypoints.data.cpu().numpy()[order]
            b[:m] = r.boxes.xyxy.cpu().numpy()[order]
            if r.boxes.id is not None:
                t[:m] = r.boxes.id.cpu().numpy().astype(np.int32)[order]
        kpts.append(k)
        boxes.append(b)
        ids.append(t)
        n_people.append(n)
    if not kpts:
        raise ValueError(f"No frames could be read from: {video_path}")
    fps_sampled = fps / stride if fps and math.isfinite(fps) and fps > 0 else float(target_fps)
    return {"kpts": np.stack(kpts), "boxes": np.stack(boxes), "track_ids": np.stack(ids),
            "n_people": np.asarray(n_people, np.int32), "width": int(width), "height": int(height),
            "fps_sampled": float(fps_sampled), "stride": int(stride)}


def save_extraction(ext: dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **ext)


def load_extraction(path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run 01_pose_extraction.ipynb first")
    with np.load(path) as z:
        ext = {k: z[k] for k in z.files}
    for key in ("width", "height", "stride"):
        ext[key] = int(ext[key])
    ext["fps_sampled"] = float(ext["fps_sampled"])
    return ext


def read_frame(video_path, frame_index: int) -> np.ndarray:
    """One frame of a video as an RGB array."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Cannot read frame {frame_index} of {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def draw_pose_frame(ax, video_path, ext: dict, t: int) -> None:
    """Draw sampled frame t with skeletons: primary person in red, others in grey."""
    stride = ext["stride"]
    ax.imshow(read_frame(video_path, t * stride + stride - 1))  # ultralytics keeps the last of each `stride` frames
    primary = select_primary_track(ext["track_ids"], ext["boxes"])
    for p in range(ext["kpts"].shape[1]):
        if ext["boxes"][t, p, 2] <= ext["boxes"][t, p, 0]:
            continue  # empty slot
        is_primary = ext["track_ids"][t, p] == primary if primary is not None else p == 0
        color = "red" if is_primary else "lightgrey"
        k = ext["kpts"][t, p]
        for a, b in SKELETON:
            if k[a, 2] >= CONF_THRESH and k[b, 2] >= CONF_THRESH:
                ax.plot([k[a, 0], k[b, 0]], [k[a, 1], k[b, 1]], color=color, lw=2)
        vis = k[:, 2] >= CONF_THRESH
        ax.scatter(k[vis, 0], k[vis, 1], s=8, color=color, zorder=3)
    ax.axis("off")


# ---------------------------------------------------------------- Approach A: hand-crafted features
STATS = ("mean", "std", "min", "max", "p10", "p90")


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle at b (degrees) between b->a and b->c; arrays (T, 2)."""
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def frame_features(s: dict) -> pd.DataFrame:
    """21 interpretable measurements per frame, in torso-length units / degrees."""
    seq, vel, conf = s["seq"], s["vel"], s["conf"]

    def P(name):
        return seq[:, K[name]]

    def dist(a, b):
        return np.linalg.norm(a - b, axis=1)

    mid_hip = (P("left_hip") + P("right_hip")) / 2
    mid_sh = (P("left_shoulder") + P("right_shoulder")) / 2
    mid_ank = (P("left_ankle") + P("right_ankle")) / 2
    lw, rw = P("left_wrist"), P("right_wrist")
    f = {
        "lwrist_lhip": dist(lw, P("left_hip")),          # hand at own-side pocket / waist
        "rwrist_rhip": dist(rw, P("right_hip")),
        "lwrist_midhip": dist(lw, mid_hip),
        "rwrist_midhip": dist(rw, mid_hip),
        "lwrist_nose": dist(lw, P("nose")),               # reaching up / hand near face
        "rwrist_nose": dist(rw, P("nose")),
        "wrist_wrist": dist(lw, rw),                      # both hands together (handling an item)
        "lwrist_height": mid_hip[:, 1] - lw[:, 1],        # image y points down: positive = above hip
        "rwrist_height": mid_hip[:, 1] - rw[:, 1],
        "lelbow_angle": _angle(P("left_shoulder"), P("left_elbow"), lw),
        "relbow_angle": _angle(P("right_shoulder"), P("right_elbow"), rw),
        "lknee_angle": _angle(P("left_hip"), P("left_knee"), P("left_ankle")),
        "rknee_angle": _angle(P("right_hip"), P("right_knee"), P("right_ankle")),
        "torso_lean": np.degrees(np.arctan2(np.abs(mid_sh[:, 0] - mid_hip[:, 0]), mid_hip[:, 1] - mid_sh[:, 1] + 1e-8)),
        "crouch": mid_ank[:, 1] - mid_hip[:, 1],          # small = crouching / bending
        "lwrist_speed": np.linalg.norm(vel[:, K["left_wrist"]], axis=1),
        "rwrist_speed": np.linalg.norm(vel[:, K["right_wrist"]], axis=1),
        "body_speed": np.linalg.norm((vel[:, K["left_hip"]] + vel[:, K["right_hip"]]) / 2, axis=1),
        "lwrist_visible": (conf[:, K["left_wrist"]] >= CONF_THRESH).astype(np.float32),  # hidden hands
        "rwrist_visible": (conf[:, K["right_wrist"]] >= CONF_THRESH).astype(np.float32),
        "n_people": s["n_people"],
    }
    return pd.DataFrame(f)


def clip_features(s: dict) -> pd.Series:
    """One row per clip: each frame feature summarised over time, plus hand-near-hip fraction."""
    ff = frame_features(s)
    a = ff.to_numpy(np.float64)
    stats = {"mean": a.mean(0), "std": a.std(0), "min": a.min(0), "max": a.max(0),
             "p10": np.percentile(a, 10, axis=0), "p90": np.percentile(a, 90, axis=0)}
    out = {f"{col}_{st}": float(stats[st][i]) for st in STATS for i, col in enumerate(ff.columns)}
    near = np.minimum(ff["lwrist_lhip"], ff["rwrist_rhip"]) < NEAR_HIP
    out["hand_near_hip_frac"] = float(near.mean())
    if not s["has_person"]:
        out = {k: 0.0 for k in out}
    return pd.Series(out)


# ---------------------------------------------------------------- Approach B: sequence model
AGGREGATIONS = ("mean", "max", "top3")


def sequence_input(s: dict) -> np.ndarray:
    """Per-frame model input (T, 85): 34 normalised coords | 34 velocities | 17 confidences."""
    T = s["seq"].shape[0]
    return np.concatenate([s["seq"].reshape(T, -1), s["vel"].reshape(T, -1), s["conf"]], axis=1).astype(np.float32)


def window_starts(T: int, window: int = WINDOW, stride: int = EVAL_STRIDE) -> list[int]:
    """Start frame of every evaluation window; the last window always ends at the clip's end."""
    if T <= window:
        return [0]
    starts = list(range(0, T - window + 1, stride))
    if starts[-1] != T - window:
        starts.append(T - window)
    return starts


def make_windows(x: np.ndarray, window: int = WINDOW, stride: int = EVAL_STRIDE):
    """Cut (T, D) into overlapping windows. Short clips -> one zero-padded window with a mask."""
    T, D = x.shape
    if T <= window:
        w = np.zeros((1, window, D), np.float32)
        m = np.zeros((1, window), np.float32)
        w[0, :T] = x
        m[0, :T] = 1
        return w, m
    starts = window_starts(T, window, stride)
    w = np.stack([x[s:s + window] for s in starts]).astype(np.float32)
    return w, np.ones((len(starts), window), np.float32)


def mirror(x: np.ndarray) -> np.ndarray:
    """Left/right mirror of sequence_input rows: swap left/right joints and negate x."""
    x = np.asarray(x)
    lead = x.shape[:-1]
    seq = x[..., :34].reshape(*lead, 17, 2)[..., FLIP_IDX, :].copy()
    vel = x[..., 34:68].reshape(*lead, 17, 2)[..., FLIP_IDX, :].copy()
    conf = x[..., 68:85][..., FLIP_IDX]
    seq[..., 0] *= -1
    vel[..., 0] *= -1
    return np.concatenate([seq.reshape(*lead, 34), vel.reshape(*lead, 34), conf], axis=-1).astype(x.dtype)


def augment(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Training-time augmentation of one window: random mirror, scale jitter, small noise."""
    if rng.random() < 0.5:
        w = mirror(w)
    w = np.array(w, dtype=np.float32, copy=True)
    w[:, :68] *= rng.uniform(0.9, 1.1)
    w[:, :68] += rng.normal(0.0, 0.02, size=w[:, :68].shape).astype(np.float32)
    return w


class PoseGRU(nn.Module):
    """Linear embed -> bidirectional GRU -> masked mean over time -> logit P(shoplifting)."""

    def __init__(self, n_features: int = N_FEATURES, hidden: int = 64, layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU())
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(1).clamp(min=1).long().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(self.embed(x), lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        pooled = (out * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.head(pooled).squeeze(-1)


@torch.no_grad()
def window_probs(model: PoseGRU, x: np.ndarray, window: int = WINDOW, stride: int = EVAL_STRIDE) -> np.ndarray:
    """P(shoplifting) for every window of one clip."""
    model.eval()
    w, m = make_windows(x, window, stride)
    return torch.sigmoid(model(torch.from_numpy(w), torch.from_numpy(m))).numpy()


def aggregate(probs, method: str = "mean") -> float:
    """Combine window probabilities into one clip probability."""
    p = np.asarray(probs, dtype=np.float64)
    if method == "mean":
        return float(p.mean())
    if method == "max":
        return float(p.max())
    if method == "top3":
        return float(np.sort(p)[-3:].mean())
    raise ValueError(f"Unknown aggregation '{method}', expected one of {AGGREGATIONS}")


# ---------------------------------------------------------------- prediction
def _require(path: Path, notebook: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found in {path.parent} - run {notebook} first")
    return path


class ShopliftingPredictor:
    """Video clip in -> shoplifting / normal out, using the saved Approach A or B model."""

    def __init__(self, approach: str = "best"):
        if approach == "best":
            best = _require(MODELS_DIR / "best_model.json", "04_evaluation.ipynb")
            approach = json.loads(best.read_text())["approach"]
        if approach not in ("A", "B"):
            raise ValueError(f"approach must be 'A', 'B' or 'best', got {approach!r}")
        self.approach = approach
        if approach == "A":
            import joblib

            self.model = joblib.load(_require(MODELS_DIR / "approach_a.joblib", "02_features_ml.ipynb"))
            self.meta = json.loads(_require(MODELS_DIR / "approach_a_meta.json", "02_features_ml.ipynb").read_text())
            self.description = f"Approach A: hand-crafted pose features + {self.meta['model']}"
        else:
            weights = _require(MODELS_DIR / "approach_b.pt", "03_sequence_model.ipynb")
            self.meta = json.loads(_require(MODELS_DIR / "approach_b_meta.json", "03_sequence_model.ipynb").read_text())
            self.model = PoseGRU(hidden=self.meta["hidden"], layers=self.meta["layers"], dropout=self.meta["dropout"])
            self.model.load_state_dict(torch.load(weights, map_location="cpu"))
            self.model.eval()
            self.description = f"Approach B: BiGRU on pose windows ({self.meta['aggregation']} of window probabilities)"
        self._pose_model = None

    @property
    def pose_model(self):
        if self._pose_model is None:
            self._pose_model = load_pose_model()
        return self._pose_model

    def predict_from_extraction(self, ext: dict) -> dict:
        s = clip_to_sequence(ext)
        if self.approach == "A":
            row = clip_features(s)[self.meta["feature_names"]].to_frame().T
            p = float(self.model.predict_proba(row)[0, 1])
        else:
            p = aggregate(window_probs(self.model, sequence_input(s), self.meta["window"], self.meta["eval_stride"]),
                          self.meta["aggregation"])
        return {"label": LABEL_NAMES[int(p >= 0.5)], "probability": p,
                "approach": self.approach, "no_person_detected": not s["has_person"]}

    def predict(self, video_path, keep_extraction: bool = False) -> dict:
        ext = extract_video(self.pose_model, video_path)
        out = {"video": str(video_path), **self.predict_from_extraction(ext)}
        if keep_extraction:
            out["extraction"] = ext
        return out


def predict_clip(video_path, approach: str = "best") -> dict:
    """One-call helper: predict_clip("clip.mp4") -> {"label": ..., "probability": ...}."""
    return ShopliftingPredictor(approach).predict(video_path)
