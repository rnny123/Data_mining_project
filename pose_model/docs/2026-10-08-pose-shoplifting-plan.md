# Pose-Based Shoplifting Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Five Jupyter notebooks plus a shared `pose_utils.py` that turn a video clip into YOLO11-pose keypoints and classify it as shoplifting / normal with two approaches (hand-crafted features + classic ML, and a BiGRU sequence model).

**Architecture:** `pose_utils.py` holds every reusable step (extraction → primary person → normalisation → features / windows → models → predictor) and is unit-tested with pytest. Notebooks are thin, narrated drivers that call it, so training, evaluation and `predict_clip` share one code path. Notebooks are authored as percent-format `.py` files in the scratchpad and converted to `.ipynb` with a small nbformat script, then executed with `nbconvert` so outputs are embedded.

**Tech Stack:** Python 3.12 venv, ultralytics (YOLO11n-pose + ByteTrack), OpenCV, PyTorch (CPU), scikit-learn, pandas, matplotlib, joblib, pytest, nbconvert.

**Spec:** `Project/pose_model/docs/2026-10-08-pose-shoplifting-design.md`

## Global Constraints

- Project root: `C:\Users\javier\Desktop\IS424 DATA MINING\Project\pose_model\` (called `PM` below). Dataset: `PM\..\cleaned_unified_deduplicated\`.
- Python: always `PM\.venv\Scripts\python.exe` (called `VPY`). Never pip-install into the global Python (it would upgrade global torch 2.10 → 2.14 and numpy 1.26 → 2.5).
- Jupyter kernel name `pose-shoplifting`, display name `Python (pose-shoplifting)`.
- CPU only. `device="cpu"` for YOLO.
- Fixed splits from `manifest.csv` (train 144 / validation 30 / test 33). Test split is touched only in `04_evaluation.ipynb` (and the smoke test of `predict_clip`).
- Constants: `TARGET_FPS = 10`, `MAX_PERSONS = 5`, `CONF_THRESH = 0.3`, `WINDOW = 32`, `EVAL_STRIDE = 8`, `SEED = 42`, `NEAR_HIP = 0.5`, `N_FEATURES = 85`.
- Labels: 1 = shoplifting, 0 = normal. `probability` always means P(shoplifting). Decision threshold 0.5.
- Clip length/duration is never a model feature.
- Not a git repository: there are no commit steps. Each task ends with its tests passing instead.
- Scratchpad (`SCRATCH`): `C:\Users\javier\AppData\Local\Temp\claude\c--Users-javier-Desktop-IS424-DATA-MINING\b06b1647-547d-452a-9b06-b5f1e3a6ec7a\scratchpad`. Notebook sources and the builder live there, not in the project.

## Review Focus

1. **Clip where nobody is detected** (dark/empty/occluded): must still return a prediction with `no_person_detected=True`, with no NaN or crash anywhere. Pinned in Task 2 (`test_clip_to_sequence_empty_clip_is_finite`) and Task 7 (`test_predictor_no_person_clip`).
2. **Clip shorter than one 3.2 s window**: a single zero-padded, masked window, and padding must not change the GRU output. Pinned in Task 6 (`test_make_windows_short_clip_padded`, `test_posegru_padding_invariance`).
3. **Tracker never assigns IDs** (`r.boxes.id is None` on every frame): fall back to the most confident detection per frame instead of reporting "no person". Pinned in Task 2 (`test_primary_sequence_falls_back_without_track_ids`).
4. **User clip at a different frame rate** (25 / 60 fps, or a broken fps header): stride adapts to keep ~10 fps, with a safe default. Pinned in Task 3 (`test_compute_stride`).
5. **Bad input path / unreadable file / models not trained yet**: clear `FileNotFoundError` / `ValueError` naming the fix, instead of a deep ultralytics/joblib traceback. Pinned in Task 3 (`test_extract_video_missing_file_raises`, `test_extract_video_unreadable_file_raises`) and Task 7 (`test_predictor_missing_models_message`).

---

### Task 1: Environment, scaffolding, manifest loader

**Files:**
- Create: `PM\requirements.txt`, `PM\pose_utils.py`, `PM\tests\conftest.py`, `PM\tests\test_pose_utils.py`
- Create: `SCRATCH\build_nb.py`

**Interfaces:**
- Produces: `pose_utils` constants (`PROJECT_DIR, DATA_DIR, CACHE_DIR, MODELS_DIR, FIGURES_DIR, RESULTS_DIR, POSE_WEIGHTS, TARGET_FPS, MAX_PERSONS, CONF_THRESH, WINDOW, EVAL_STRIDE, SEED, NEAR_HIP, N_FEATURES, LABEL_NAMES, KEYPOINT_NAMES, K, FLIP_IDX, SKELETON`), `load_manifest() -> pd.DataFrame` (adds `source`, `video_path`, `cache_path`), `cache_path_for(relative_path: str) -> Path`. Test fixtures `make_ext`, `standing_pose`.

- [ ] **Step 1: Create the venv and install packages**

```powershell
$PM = "C:\Users\javier\Desktop\IS424 DATA MINING\Project\pose_model"
python -m venv "$PM\.venv"
& "$PM\.venv\Scripts\python.exe" -m pip install --upgrade pip
& "$PM\.venv\Scripts\python.exe" -m pip install torch torchvision ultralytics lap opencv-python numpy pandas scikit-learn matplotlib joblib ipykernel nbconvert nbformat pytest
& "$PM\.venv\Scripts\python.exe" -m ipykernel install --user --name pose-shoplifting --display-name "Python (pose-shoplifting)"
```
(`lap` is pre-installed because ByteTrack otherwise tries to pip-install it at runtime.)

Expected: `Installed kernelspec pose-shoplifting in ...`.

- [ ] **Step 2: Write `requirements.txt` pinned to what was installed**

```powershell
& "$PM\.venv\Scripts\python.exe" -m pip freeze | Select-String -Pattern '^(torch|torchvision|ultralytics|lap|opencv-python|numpy|pandas|scikit-learn|matplotlib|joblib|ipykernel|nbconvert|nbformat|pytest)==' | ForEach-Object { $_.Line } | Set-Content -Encoding utf8 "$PM\requirements.txt"
```

- [ ] **Step 3: Write the failing tests** — `PM\tests\conftest.py`

```python
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pose_utils as pu  # noqa: E402


def _standing_pose(cx=300.0, cy=200.0, torso=100.0):
    """Synthetic upright person: mid-hip at (cx, cy), torso length `torso` px, every keypoint conf 0.9."""
    pts = {
        "nose": (0, -1.5), "left_eye": (-0.1, -1.6), "right_eye": (0.1, -1.6),
        "left_ear": (-0.2, -1.55), "right_ear": (0.2, -1.55),
        "left_shoulder": (-0.4, -1.0), "right_shoulder": (0.4, -1.0),
        "left_elbow": (-0.5, -0.5), "right_elbow": (0.5, -0.5),
        "left_wrist": (-0.5, 0.0), "right_wrist": (0.5, 0.0),
        "left_hip": (-0.25, 0.0), "right_hip": (0.25, 0.0),
        "left_knee": (-0.25, 1.0), "right_knee": (0.25, 1.0),
        "left_ankle": (-0.25, 2.0), "right_ankle": (0.25, 2.0),
    }
    k = np.zeros((17, 3), np.float32)
    for name, (x, y) in pts.items():
        k[pu.K[name]] = (cx + x * torso, cy + y * torso, 0.9)
    return k


def _make_ext(T=20, P=pu.MAX_PERSONS):
    """An extraction dict in which nobody was detected."""
    return {
        "kpts": np.zeros((T, P, 17, 3), np.float32),
        "boxes": np.zeros((T, P, 4), np.float32),
        "track_ids": np.full((T, P), -1, np.int32),
        "n_people": np.zeros(T, np.int32),
        "width": 640, "height": 480, "fps_sampled": 10.0, "stride": 3,
    }


@pytest.fixture
def standing_pose():
    return _standing_pose


@pytest.fixture
def make_ext():
    return _make_ext
```

`PM\tests\test_pose_utils.py`

```python
import numpy as np
import pytest
import torch

import pose_utils as pu


def test_cache_path_for_drops_videos_prefix():
    assert pu.cache_path_for("videos/normal/normal-1.mp4") == pu.CACHE_DIR / "normal" / "normal-1.npz"


@pytest.mark.skipif(not (pu.DATA_DIR / "manifest.csv").exists(), reason="dataset not available")
def test_load_manifest_adds_columns():
    m = pu.load_manifest()
    assert len(m) == 207
    assert set(m["source"]) == {"staged", "dcsass"}
    assert m["cache_path"].is_unique
    assert set(m["split"]) == {"train", "validation", "test"}
```

- [ ] **Step 4: Run to verify failure**

Run: `& "$PM\.venv\Scripts\python.exe" -m pytest "$PM\tests" -q`
Expected: FAIL / error `ModuleNotFoundError: No module named 'pose_utils'`.

- [ ] **Step 5: Implement `PM\pose_utils.py` (header section)**

```python
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
```

- [ ] **Step 6: Run tests to verify pass**

Run: `& "$PM\.venv\Scripts\python.exe" -m pytest "$PM\tests" -q`
Expected: `2 passed`.

- [ ] **Step 7: Write the notebook builder** — `SCRATCH\build_nb.py`

```python
"""Convert percent-format .py sources (# %% / # %% [markdown]) into .ipynb files.

Usage: python build_nb.py <src.py> <dst.ipynb>
"""
import re
import sys
from pathlib import Path

import nbformat


def convert(src: Path, dst: Path) -> None:
    text = Path(src).read_text(encoding="utf-8")
    chunks = re.split(r"^# %%(.*)$", text, flags=re.M)
    cells = []
    for header, body in zip(chunks[1::2], chunks[2::2]):
        body = body.strip("\n")
        if "[markdown]" in header:
            lines = []
            for line in body.splitlines():
                if line == "#":
                    lines.append("")
                elif line.startswith("# "):
                    lines.append(line[2:])
                else:
                    lines.append(line)
            cells.append(nbformat.v4.new_markdown_cell("\n".join(lines)))
        else:
            cells.append(nbformat.v4.new_code_cell(body))
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "pose-shoplifting", "display_name": "Python (pose-shoplifting)", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    nbformat.write(nb, str(dst))


if __name__ == "__main__":
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
```

---

### Task 2: Primary person, gap filling, normalisation

**Files:**
- Modify: `PM\pose_utils.py` (append section)
- Test: `PM\tests\test_pose_utils.py` (append)

**Interfaces:**
- Consumes: constants from Task 1.
- Produces:
  - `select_primary_track(track_ids: np.ndarray (T,P), boxes: np.ndarray (T,P,4)) -> int | None`
  - `primary_sequence(ext: dict) -> tuple[np.ndarray (T,17,3) NaN where missing, np.ndarray (T,) box heights, bool has_person]`
  - `fill_gaps(arr: np.ndarray) -> np.ndarray` (fills along axis 0)
  - `normalize_pose(kpts (T,17,3), box_h (T,)) -> {"seq": (T,17,2), "vel": (T,17,2), "conf": (T,17), "scale": float}`
  - `clip_to_sequence(ext: dict) -> {"seq", "vel", "conf", "scale", "n_people": (T,) float32, "has_person": bool}`
  - An extraction dict `ext` has keys `kpts (T,P,17,3)`, `boxes (T,P,4)`, `track_ids (T,P) int32`, `n_people (T,)`, `width`, `height`, `fps_sampled`, `stride`. Empty person slots have zero boxes and track id -1. Slots are sorted by detection confidence (slot 0 = most confident).

- [ ] **Step 1: Write the failing tests** (append to `test_pose_utils.py`)

```python
def test_select_primary_track_most_frames():
    ids = np.array([[1, 2], [1, -1], [1, 2], [3, -1]])
    assert pu.select_primary_track(ids, np.ones((4, 2, 4))) == 1


def test_select_primary_track_tie_prefers_bigger_box():
    ids = np.array([[1, 2], [1, 2]])
    boxes = np.zeros((2, 2, 4))
    boxes[:, 0] = [0, 0, 10, 10]
    boxes[:, 1] = [0, 0, 50, 50]
    assert pu.select_primary_track(ids, boxes) == 2


def test_select_primary_track_none():
    assert pu.select_primary_track(np.full((3, 2), -1), np.zeros((3, 2, 4))) is None


def test_fill_gaps():
    np.testing.assert_allclose(pu.fill_gaps(np.array([np.nan, 1.0, np.nan, 3.0, np.nan])), [1, 1, 2, 3, 3])
    b = np.full((4, 2), np.nan)
    b[:, 1] = [0, np.nan, np.nan, 3]
    np.testing.assert_allclose(pu.fill_gaps(b), [[0, 0], [0, 1], [0, 2], [0, 3]])


def test_primary_sequence_picks_tracked_person(make_ext, standing_pose):
    ext = make_ext(T=4)
    for t in (0, 2, 3):  # person 7 missing in frame 1
        ext["kpts"][t, 0] = standing_pose(cx=100)
        ext["boxes"][t, 0] = [50, 40, 150, 400]
        ext["track_ids"][t, 0] = 7
    ext["kpts"][0, 1] = standing_pose(cx=500)
    ext["boxes"][0, 1] = [450, 40, 550, 400]
    ext["track_ids"][0, 1] = 9
    kpts, box_h, has_person = pu.primary_sequence(ext)
    assert has_person
    assert kpts[0, pu.K["nose"], 0] == pytest.approx(100)
    assert np.isnan(kpts[1]).all() and np.isnan(box_h[1])


def test_primary_sequence_falls_back_without_track_ids(make_ext, standing_pose):
    ext = make_ext(T=4)
    for t in range(4):
        ext["kpts"][t, 0] = standing_pose()
        ext["boxes"][t, 0] = [250, 40, 350, 400]
    kpts, box_h, has_person = pu.primary_sequence(ext)
    assert has_person and np.isfinite(kpts).all()
    np.testing.assert_allclose(box_h, 360)


def test_normalize_pose_centres_and_scales(standing_pose):
    kpts = np.stack([standing_pose(cx=300 + 10 * t, torso=100) for t in range(10)])
    out = pu.normalize_pose(kpts, np.full(10, 360.0))
    seq = out["seq"]
    mid_hip = (seq[:, pu.K["left_hip"]] + seq[:, pu.K["right_hip"]]) / 2
    mid_sh = (seq[:, pu.K["left_shoulder"]] + seq[:, pu.K["right_shoulder"]]) / 2
    np.testing.assert_allclose(mid_hip, 0, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(mid_sh, axis=1), 1.0, atol=1e-5)
    assert out["scale"] == pytest.approx(100.0)
    np.testing.assert_allclose(out["vel"][1:, pu.K["left_hip"], 0], 0.1, atol=1e-5)  # 10 px/frame = 0.1 torso
    assert (out["vel"][0] == 0).all()


def test_normalize_pose_same_for_different_resolutions(standing_pose):
    small = pu.normalize_pose(standing_pose(160, 120, 50)[None], np.array([180.0]))
    big = pu.normalize_pose(standing_pose(640, 480, 200)[None], np.array([720.0]))
    np.testing.assert_allclose(small["seq"], big["seq"], atol=1e-5)


def test_normalize_pose_low_conf_keypoint_is_interpolated(standing_pose):
    kpts = np.stack([standing_pose() for _ in range(3)])
    kpts[1, pu.K["left_wrist"]] = (0, 0, 0.1)  # unreliable garbage in the middle frame
    out = pu.normalize_pose(kpts, np.full(3, 360.0))
    np.testing.assert_allclose(out["seq"][1, pu.K["left_wrist"]], out["seq"][0, pu.K["left_wrist"]], atol=1e-5)
    assert out["conf"][1, pu.K["left_wrist"]] == pytest.approx(0.1)


def test_clip_to_sequence_empty_clip_is_finite(make_ext):
    s = pu.clip_to_sequence(make_ext())
    assert s["has_person"] is False
    for key in ("seq", "vel", "conf", "n_people"):
        assert np.isfinite(s[key]).all()
    assert s["seq"].shape == (20, 17, 2)
```

- [ ] **Step 2: Run to verify failure**

Run: `& "$PM\.venv\Scripts\python.exe" -m pytest "$PM\tests" -q`
Expected: FAIL with `AttributeError: module 'pose_utils' has no attribute 'select_primary_track'`.

- [ ] **Step 3: Implement** (append to `pose_utils.py`)

```python
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


def normalize_pose(kpts: np.ndarray, box_h: np.ndarray) -> dict:
    """Pixel keypoints -> body-relative coordinates: mid-hip at the origin, 1 unit = torso length.

    Removes camera resolution, zoom and position, so a 320x240 CCTV clip and a
    640x480 staged clip of the same movement give the same numbers.
    """
    xy = kpts[..., :2].astype(np.float32).copy()
    conf = np.nan_to_num(kpts[..., 2], nan=0.0).astype(np.float32)
    xy[conf < CONF_THRESH] = np.nan
    seen = ~np.isnan(xy[..., 0])
    torso_seen = seen[:, [K["left_shoulder"], K["right_shoulder"], K["left_hip"], K["right_hip"]]].all(axis=1)
    xy = fill_gaps(xy)

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

    seq = (xy - mid_hip[:, None, :]) / scale
    vel = np.zeros_like(xy)
    vel[1:] = (xy[1:] - xy[:-1]) / scale  # keeps whole-body motion (walking away, etc.)
    return {"seq": seq.astype(np.float32), "vel": vel.astype(np.float32), "conf": conf, "scale": scale}


def clip_to_sequence(ext: dict) -> dict:
    """Extraction dict -> normalised pose sequence of the primary person."""
    kpts, box_h, has_person = primary_sequence(ext)
    out = normalize_pose(kpts, box_h)
    out["n_people"] = np.asarray(ext["n_people"], dtype=np.float32)
    out["has_person"] = has_person
    return out
```

- [ ] **Step 4: Run tests to verify pass**

Run: `& "$PM\.venv\Scripts\python.exe" -m pytest "$PM\tests" -q`
Expected: `12 passed`.

---

### Task 3: Pose extraction with YOLO11-pose

**Files:**
- Modify: `PM\pose_utils.py` (append section)
- Test: `PM\tests\test_pose_utils.py` (append `test_compute_stride`), create `PM\tests\test_extraction.py`

**Interfaces:**
- Consumes: `select_primary_track`, `clip_to_sequence`, constants.
- Produces:
  - `compute_stride(fps: float, target_fps: int = TARGET_FPS) -> int`
  - `load_pose_model(weights=POSE_WEIGHTS) -> ultralytics.YOLO` (weights stored in `models/`)
  - `extract_video(model, video_path, target_fps=TARGET_FPS, max_persons=MAX_PERSONS) -> dict` (extraction dict, see Task 2)
  - `save_extraction(ext: dict, path) -> None`, `load_extraction(path) -> dict`
  - `read_frame(video_path, frame_index: int) -> np.ndarray (H,W,3) RGB`
  - `draw_pose_frame(ax, video_path, ext: dict, t: int) -> None`

- [ ] **Step 1: Write the failing tests**

Append to `test_pose_utils.py`:

```python
def test_compute_stride():
    assert pu.compute_stride(30) == 3
    assert pu.compute_stride(60) == 6
    assert pu.compute_stride(25) == 2
    assert pu.compute_stride(5) == 1
    assert pu.compute_stride(0) == 3
    assert pu.compute_stride(float("nan")) == 3
```

Create `PM\tests\test_extraction.py`:

```python
import numpy as np
import pytest

import pose_utils as pu

pytestmark = pytest.mark.skipif(not (pu.DATA_DIR / "manifest.csv").exists(), reason="dataset not available")


@pytest.fixture(scope="module")
def pose_model():
    return pu.load_pose_model()


def test_extract_video_missing_file_raises(pose_model, tmp_path):
    with pytest.raises(FileNotFoundError):
        pu.extract_video(pose_model, tmp_path / "nope.mp4")


def test_extract_video_unreadable_file_raises(pose_model, tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"this is not a video")
    with pytest.raises(ValueError):
        pu.extract_video(pose_model, bad)


def test_extract_video_real_clip_roundtrip(pose_model, tmp_path):
    m = pu.load_manifest()
    row = m[m.source == "staged"].sort_values("duration_seconds").iloc[0]
    ext = pu.extract_video(pose_model, row.video_path)
    T = ext["kpts"].shape[0]
    assert abs(T - row.duration_seconds * pu.TARGET_FPS) <= 3
    assert ext["kpts"].shape == (T, pu.MAX_PERSONS, 17, 3)
    assert ext["boxes"].shape == (T, pu.MAX_PERSONS, 4)
    assert ext["track_ids"].shape == (T, pu.MAX_PERSONS)
    assert ext["stride"] == 3
    assert (ext["width"], ext["height"]) == (row.width, row.height)
    assert pu.clip_to_sequence(ext)["has_person"]

    pu.save_extraction(ext, tmp_path / "sub" / "x.npz")
    back = pu.load_extraction(tmp_path / "sub" / "x.npz")
    np.testing.assert_array_equal(back["kpts"], ext["kpts"])
    assert back["stride"] == 3 and isinstance(back["width"], int)
```

- [ ] **Step 2: Run to verify failure**

Run: `& "$PM\.venv\Scripts\python.exe" -m pytest "$PM\tests" -q`
Expected: FAIL with `AttributeError: ... 'compute_stride'` / `'load_pose_model'`.

- [ ] **Step 3: Implement** (append to `pose_utils.py`)

```python
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
```

- [ ] **Step 4: Run tests to verify pass**

Run: `& "$PM\.venv\Scripts\python.exe" -m pytest "$PM\tests" -q`
Expected: `16 passed` (first run downloads `yolo11n-pose.pt` into `models/`).

---

### Task 4: Notebook 01 — pose extraction (+ full extraction run)

**Files:**
- Create: `SCRATCH\nb_src\01_pose_extraction.py` → `PM\01_pose_extraction.ipynb`
- Outputs: `PM\cache\keypoints\**\*.npz` (207), `PM\results\pose_quality.csv`, `PM\results\extraction_log.csv`, `PM\figures\01_dataset_overview.png`, `PM\figures\01_pose_examples.png`

**Interfaces:**
- Consumes: `load_manifest, load_pose_model, extract_video, save_extraction, load_extraction, primary_sequence, draw_pose_frame`, dir constants.
- Produces: one cached `.npz` per manifest row at `row.cache_path`.

- [ ] **Step 1: Write the notebook source** — `SCRATCH\nb_src\01_pose_extraction.py`

```python
# %% [markdown]
# # 01 · Pose Extraction
#
# **Goal:** turn every video clip into a sequence of human body keypoints.
#
# We run **YOLO11-pose** (Ultralytics) on each clip at about 10 frames per second. For every sampled frame it finds each person and returns 17 **COCO keypoints** (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles), each with an (x, y) pixel position and a confidence score. The built-in **ByteTrack** tracker gives every person an ID that stays the same across frames, so later we can follow one person through the whole clip.
#
# Output: one compressed `.npz` file per clip in `cache/keypoints/`. This is the slow step (roughly an hour on CPU), so it is **resumable**: clips already cached are skipped when you re-run.
#
# > Kernel: select **Python (pose-shoplifting)** (the project's own virtual environment).

# %%
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pose_utils as pu

for d in (pu.CACHE_DIR, pu.MODELS_DIR, pu.FIGURES_DIR, pu.RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
print("Dataset folder:", pu.DATA_DIR)

# %% [markdown]
# ## 1. The dataset
#
# `manifest.csv` lists every clip with its label, source dataset and fixed train / validation / test split. We add a coarse `source` column (`staged` vs `dcsass`) because the two sources look very different, and we need to keep an eye on that.

# %%
manifest = pu.load_manifest()
print(f"{len(manifest)} clips")
pd.crosstab([manifest["source"], manifest["label_name"]], manifest["split"], margins=True)

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
for src, g in manifest.groupby("source"):
    axes[0].hist(g["duration_seconds"], bins=20, alpha=0.7, label=src)
axes[0].set(title="Clip duration by source", xlabel="seconds", ylabel="clips")
axes[0].legend()
manifest.groupby(["source", "label_name"]).size().unstack().plot.bar(ax=axes[1], rot=0)
axes[1].set(title="Clips per source and label", ylabel="clips")
plt.tight_layout()
plt.savefig(pu.FIGURES_DIR / "01_dataset_overview.png", dpi=150)
plt.show()

# %% [markdown]
# **A confound to watch:** the DCSASS clips are long, low-resolution CCTV footage and 24 of the 28 are shoplifting, while the staged clips are short and balanced. A model could learn "looks like DCSASS ⇒ shoplifting" instead of the behaviour itself. Two defences:
# 1. keypoints are normalised to the person's body size (notebooks 02/03), which removes resolution and camera distance;
# 2. clip length is never used as a feature.
#
# Notebook 04 reports test results **per source** so we can check whether it worked.

# %% [markdown]
# ## 2. Load the pose model and time one clip

# %%
pose_model = pu.load_pose_model()
sample = manifest.sort_values("duration_seconds").iloc[0]
t0 = time.time()
ext = pu.extract_video(pose_model, sample["video_path"])
dt = time.time() - t0
n = ext["kpts"].shape[0]
print(f"{sample['relative_path']}: {n} sampled frames in {dt:.1f}s ({n / dt:.1f} frames/s)")
print({k: v.shape for k, v in ext.items() if isinstance(v, np.ndarray)})
est_frames = (manifest["duration_seconds"] * pu.TARGET_FPS).sum()
print(f"Whole dataset: ~{est_frames:,.0f} frames -> roughly {est_frames / (n / dt) / 60:.0f} minutes")

# %% [markdown]
# ## 3. Extract every clip (resumable)

# %%
todo = manifest[[not Path(p).exists() for p in manifest["cache_path"]]]
print(f"{len(manifest) - len(todo)} already cached, {len(todo)} to extract")
log = []
start = time.time()
for i, row in enumerate(todo.itertuples(), 1):
    t0 = time.time()
    try:
        ext = pu.extract_video(pose_model, row.video_path)
        pu.save_extraction(ext, row.cache_path)
        log.append({"relative_path": row.relative_path, "frames": len(ext["kpts"]), "seconds": time.time() - t0, "error": ""})
    except Exception as e:  # keep going; failures are listed below
        log.append({"relative_path": row.relative_path, "frames": 0, "seconds": time.time() - t0, "error": repr(e)})
    if i % 10 == 0 or i == len(todo):
        print(f"  {i}/{len(todo)} clips, {(time.time() - start) / 60:.1f} min elapsed")
log = pd.DataFrame(log, columns=["relative_path", "frames", "seconds", "error"])
if len(log):
    log.to_csv(pu.RESULTS_DIR / "extraction_log.csv", index=False)
failed = log[log["error"] != ""]
print(f"Failed clips: {len(failed)}")
failed

# %% [markdown]
# ## 4. Pose quality check
#
# For each clip we pick the **primary person**: the tracked person seen in the most frames. If the primary person is missing from many frames, or keypoint confidence is low, that clip's pose signal is weak.

# %%
rows = []
for row in manifest.itertuples():
    ext = pu.load_extraction(row.cache_path)
    kpts, _, has_person = pu.primary_sequence(ext)
    present = ~np.isnan(kpts[..., 2]).all(axis=1)
    rows.append({
        "relative_path": row.relative_path, "source": row.source, "label_name": row.label_name,
        "frames": len(kpts), "primary_rate": present.mean(), "mean_people": ext["n_people"].mean(),
        "mean_kpt_conf": float(np.nanmean(kpts[present][..., 2])) if present.any() else 0.0,
    })
quality = pd.DataFrame(rows)
quality.to_csv(pu.RESULTS_DIR / "pose_quality.csv", index=False)
print("Clips with nobody detected:", int((quality["primary_rate"] == 0).sum()))
quality.groupby(["source", "label_name"])[["frames", "primary_rate", "mean_people", "mean_kpt_conf"]].mean().round(3)

# %% [markdown]
# ## 5. Visual sanity check
#
# Skeletons on the middle frame of one clip per (source, label). The **primary person is red**, everyone else is grey.

# %%
examples = pd.concat([
    manifest[(manifest.source == s) & (manifest.label_name == l)].head(1)
    for s, l in [("staged", "normal"), ("staged", "shoplifting"), ("dcsass", "normal"), ("dcsass", "shoplifting")]
])
fig, axes = plt.subplots(1, len(examples), figsize=(4.5 * len(examples), 4))
for ax, row in zip(axes, examples.itertuples()):
    ext = pu.load_extraction(row.cache_path)
    pu.draw_pose_frame(ax, row.video_path, ext, len(ext["kpts"]) // 2)
    ax.set_title(f"{row.source} / {row.label_name}\n{Path(row.relative_path).name}", fontsize=9)
plt.tight_layout()
plt.savefig(pu.FIGURES_DIR / "01_pose_examples.png", dpi=150)
plt.show()
```

- [ ] **Step 2: Build the notebook**

Run: `& "$PM\.venv\Scripts\python.exe" "$SCRATCH\build_nb.py" "$SCRATCH\nb_src\01_pose_extraction.py" "$PM\01_pose_extraction.ipynb"`

- [ ] **Step 3: Execute in the background (long)**

Run (background): `& "$PM\.venv\Scripts\jupyter.exe" nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=pose-shoplifting "$PM\01_pose_extraction.ipynb"`
Expected: exits 0, `Failed clips: 0`, 207 `.npz` files under `cache\keypoints`.

- [ ] **Step 4: Verify outputs**

Run: `(Get-ChildItem "$PM\cache\keypoints" -Recurse -Filter *.npz).Count` → `207`. Open `figures\01_pose_examples.png`: skeletons line up with people.

---

### Task 5: Approach A — features + notebook 02

**Files:**
- Modify: `PM\pose_utils.py` (append), `PM\tests\test_pose_utils.py` (append)
- Create: `SCRATCH\nb_src\02_features_ml.py` → `PM\02_features_ml.ipynb`
- Outputs: `results\clip_features.csv`, `results\approach_a_model_comparison.csv`, `models\approach_a.joblib`, `models\approach_a_meta.json`, `figures\02_*.png`

**Interfaces:**
- Consumes: `clip_to_sequence`, `load_extraction`, `load_manifest`.
- Produces:
  - `frame_features(s: dict) -> pd.DataFrame` (T rows × 21 columns)
  - `clip_features(s: dict) -> pd.Series` (127 values: `<feature>_<stat>` for stat in `STATS`, plus `hand_near_hip_frac`; all zeros when `s["has_person"]` is False)
  - `STATS = ("mean", "std", "min", "max", "p10", "p90")`
  - `models\approach_a_meta.json` keys: `model, feature_names, cv_f1, val_f1, val_auc, val_acc, best_params`

- [ ] **Step 1: Write the failing tests** (append)

```python
def _sequence_from_poses(kpts):
    T = len(kpts)
    return {**pu.normalize_pose(kpts, np.full(T, 360.0)), "n_people": np.ones(T, np.float32), "has_person": True}


def test_frame_features_shape(standing_pose):
    ff = pu.frame_features(_sequence_from_poses(np.stack([standing_pose() for _ in range(6)])))
    assert ff.shape == (6, 21)
    assert np.isfinite(ff.to_numpy()).all()


def test_clip_features_values(standing_pose):
    f = pu.clip_features(_sequence_from_poses(np.stack([standing_pose() for _ in range(10)])))
    assert len(f) == 21 * 6 + 1
    assert np.isfinite(f.to_numpy()).all()
    assert f["hand_near_hip_frac"] == pytest.approx(1.0)  # wrists are 0.25 torso lengths from the hips
    assert f["lelbow_angle_mean"] > 150                   # arms hang almost straight
    assert f["torso_lean_mean"] == pytest.approx(0.0, abs=1e-3)
    assert f["n_people_mean"] == pytest.approx(1.0)


def test_clip_features_no_person_is_all_zero(make_ext):
    f = pu.clip_features(pu.clip_to_sequence(make_ext()))
    assert len(f) == 127 and (f.to_numpy() == 0).all()
```

- [ ] **Step 2: Run to verify failure**

Run: `& "$PM\.venv\Scripts\python.exe" -m pytest "$PM\tests\test_pose_utils.py" -q`
Expected: FAIL with `AttributeError: ... 'frame_features'`.

- [ ] **Step 3: Implement** (append to `pose_utils.py`)

```python
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
```

- [ ] **Step 4: Run tests to verify pass**

Run: `& "$PM\.venv\Scripts\python.exe" -m pytest "$PM\tests\test_pose_utils.py" -q`
Expected: all pass.

- [ ] **Step 5: Write the notebook source** — `SCRATCH\nb_src\02_features_ml.py`

```python
# %% [markdown]
# # 02 · Approach A: hand-crafted pose features + classic machine learning
#
# **Idea:** shoplifting is mostly about what the **hands** do relative to the **body**: reaching, then moving an item to a pocket, waistband or bag, often while crouching or turned away. We measure these things directly from the keypoints and let standard classifiers learn which ones matter.
#
# Steps for every clip:
# 1. take the **primary person** (the person tracked for the longest);
# 2. **normalise** the pose: put the mid-hip at (0, 0) and measure everything in **torso lengths**, so camera resolution and distance do not matter;
# 3. compute **21 measurements per frame** (wrist-to-hip distances, elbow/knee angles, torso lean, crouch, hand speed, hand visibility, number of people, ...);
# 4. summarise each over the clip with **6 statistics** (mean, std, min, max, 10th and 90th percentile), plus the fraction of time a hand is near the hip/pocket. This gives **127 features per clip**.
#
# Then three classifiers are tuned with 5-fold cross-validation on the training set and compared on the validation set. The test set is **not** used here.

# %%
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import pose_utils as pu

manifest = pu.load_manifest()

# %% [markdown]
# ## 1. Build one feature row per clip

# %%
X_all = pd.DataFrame([pu.clip_features(pu.clip_to_sequence(pu.load_extraction(p))) for p in manifest["cache_path"]])
feature_names = list(X_all.columns)
meta_cols = ["relative_path", "label", "label_name", "source", "split"]
features = pd.concat([manifest[meta_cols].reset_index(drop=True), X_all], axis=1)
features.to_csv(pu.RESULTS_DIR / "clip_features.csv", index=False)
print(f"{features.shape[0]} clips x {len(feature_names)} features")
features.head()

# %% [markdown]
# ### What a feature looks like over time
#
# Distance from each wrist to the same-side hip, in torso lengths, for one normal and one shoplifting clip. The dashed line is the "hand near hip/pocket" threshold.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), sharey=True)
for ax, label in zip(axes, ["normal", "shoplifting"]):
    row = manifest[(manifest.source == "staged") & (manifest.label_name == label) & (manifest.split == "train")].iloc[0]
    ff = pu.frame_features(pu.clip_to_sequence(pu.load_extraction(row.cache_path)))
    t = np.arange(len(ff)) / pu.TARGET_FPS
    ax.plot(t, ff["lwrist_lhip"], label="left wrist to left hip")
    ax.plot(t, ff["rwrist_rhip"], label="right wrist to right hip")
    ax.axhline(pu.NEAR_HIP, ls="--", c="grey", lw=1)
    ax.set(title=f"{label}: {Path(row.relative_path).name}", xlabel="seconds", ylabel="torso lengths")
axes[0].legend()
plt.tight_layout()
plt.savefig(pu.FIGURES_DIR / "02_wrist_hip_timeline.png", dpi=150)
plt.show()

# %% [markdown]
# ## 2. Train and compare three classifiers
#
# - **Logistic Regression**: a linear baseline (features standardised first).
# - **Random Forest**: many decision trees voting; robust on small data.
# - **Histogram Gradient Boosting**: trees built one after another, each fixing the previous ones' mistakes.
#
# All use `class_weight="balanced"` (116 shoplifting vs 91 normal clips). Hyperparameters are chosen by 5-fold stratified cross-validation on **train** (scored by F1), and models are then compared on **validation**.

# %%
train = features[features.split == "train"]
val = features[features.split == "validation"]
X_train, y_train = train[feature_names], train["label"]
X_val, y_val = val[feature_names], val["label"]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=pu.SEED)

candidates = {
    "LogisticRegression": (
        make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000)),
        {"logisticregression__C": [0.01, 0.1, 1, 10]},
    ),
    "RandomForest": (
        RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=pu.SEED),
        {"max_depth": [None, 5, 10], "min_samples_leaf": [1, 3, 5], "max_features": ["sqrt", 0.3]},
    ),
    "HistGradientBoosting": (
        HistGradientBoostingClassifier(class_weight="balanced", random_state=pu.SEED),
        {"learning_rate": [0.05, 0.1], "max_depth": [3, None], "max_iter": [100, 300], "l2_regularization": [0.0, 1.0]},
    ),
}
rows, fitted = [], {}
for name, (estimator, grid) in candidates.items():
    search = GridSearchCV(estimator, grid, scoring="f1", cv=cv, n_jobs=-1)
    search.fit(X_train, y_train)
    p_val = search.best_estimator_.predict_proba(X_val)[:, 1]
    rows.append({"model": name, "cv_f1": search.best_score_,
                 "val_f1": f1_score(y_val, p_val >= 0.5), "val_auc": roc_auc_score(y_val, p_val),
                 "val_acc": accuracy_score(y_val, p_val >= 0.5), "best_params": search.best_params_})
    fitted[name] = search.best_estimator_
    print(f"{name:22s} cv_f1={search.best_score_:.3f}  val_f1={rows[-1]['val_f1']:.3f}  val_auc={rows[-1]['val_auc']:.3f}")
comparison = pd.DataFrame(rows).sort_values(["val_f1", "val_auc"], ascending=False).reset_index(drop=True)
comparison.drop(columns="best_params").to_csv(pu.RESULTS_DIR / "approach_a_model_comparison.csv", index=False)
comparison

# %%
ax = comparison.set_index("model")[["cv_f1", "val_f1", "val_auc", "val_acc"]].plot.bar(rot=0, figsize=(9, 3.8), ylim=(0, 1.05))
ax.set(title="Approach A: classifier comparison", ylabel="score", xlabel="")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(pu.FIGURES_DIR / "02_model_comparison.png", dpi=150)
plt.show()

# %% [markdown]
# ## 3. Save the best model (chosen on validation F1, ties broken by ROC-AUC)

# %%
best = comparison.iloc[0]
best_model = fitted[best["model"]]
joblib.dump(best_model, pu.MODELS_DIR / "approach_a.joblib")
meta = {"model": best["model"], "feature_names": feature_names,
        "cv_f1": float(best["cv_f1"]), "val_f1": float(best["val_f1"]),
        "val_auc": float(best["val_auc"]), "val_acc": float(best["val_acc"]),
        "best_params": best["best_params"]}
(pu.MODELS_DIR / "approach_a_meta.json").write_text(json.dumps(meta, indent=2, default=str))
print("Saved", best["model"], "with params", best["best_params"])

# %% [markdown]
# ## 4. Which features matter?
#
# **Permutation importance** on the validation set: shuffle one feature at a time and measure how much ROC-AUC drops. A bigger drop means the model relies on that feature more.

# %%
imp = permutation_importance(best_model, X_val, y_val, scoring="roc_auc", n_repeats=30, random_state=pu.SEED, n_jobs=-1)
importance = pd.Series(imp.importances_mean, index=feature_names).sort_values(ascending=False)
importance.to_csv(pu.RESULTS_DIR / "approach_a_feature_importance.csv", header=["importance"])
ax = importance.head(20)[::-1].plot.barh(figsize=(7, 6))
ax.set(title=f"Top 20 features ({best['model']}, permutation importance)", xlabel="mean drop in validation ROC-AUC")
plt.tight_layout()
plt.savefig(pu.FIGURES_DIR / "02_feature_importance.png", dpi=150)
plt.show()

# %%
top = list(importance.head(4).index)
fig, axes = plt.subplots(1, len(top), figsize=(4 * len(top), 3.5))
for ax, f in zip(axes, top):
    data = [features.loc[features.label_name == lbl, f] for lbl in ["normal", "shoplifting"]]
    ax.boxplot(data, tick_labels=["normal", "shoplifting"])
    ax.set_title(f, fontsize=10)
plt.suptitle("Top features by class (all clips)")
plt.tight_layout()
plt.savefig(pu.FIGURES_DIR / "02_top_features_by_class.png", dpi=150)
plt.show()
```

- [ ] **Step 6: Build and execute**

Run:
```powershell
& "$PM\.venv\Scripts\python.exe" "$SCRATCH\build_nb.py" "$SCRATCH\nb_src\02_features_ml.py" "$PM\02_features_ml.ipynb"
& "$PM\.venv\Scripts\jupyter.exe" nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=pose-shoplifting "$PM\02_features_ml.ipynb"
```
Expected: exit 0; `models\approach_a.joblib` and `models\approach_a_meta.json` exist; comparison table printed.

---

### Task 6: Approach B — sequence utilities + notebook 03

**Files:**
- Modify: `PM\pose_utils.py` (append), `PM\tests\test_pose_utils.py` (append)
- Create: `SCRATCH\nb_src\03_sequence_model.py` → `PM\03_sequence_model.ipynb`
- Outputs: `models\approach_b.pt`, `models\approach_b_meta.json`, `results\approach_b_history.csv`, `results\approach_b_aggregation.csv`, `figures\03_*.png`

**Interfaces:**
- Consumes: `clip_to_sequence`, `load_extraction`, `FLIP_IDX`, constants.
- Produces:
  - `sequence_input(s: dict) -> np.ndarray (T, 85)` = `[seq (34) | vel (34) | conf (17)]`
  - `window_starts(T: int, window=WINDOW, stride=EVAL_STRIDE) -> list[int]`
  - `make_windows(x (T,D), window=WINDOW, stride=EVAL_STRIDE) -> (windows (N,window,D) float32, masks (N,window) float32)`
  - `mirror(x (..., 85)) -> same shape`, `augment(w (W,85), rng: np.random.Generator) -> (W,85) float32`
  - `class PoseGRU(nn.Module)`: `__init__(n_features=N_FEATURES, hidden=64, layers=2, dropout=0.3)`, `forward(x (B,W,85) tensor, mask (B,W) tensor) -> logits (B,)`
  - `window_probs(model, x (T,85), window=WINDOW, stride=EVAL_STRIDE) -> np.ndarray (N,)`
  - `AGGREGATIONS = ("mean", "max", "top3")`, `aggregate(probs, method="mean") -> float`
  - `models\approach_b_meta.json` keys: `hidden, layers, dropout, window, eval_stride, aggregation, best_epoch, val_f1, val_auc, val_acc`

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_sequence_input_layout(standing_pose):
    x = pu.sequence_input(_sequence_from_poses(np.stack([standing_pose() for _ in range(5)])))
    assert x.shape == (5, pu.N_FEATURES) and x.dtype == np.float32
    np.testing.assert_allclose(x[:, 68:], 0.9)  # confidences last


def test_make_windows_short_clip_padded():
    x = np.ones((10, pu.N_FEATURES), np.float32)
    w, m = pu.make_windows(x)
    assert w.shape == (1, pu.WINDOW, pu.N_FEATURES)
    assert m[0, :10].all() and not m[0, 10:].any()
    assert (w[0, 10:] == 0).all()


def test_make_windows_long_clip_covers_end():
    x = np.arange(100, dtype=np.float32)[:, None].repeat(3, axis=1)
    w, m = pu.make_windows(x, window=32, stride=8)
    assert w.shape == (10, 32, 3)  # starts 0, 8, ..., 64 plus a final window starting at 68
    assert w[-1, -1, 0] == 99 and m.all()
    assert pu.window_starts(100, 32, 8) == [0, 8, 16, 24, 32, 40, 48, 56, 64, 68]
    assert pu.window_starts(20) == [0]


def test_mirror_swaps_and_negates():
    x = np.random.default_rng(0).normal(size=(5, pu.N_FEATURES)).astype(np.float32)
    y = pu.mirror(x)
    np.testing.assert_allclose(pu.mirror(y), x, atol=1e-6)
    lw, rw = pu.K["left_wrist"], pu.K["right_wrist"]
    assert y[0, 2 * lw] == pytest.approx(-x[0, 2 * rw])           # x coordinate: swapped and negated
    assert y[0, 2 * lw + 1] == pytest.approx(x[0, 2 * rw + 1])    # y coordinate: swapped only
    assert y[0, 34 + 2 * lw] == pytest.approx(-x[0, 34 + 2 * rw])  # velocity x
    assert y[0, 68 + lw] == pytest.approx(x[0, 68 + rw])           # confidence: swapped only


def test_augment_keeps_shape_and_confidences():
    w = np.random.default_rng(1).normal(size=(pu.WINDOW, pu.N_FEATURES)).astype(np.float32)
    w[:, 68:] = 0.5
    a = pu.augment(w, np.random.default_rng(0))
    assert a.shape == w.shape and a.dtype == np.float32
    np.testing.assert_allclose(a[:, 68:], 0.5)


def test_posegru_padding_invariance():
    torch.manual_seed(0)
    model = pu.PoseGRU().eval()
    x = torch.randn(2, pu.WINDOW, pu.N_FEATURES)
    m = torch.ones(2, pu.WINDOW)
    m[1, 20:] = 0
    out = model(x, m)
    assert out.shape == (2,)
    x2 = x.clone()
    x2[1, 20:] = 99.0  # garbage in the padded part must not change the output
    torch.testing.assert_close(model(x2, m), out)


def test_window_probs_and_aggregate():
    p = pu.window_probs(pu.PoseGRU(), np.zeros((50, pu.N_FEATURES), np.float32))
    assert p.shape == (4,) and ((p > 0) & (p < 1)).all()  # windows start at 0, 8, 16, 18
    probs = [0.1, 0.2, 0.9, 0.8, 0.3]
    assert pu.aggregate(probs, "mean") == pytest.approx(0.46)
    assert pu.aggregate(probs, "max") == pytest.approx(0.9)
    assert pu.aggregate(probs, "top3") == pytest.approx((0.9 + 0.8 + 0.3) / 3)
    with pytest.raises(ValueError):
        pu.aggregate(probs, "median")


def test_empty_clip_through_gru_is_finite(make_ext):
    x = pu.sequence_input(pu.clip_to_sequence(make_ext()))
    assert np.isfinite(pu.window_probs(pu.PoseGRU(), x)).all()
```

- [ ] **Step 2: Run to verify failure**

Run: `& "$PM\.venv\Scripts\python.exe" -m pytest "$PM\tests\test_pose_utils.py" -q`
Expected: FAIL with `AttributeError: ... 'sequence_input'`.

- [ ] **Step 3: Implement** (append to `pose_utils.py`)

```python
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
```

- [ ] **Step 4: Run tests to verify pass**

Run: `& "$PM\.venv\Scripts\python.exe" -m pytest "$PM\tests\test_pose_utils.py" -q`
Expected: all pass.

- [ ] **Step 5: Write the notebook source** — `SCRATCH\nb_src\03_sequence_model.py`

```python
# %% [markdown]
# # 03 · Approach B: sequence model (BiGRU) on keypoint windows
#
# Approach A summarises a whole clip into statistics, so it cannot see the **order** of movements: "reach to shelf → pause → hand to pocket" looks the same as the reverse. Here a small recurrent network reads the pose **frame by frame**.
#
# - **Input per frame (85 numbers):** 34 normalised keypoint coordinates (17 joints × x, y), 34 velocities (how far each joint moved since the previous frame), 17 keypoint confidences (low = joint hidden or uncertain).
# - **Windows:** clips are cut into **3.2-second windows** (32 frames at 10 fps). Every window inherits its clip's label. In each training epoch we draw up to **8 random windows per clip**, so the long DCSASS clips do not outweigh the short staged ones, and the model sees many slightly different views of each clip.
# - **Augmentation (training only):** random left/right mirroring, ±10% scale jitter, small Gaussian noise.
# - **Model:** linear embedding → 2-layer **bidirectional GRU** (hidden size 64) → average over time → probability of shoplifting.
# - **Clip prediction:** run every window of a clip (step of 8 frames) and combine the window probabilities (**mean**, **max** or **top-3 mean**, chosen on validation).
#
# Early stopping uses the validation clip-level ROC-AUC. The test set is **not** used here.

# %%
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn

import pose_utils as pu

torch.manual_seed(pu.SEED)
manifest = pu.load_manifest()

# %% [markdown]
# ## 1. Load sequences

# %%
seqs = {p: pu.sequence_input(pu.clip_to_sequence(pu.load_extraction(c)))
        for p, c in zip(manifest["relative_path"], manifest["cache_path"])}
manifest["frames"] = manifest["relative_path"].map(lambda p: len(seqs[p]))
print("Per-frame input size:", next(iter(seqs.values())).shape[1])
manifest.groupby("source")["frames"].describe()


# %%
def videos(split):
    m = manifest[manifest.split == split]
    return [(seqs[p], y) for p, y in zip(m["relative_path"], m["label"])]


train_videos, val_videos = videos("train"), videos("validation")
y_train = np.array([y for _, y in train_videos])
y_val = np.array([y for _, y in val_videos])
print(f"train clips: {len(train_videos)}, validation clips: {len(val_videos)}")

# %% [markdown]
# ## 2. Train

# %%
HIDDEN, LAYERS, DROPOUT = 64, 2, 0.3
WINDOWS_PER_VIDEO, BATCH, EPOCHS, PATIENCE = 8, 64, 80, 15
rng = np.random.default_rng(pu.SEED)


def epoch_windows(vids):
    """Up to WINDOWS_PER_VIDEO random augmented windows from every clip."""
    W, M, Y = [], [], []
    for x, y in vids:
        if len(x) <= pu.WINDOW:
            w, m = pu.make_windows(x)
            picks = [(w[0], m[0])]
        else:
            starts = rng.integers(0, len(x) - pu.WINDOW + 1, size=WINDOWS_PER_VIDEO)
            picks = [(x[s:s + pu.WINDOW], np.ones(pu.WINDOW, np.float32)) for s in starts]
        for w, m in picks:
            W.append(pu.augment(w, rng))
            M.append(m)
            Y.append(y)
    return torch.from_numpy(np.stack(W)), torch.from_numpy(np.stack(M)), torch.tensor(Y, dtype=torch.float32)


def clip_probs(model, vids, method="mean"):
    return np.array([pu.aggregate(pu.window_probs(model, x), method) for x, _ in vids])


model = pu.PoseGRU(hidden=HIDDEN, layers=LAYERS, dropout=DROPOUT)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
pos_weight = torch.tensor((y_train == 0).sum() / (y_train == 1).sum(), dtype=torch.float32)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
print(f"{sum(p.numel() for p in model.parameters()):,} trainable parameters")

history, best_auc, best_state, best_epoch, waited = [], -1.0, None, 0, 0
for epoch in range(1, EPOCHS + 1):
    model.train()
    Xw, Mw, Yw = epoch_windows(train_videos)
    order = torch.randperm(len(Yw))
    total = 0.0
    for i in range(0, len(order), BATCH):
        idx = order[i:i + BATCH]
        optimizer.zero_grad()
        loss = loss_fn(model(Xw[idx], Mw[idx]), Yw[idx])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * len(idx)
    p_val = clip_probs(model, val_videos)
    val_auc, val_acc = roc_auc_score(y_val, p_val), accuracy_score(y_val, p_val >= 0.5)
    history.append({"epoch": epoch, "train_loss": total / len(order), "val_auc": val_auc, "val_acc": val_acc})
    if val_auc > best_auc:
        best_auc, best_epoch, waited = val_auc, epoch, 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        waited += 1
    if epoch == 1 or epoch % 5 == 0:
        print(f"epoch {epoch:3d}  train_loss {history[-1]['train_loss']:.4f}  val_auc {val_auc:.3f}  val_acc {val_acc:.3f}")
    if waited >= PATIENCE:
        print(f"Early stopping at epoch {epoch}")
        break
model.load_state_dict(best_state)
history = pd.DataFrame(history)
history.to_csv(pu.RESULTS_DIR / "approach_b_history.csv", index=False)
print(f"Best epoch {best_epoch}: validation ROC-AUC {best_auc:.3f}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].plot(history["epoch"], history["train_loss"])
axes[0].set(title="Training loss", xlabel="epoch", ylabel="BCE loss")
axes[1].plot(history["epoch"], history["val_auc"], label="val ROC-AUC")
axes[1].plot(history["epoch"], history["val_acc"], label="val accuracy")
axes[1].axvline(best_epoch, ls="--", c="grey", lw=1, label=f"best epoch ({best_epoch})")
axes[1].set(title="Validation (clip level)", xlabel="epoch", ylim=(0, 1.05))
axes[1].legend()
plt.tight_layout()
plt.savefig(pu.FIGURES_DIR / "03_training_curves.png", dpi=150)
plt.show()

# %% [markdown]
# ## 3. How to combine window predictions into a clip prediction
#
# - **mean**: average over all windows (shoplifting has to be visible for much of the clip);
# - **max**: the single most suspicious window decides (good when the theft is brief, but sensitive to one false alarm);
# - **top-3 mean**: a compromise.
#
# The method with the best validation F1 (ties broken by ROC-AUC) is kept.

# %%
agg = []
for method in pu.AGGREGATIONS:
    p = clip_probs(model, val_videos, method)
    agg.append({"aggregation": method, "val_f1": f1_score(y_val, p >= 0.5),
                "val_auc": roc_auc_score(y_val, p), "val_acc": accuracy_score(y_val, p >= 0.5)})
agg = pd.DataFrame(agg).sort_values(["val_f1", "val_auc"], ascending=False).reset_index(drop=True)
agg.to_csv(pu.RESULTS_DIR / "approach_b_aggregation.csv", index=False)
agg

# %%
best_agg = agg.iloc[0]
torch.save(model.state_dict(), pu.MODELS_DIR / "approach_b.pt")
meta = {"hidden": HIDDEN, "layers": LAYERS, "dropout": DROPOUT, "window": pu.WINDOW, "eval_stride": pu.EVAL_STRIDE,
        "aggregation": best_agg["aggregation"], "best_epoch": int(best_epoch),
        "val_f1": float(best_agg["val_f1"]), "val_auc": float(best_agg["val_auc"]), "val_acc": float(best_agg["val_acc"])}
(pu.MODELS_DIR / "approach_b_meta.json").write_text(json.dumps(meta, indent=2))
meta

# %% [markdown]
# ## 4. Where in a clip does the model see shoplifting?
#
# Window-by-window probability across validation clips. For long CCTV clips this shows *when* the model gets suspicious.

# %%
val_m = manifest[manifest.split == "validation"]
picks = pd.concat([val_m[(val_m.source == s) & (val_m.label_name == l)].head(1)
                   for s, l in [("staged", "normal"), ("staged", "shoplifting"), ("dcsass", "shoplifting")]])
fig, axes = plt.subplots(1, len(picks), figsize=(5 * len(picks), 3.2), sharey=True)
for ax, row in zip(np.atleast_1d(axes), picks.itertuples()):
    x = seqs[row.relative_path]
    p = pu.window_probs(model, x)
    starts = np.array(pu.window_starts(len(x))) / pu.TARGET_FPS
    ax.plot(starts, p, marker="o", ms=3)
    ax.axhline(0.5, ls="--", c="grey", lw=1)
    ax.set(title=f"{row.source} / {row.label_name}", xlabel="window start (s)", ylim=(0, 1))
np.atleast_1d(axes)[0].set_ylabel("P(shoplifting)")
plt.tight_layout()
plt.savefig(pu.FIGURES_DIR / "03_window_probabilities.png", dpi=150)
plt.show()
```

- [ ] **Step 6: Build and execute**

Run:
```powershell
& "$PM\.venv\Scripts\python.exe" "$SCRATCH\build_nb.py" "$SCRATCH\nb_src\03_sequence_model.py" "$PM\03_sequence_model.ipynb"
& "$PM\.venv\Scripts\jupyter.exe" nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 --ExecutePreprocessor.kernel_name=pose-shoplifting "$PM\03_sequence_model.ipynb"
```
Expected: exit 0; `models\approach_b.pt` and `models\approach_b_meta.json` exist.

---

### Task 7: Predictor (`ShopliftingPredictor`, `predict_clip`)

**Files:**
- Modify: `PM\pose_utils.py` (append)
- Create: `PM\tests\test_predictor.py`

**Interfaces:**
- Consumes: everything above, plus `models\approach_a.joblib`, `approach_a_meta.json`, `approach_b.pt`, `approach_b_meta.json`, `best_model.json` (`{"approach": "A"|"B", ...}`, written by Task 8).
- Produces:
  - `class ShopliftingPredictor`: `__init__(approach="best")`; attributes `approach`, `description`, `meta`, lazy `pose_model`; `predict_from_extraction(ext) -> dict`; `predict(video_path, keep_extraction=False) -> dict`.
  - Result dict: `{"label": "shoplifting"|"normal", "probability": float P(shoplifting), "approach": "A"|"B", "no_person_detected": bool}` plus `"video"` (and `"extraction"` if requested) from `predict`.
  - `predict_clip(video_path, approach="best") -> dict`

- [ ] **Step 1: Write the failing tests** — `PM\tests\test_predictor.py`

```python
import pytest

import pose_utils as pu

needs_models = pytest.mark.skipif(
    not all((pu.MODELS_DIR / f).exists() for f in ["approach_a.joblib", "approach_b.pt", "best_model.json"]),
    reason="run notebooks 02-04 first",
)


def test_predictor_missing_models_message(tmp_path, monkeypatch):
    monkeypatch.setattr(pu, "MODELS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="04_evaluation"):
        pu.ShopliftingPredictor("best")
    with pytest.raises(FileNotFoundError, match="02_features_ml"):
        pu.ShopliftingPredictor("A")
    with pytest.raises(FileNotFoundError, match="03_sequence_model"):
        pu.ShopliftingPredictor("B")


def test_predictor_rejects_unknown_approach():
    with pytest.raises(ValueError):
        pu.ShopliftingPredictor("C")


@needs_models
@pytest.mark.parametrize("approach", ["A", "B"])
def test_predictor_no_person_clip(approach, make_ext):
    out = pu.ShopliftingPredictor(approach).predict_from_extraction(make_ext())
    assert out["no_person_detected"] is True
    assert out["label"] in ("normal", "shoplifting") and 0.0 <= out["probability"] <= 1.0
    assert out["approach"] == approach


@needs_models
def test_predict_clip_on_real_test_videos():
    m = pu.load_manifest()
    test = m[(m.split == "test") & (m.source == "staged")]
    predictor = pu.ShopliftingPredictor("best")
    for label in ("shoplifting", "normal"):
        out = predictor.predict(test[test.label_name == label].iloc[0].video_path)
        assert out["label"] in ("normal", "shoplifting") and 0.0 <= out["probability"] <= 1.0
        assert out["no_person_detected"] is False
```

- [ ] **Step 2: Run to verify failure**

Run: `& "$PM\.venv\Scripts\python.exe" -m pytest "$PM\tests\test_predictor.py" -q`
Expected: FAIL with `AttributeError: ... 'ShopliftingPredictor'`.

- [ ] **Step 3: Implement** (append to `pose_utils.py`)

```python
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
```

- [ ] **Step 4: Run tests**

Run: `& "$PM\.venv\Scripts\python.exe" -m pytest "$PM\tests\test_predictor.py" -q`
Expected: `2 passed, 3 skipped` (the skipped ones need `best_model.json` from Task 8 and are re-run in Task 9).

---

### Task 8: Notebook 04 — evaluation on the test set

**Files:**
- Create: `SCRATCH\nb_src\04_evaluation.py` → `PM\04_evaluation.ipynb`
- Outputs: `models\best_model.json`, `results\test_metrics.csv`, `results\test_metrics_by_source.csv`, `results\test_predictions.csv`, `figures\04_*.png`

**Interfaces:**
- Consumes: `ShopliftingPredictor("A"/"B").predict_from_extraction`, `load_extraction`, meta JSONs.
- Produces: `best_model.json = {"approach": "A"|"B", "description": str, "val_f1": float, "val_auc": float, "chosen_by": "validation F1, then validation ROC-AUC"}`.

- [ ] **Step 1: Write the notebook source** — `SCRATCH\nb_src\04_evaluation.py`

```python
# %% [markdown]
# # 04 · Evaluation on the held-out test set
#
# Both approaches are scored on the **33 test clips**, which neither model has seen. Both are scored through the same `ShopliftingPredictor` code that `05_predict.ipynb` uses, so these numbers describe exactly what a user gets.
#
# **Which model is "best" is decided on the validation set, not here**, so the test numbers stay an honest estimate.

# %%
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)

import pose_utils as pu

manifest = pu.load_manifest()
test = manifest[manifest.split == "test"].reset_index(drop=True)
print(f"{len(test)} test clips")
pd.crosstab(test["source"], test["label_name"])

# %% [markdown]
# ## 1. Pick the deployed model on validation

# %%
predictors = {a: pu.ShopliftingPredictor(a) for a in ("A", "B")}
val_scores = {a: (p.meta["val_f1"], p.meta["val_auc"]) for a, p in predictors.items()}
best = max(val_scores, key=lambda a: val_scores[a])
best_info = {"approach": best, "description": predictors[best].description,
             "val_f1": val_scores[best][0], "val_auc": val_scores[best][1],
             "chosen_by": "validation F1, then validation ROC-AUC"}
(pu.MODELS_DIR / "best_model.json").write_text(json.dumps(best_info, indent=2))
for a, p in predictors.items():
    print(f"{a}: {p.description}  (val F1 {val_scores[a][0]:.3f}, val AUC {val_scores[a][1]:.3f})")
print("Deployed:", best)

# %% [markdown]
# ## 2. Score the test clips

# %%
names = {a: p.description.split(": ", 1)[1] for a, p in predictors.items()}
for a, predictor in predictors.items():
    test[f"p_{a}"] = [predictor.predict_from_extraction(pu.load_extraction(c))["probability"] for c in test["cache_path"]]
    test[f"pred_{a}"] = (test[f"p_{a}"] >= 0.5).astype(int)
test.drop(columns=["video_path", "cache_path"]).to_csv(pu.RESULTS_DIR / "test_predictions.csv", index=False)


def metrics(y, p):
    yhat = (np.asarray(p) >= 0.5).astype(int)
    return {"n": len(y), "accuracy": accuracy_score(y, yhat),
            "precision": precision_score(y, yhat, zero_division=0), "recall": recall_score(y, yhat, zero_division=0),
            "f1": f1_score(y, yhat, zero_division=0),
            "roc_auc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan}


table = pd.DataFrame({f"{a}: {names[a]}": metrics(test["label"], test[f"p_{a}"]) for a in predictors}).T
table.to_csv(pu.RESULTS_DIR / "test_metrics.csv")
table.round(3)

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, a in zip(axes[:2], predictors):
    cm = confusion_matrix(test["label"], test[f"pred_{a}"], labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=["normal", "shoplifting"]).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Approach {a}: {names[a]}", fontsize=10)
for a in predictors:
    fpr, tpr, _ = roc_curve(test["label"], test[f"p_{a}"])
    axes[2].plot(fpr, tpr, label=f"Approach {a} (AUC {roc_auc_score(test['label'], test[f'p_{a}']):.2f})")
axes[2].plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
axes[2].set(title="ROC curve (test)", xlabel="false positive rate", ylabel="true positive rate")
axes[2].legend(loc="lower right")
plt.tight_layout()
plt.savefig(pu.FIGURES_DIR / "04_confusion_and_roc.png", dpi=150)
plt.show()

# %% [markdown]
# ## 3. Per-source check (the DCSASS confound)
#
# If a model were just recognising "DCSASS-looking" clips, it would score very well on DCSASS and poorly on staged clips. Note that the DCSASS test clips are **all shoplifting**, so only accuracy/recall are meaningful there (ROC-AUC needs both classes and shows as NaN).

# %%
by_source = []
for a in predictors:
    for src, g in test.groupby("source"):
        by_source.append({"approach": a, "source": src, **metrics(g["label"], g[f"p_{a}"])})
by_source = pd.DataFrame(by_source)
by_source.to_csv(pu.RESULTS_DIR / "test_metrics_by_source.csv", index=False)
by_source.round(3)

# %% [markdown]
# ## 4. Misclassified clips

# %%
wrong = test[(test["pred_A"] != test["label"]) | (test["pred_B"] != test["label"])]
wrong[["relative_path", "source", "label_name", "p_A", "p_B"]].round(3)

# %% [markdown]
# ## 5. Summary
#
# The tables above (also saved in `results/`) are the numbers for the report. The figures are in `figures/`.

# %%
print(f"Deployed model: Approach {best} ({names[best]}), chosen on validation F1")
for a in predictors:
    r = table.loc[f"{a}: {names[a]}"]
    print(f"Approach {a}: test accuracy {r['accuracy']:.3f}, F1 {r['f1']:.3f}, ROC-AUC {r['roc_auc']:.3f}")
```

- [ ] **Step 2: Build and execute**

```powershell
& "$PM\.venv\Scripts\python.exe" "$SCRATCH\build_nb.py" "$SCRATCH\nb_src\04_evaluation.py" "$PM\04_evaluation.ipynb"
& "$PM\.venv\Scripts\jupyter.exe" nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=pose-shoplifting "$PM\04_evaluation.ipynb"
```
Expected: exit 0; `models\best_model.json` exists; metric tables are printed.

---

### Task 9: Notebook 05 — predict, then full verification

**Files:**
- Create: `SCRATCH\nb_src\05_predict.py` → `PM\05_predict.ipynb`

**Interfaces:**
- Consumes: `ShopliftingPredictor("best")`, `draw_pose_frame`, `load_manifest`.

- [ ] **Step 1: Write the notebook source** — `SCRATCH\nb_src\05_predict.py`

```python
# %% [markdown]
# # 05 · Predict: video clip in → shoplifting / normal out
#
# Loads the deployed model (chosen on validation in notebook 04) and classifies any `.mp4`. Under the hood: YOLO11-pose keypoints → primary person → normalisation → model → **P(shoplifting)**, and a label using a 0.5 threshold.
#
# **To use on your own clip:** change `VIDEO` below to its path and run the cell.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pose_utils as pu

predictor = pu.ShopliftingPredictor("best")
print("Model:", predictor.description)

# %% [markdown]
# ## 1. One clip

# %%
manifest = pu.load_manifest()
test = manifest[manifest.split == "test"]
VIDEO = test[(test.label_name == "shoplifting") & (test.source == "staged")].iloc[0].video_path  # <- put your own path here

result = predictor.predict(VIDEO, keep_extraction=True)
print(f"{Path(VIDEO).name}: {result['label'].upper()}  (P(shoplifting) = {result['probability']:.2f})")
if result["no_person_detected"]:
    print("Warning: no person was detected, so this prediction is not meaningful.")


# %%
def show_prediction(result, n_frames=4):
    """A few frames with the detected skeletons and the prediction in the title."""
    ext = result["extraction"]
    ts = np.linspace(0, len(ext["kpts"]) - 1, n_frames).astype(int)
    fig, axes = plt.subplots(1, n_frames, figsize=(4.2 * n_frames, 3.6))
    for ax, t in zip(axes, ts):
        pu.draw_pose_frame(ax, result["video"], ext, t)
        ax.set_title(f"t = {t / ext['fps_sampled']:.1f}s", fontsize=9)
    fig.suptitle(f"{Path(result['video']).name}: {result['label'].upper()} (P(shoplifting) = {result['probability']:.2f})")
    plt.tight_layout()
    plt.show()


show_prediction(result)

# %%
VIDEO2 = test[(test.label_name == "normal") & (test.source == "staged")].iloc[0].video_path
result2 = predictor.predict(VIDEO2, keep_extraction=True)
show_prediction(result2)

# %% [markdown]
# ## 2. A whole folder
#
# `predict_folder` classifies every `.mp4` in a folder. The demo uses the 3 quarantined clips: exact duplicates of normal training clips, removed during cleaning. They only show that the function works; they are **not** a fair accuracy test.


# %%
def predict_folder(folder, pattern="*.mp4"):
    rows = [predictor.predict(p) for p in sorted(Path(folder).glob(pattern))]
    return pd.DataFrame(rows, columns=["video", "label", "probability", "approach", "no_person_detected"])


demo_folder = pu.DATA_DIR / "quarantine" / "exact_duplicates" / "videos" / "normal"
predict_folder(demo_folder).assign(video=lambda d: d["video"].map(lambda p: Path(p).name)).round(3)

# %% [markdown]
# ## 3. One-liner
#
# From any notebook or script in this folder:
# ```python
# import pose_utils as pu
# pu.predict_clip("path/to/clip.mp4")   # {'label': 'shoplifting', 'probability': 0.87, ...}
# ```

# %%
pu.predict_clip(VIDEO2)
```

- [ ] **Step 2: Build and execute**

```powershell
& "$PM\.venv\Scripts\python.exe" "$SCRATCH\build_nb.py" "$SCRATCH\nb_src\05_predict.py" "$PM\05_predict.ipynb"
& "$PM\.venv\Scripts\jupyter.exe" nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=pose-shoplifting "$PM\05_predict.ipynb"
```
Expected: exit 0.

- [ ] **Step 3: Full test suite (models now exist, so nothing is skipped)**

Run: `& "$PM\.venv\Scripts\python.exe" -m pytest "$PM\tests" -q`
Expected: all pass, 0 skipped.

- [ ] **Step 4: Check notebooks for errors and outputs**

```powershell
& "$PM\.venv\Scripts\python.exe" -c "import nbformat,glob,sys; bad=[(f,o.get('ename')) for f in glob.glob(r'$PM\0*.ipynb') for c in nbformat.read(f,4).cells if c.cell_type=='code' for o in c.get('outputs',[]) if o.get('output_type')=='error']; print(bad or 'no errors'); sys.exit(1 if bad else 0)"
Get-ChildItem "$PM\figures", "$PM\results", "$PM\models" | Select-Object Name, Length
```
Expected: `no errors`; the figures and results listed in the spec are present.

- [ ] **Step 5: Report actual metrics** from `results\test_metrics.csv` and `results\test_metrics_by_source.csv` to the user, whatever they are.
