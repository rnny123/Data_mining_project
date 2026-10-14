# Pose-Based Shoplifting Detection — Design Spec

Date: 2026-10-08
Status: Approved in conversation; pending written-spec review

## 1. Goal

Given a video clip, output a binary prediction: **shoplifting** or **normal**, using
human pose (body keypoints) as the only signal. Built in Python as Jupyter notebooks
for the IS424 Data Mining final project. Standalone — no comparison with the earlier
ResNet18-LSTM baseline is required.

Success criteria:
- All notebooks execute end-to-end on the user's CPU-only Windows machine.
- A `predict_clip(path)` function returns label + probability for any `.mp4`.
- Test-set metrics, figures, and a per-source breakdown are produced for the report.

## 2. Inputs and Constraints

- Dataset root: `Project/cleaned_unified_deduplicated/`
  - `manifest.csv` columns: `relative_path, label, label_name, dataset, source_id, sha256,
    video_codec, width, height, fps, duration_seconds, split`
  - Fixed splits (seed 42): train 144, validation 30, test 33. Labels: 1 = shoplifting, 0 = normal.
  - Sources: `staged_normal` (87, 640×480, ~11 s), `staged_shoplifting` (92, 640×480, ~11 s),
    `dcsass_shoplifting` (28, 320×240, 32–160 s; 24 shoplifting / 4 normal). All 30 fps.
- Hardware: CPU only. Python 3.12, PyTorch 2.10 CPU. OpenCV and ultralytics to be installed.
- Known confound: DCSASS clips are mostly shoplifting and differ in resolution/length. Mitigated
  by per-person keypoint normalisation; measured via a per-source test breakdown. DCSASS clips are kept.

## 3. Output Layout

```
Project/pose_model/
  pose_utils.py              shared code (I/O, normalisation, features, windows, models)
  requirements.txt
  01_pose_extraction.ipynb
  02_features_ml.ipynb       Approach A
  03_sequence_model.ipynb    Approach B
  04_evaluation.ipynb
  05_predict.ipynb
  cache/keypoints/<label_name>/<stem>.npz
  models/                    approach_a.joblib, approach_b.pt, *_meta.json, best_model.json
  figures/                   PNGs for the report
  results/                   CSV metric tables, predictions
```

## 4. Components

### 4.1 Pose extraction (notebook 01, functions in `pose_utils.py`)
- Model: `yolo11n-pose.pt` (ultralytics), tracker ByteTrack, `persist=False` per video so
  track IDs reset between clips.
- Sampling: target 10 fps → `vid_stride = round(fps / 10)` (3 for all current clips).
- Per frame keep up to `MAX_PERSONS = 5` detections (highest box confidence first).
- Saved per clip (`.npz`): `kpts (T, P, 17, 3)` [x, y, conf in pixels], `boxes (T, P, 4)` xyxy,
  `track_ids (T, P)` (-1 = empty), `n_people (T,)`, `width`, `height`, `fps_sampled`, `stride`.
- Environment: a project-local virtualenv `Project/pose_model/.venv` (kernel "Python (pose-shoplifting)"),
  because installing ultralytics globally would upgrade the user's global torch and numpy.
- Resumable: skip clips whose `.npz` exists.
- Sanity outputs: skeleton overlay on sample frames; detection rate per source.

### 4.2 Primary person + normalisation (`pose_utils.py`)
- Primary person = track ID present in the most frames; tie-break by mean box area.
  Clips with no tracks → all-NaN sequence (flagged).
- Keypoints with conf < 0.3 → NaN. Interior NaN gaps linearly interpolated per coordinate
  (`np.interp`); leading/trailing NaNs → nearest valid value.
  The confidence channel keeps the original (low) values so models can see what was filled.
- Normalisation: subtract the body centre = mean of the hips that were seen at least once in the
  clip (fallback: seen shoulders, then any seen keypoint); divide by median torso length
  (mid-shoulder to mid-hip, frames where all four were seen); fallback 0.3 × median box height.
- Keypoints never seen in the whole clip → 0 in **normalised** coordinates and 0 velocity, applied
  after centring/scaling, so they carry no information about image position or resolution.
  (Revised 2026-10-14 after independent review: the first version zeroed them in pixel space before
  centring, which made them position-dependent.)
- `has_person` = a primary person was found **and** at least one of their keypoints was confident.
- Output: `seq (T, 17, 2)` normalised coords, `conf (T, 17)`, `n_people (T,)`.

### 4.3 Approach A — hand-crafted features + classic ML (notebook 02)
- Per-frame features: wrist→hip distances (L/R, same-side and mid-hip), wrist→nose distances,
  wrist↔wrist distance, wrist height relative to hip, elbow angles, knee angles, torso lean,
  crouch ratio, wrist speeds, body-centre speed, wrist visibility (conf ≥ 0.3), number of people.
- Clip aggregation per feature: mean, std, min, max, p10, p90. Plus one extra feature: the
  fraction of frames with a wrist within 0.5 torso-lengths of the hip.
- Clip length / duration is deliberately **not** a feature (it would leak the DCSASS confound).
- Models: LogisticRegression (StandardScaler, balanced), RandomForest (balanced),
  HistGradientBoosting. Hyperparameters via `GridSearchCV`, 5-fold stratified CV on train,
  scoring F1. Best model chosen by validation F1 (ties → ROC-AUC). Saved with `joblib`.
- Report figures: CV/val comparison bar chart, permutation feature importance (top 20).

### 4.4 Approach B — sequence model (notebook 03)
- Per-timestep input (85 dims): 34 normalised coords, 34 velocities (frame diff), 17 confidences.
- Windows: length 32 (3.2 s). Train: per epoch sample up to 8 random windows per video
  (balances long DCSASS clips). Eval: stride 8, clips shorter than 32 zero-padded with mask.
- Augmentation (train only): left/right mirror (swap COCO L/R pairs, negate x), Gaussian noise
  σ=0.02, scale jitter ×[0.9, 1.1].
- Model: BiGRU, 2 layers, hidden 64, dropout 0.3, masked mean-pool over time → Linear → logit.
- Training: AdamW lr 1e-3, weight decay 1e-3, BCEWithLogits with pos_weight from train balance,
  max 80 epochs, early stopping patience 15 on validation clip-level ROC-AUC; seed 42.
- Clip aggregation of window probabilities: mean, max, top-3 mean — chosen on validation.
- Report figures: training curves, validation aggregation comparison.

### 4.5 Evaluation (notebook 04)
- Test set used only here. Threshold 0.5 for both approaches.
- Metrics: accuracy, precision, recall, F1, ROC-AUC. Confusion matrices, ROC curves,
  A vs B table, per-source (staged vs DCSASS) metrics, misclassified-clip list.
- Honest-reporting additions (2026-10-14): metrics on subsets (without empty-room clips, staged only,
  staged with a person visible — the headline estimate free of the source confound); bootstrap 95% CIs
  and McNemar's exact test for A vs B; a source-predictability diagnostic plus predictions on validation
  DCSASS normal clips; a note that validation drove three decisions.
- Writes `models/best_model.json` = approach with higher validation F1 (decided from validation,
  not test). Saves tables to `results/` and figures to `figures/`.

### 4.6 Prediction (notebook 05)
- `pose_utils.predict_clip(path, approach="best")` (wrapping `ShopliftingPredictor`) →
  `{"label": "shoplifting"|"normal", "probability": p, "approach": ..., "no_person_detected": bool}`.
  `probability` is P(shoplifting). Runs extraction → primary person → normalisation → model.
  Notebook 04 also scores the test set through `ShopliftingPredictor`, so evaluation and
  prediction share one code path.
- Batch mode over a folder; optional skeleton-overlay preview frames.

## 5. Error Handling

- Unreadable video / zero frames → extraction records the failure and continues; notebook prints a list.
- No person detected in a clip → features all zero/NaN-filled; prediction still returned but
  flagged `"no_person_detected": True`.
- Missing cache/model files → clear error telling the user which notebook to run first.

## 6. Verification

- Install `ultralytics` (pulls OpenCV). Smoke-test extraction on 2 clips, then full run.
- Execute every notebook end-to-end with `jupyter nbconvert --execute` so outputs are embedded.
- `predict_clip` checked on one known shoplifting and one known normal test clip.
- Report actual numbers honestly, whatever they are.

## 7. Out of Scope

- ST-GCN or other graph models; fine-tuning the pose model; GPU support; real-time streaming;
  multiple-instance learning across all tracks (noted as future work).
