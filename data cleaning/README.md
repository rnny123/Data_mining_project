# Unified Video Dataset Handoff

This folder combines staged normal/shoplifting clips with the DCSASS Shoplifting videos into one binary video dataset for model development.

## Recommended dataset to share

Use `cleaned_data/unified_video_deduplicated/` for model training. It contains 207 unique videos: 91 normal (label `0`) and 116 shoplifting (label `1`). Its fixed splits contain 144 training, 30 validation, and 33 test videos. The split seed is 42. Exact duplicate videos removed from the usable set are retained under `quarantine/exact_duplicates/` for audit and should not be used for training.

Share the **entire** `cleaned_data/unified_video_deduplicated/` folder, including `videos/`, `splits/`, `manifest.csv`, and `summary.json`. The CSVs contain paths, not the video content. For collaboration, place this folder in a shared drive or send it as an archive if the file size and dataset terms allow. The raw archives and cleaning script are not required to train a model.

Before redistributing video files, check the source datasets' terms. If video redistribution is restricted, share the cleaning script and split/manifest files only through an approved arrangement, and have each teammate obtain the source data independently.

## Rebuild the dataset

Requirements: Python 3.10 or newer and FFmpeg (including `ffmpeg` and `ffprobe` on `PATH`). No third-party Python packages are required.

Expected inputs relative to this folder:

```text
dataset/staged_clips/normal/*.mp4
dataset/staged_clips/shoplifting/*.mp4
dataset/dcsass/DCSASS Dataset/Shoplifting/<source>/<source>_<0-31>.mp4
dataset/dcsass/DCSASS Dataset/Labels/Shoplifting.csv
```

From the project root in PowerShell, run a dry run first. Its output folder must be a new path even though the dry run does not write files:

```powershell
Set-Location "data cleaning"
python clean_scripts/clean_unified_dataset.py --dry-run --output cleaned_data/unified_video_dry_run_check
```

The dry run checks expected folders, DCSASS labels, and segment naming/counts. It does not decode the videos. To build the dataset, choose another output path that does not already exist:

```powershell
python clean_scripts/clean_unified_dataset.py --output cleaned_data/unified_video_for_model
```

The full run decodes and probes videos, checks DCSASS segments, joins each source's 32 segments, quarantines invalid inputs, hashes videos to identify exact duplicates, creates fixed splits, and writes a summary. Without `--output`, it writes to `cleaned_data/unified_video_deduplicated/`, which is also the default input for the PoseLift handoff below. Existing output directories are never overwritten, so use a new `--output` path when rebuilding.

The script's default seed is `42`; use `--seed 42` to make that choice explicit. The output is created at the path passed to `--output`, relative to the current working directory if a relative path is used. The current script deduplicates exact output videos automatically, so a successful fresh build should have no repeated video hashes in `videos/`.

## Files and labels
## Create a PoseLift test-only handoff

To copy the cleaned unified video dataset and add PoseLift as a separate, cleaned test-only dataset, run from this folder:

```powershell
python clean_scripts/create_pose_dataset_copy.py --dry-run
python clean_scripts/create_pose_dataset_copy.py
```

The output is `cleaned_data/unified_video_poselift_test/`. The existing unified video files and their train/validation/test splits are copied unchanged. Those splits combine the staged clips and DCSASS videos; the two sources are split together, with exact duplicates kept out of cross-split leakage. PoseLift is built only from `dataset/poselift/Pickle_files/Test` and its matching `GT` labels; its training annotations are not copied. The separate `pose_lift_test/` folder contains pose samples, a test-only manifest and split, duplicate/quarantine reports, and a summary. Use it as an additional, independent test set and report its results separately from the unified video's `splits/test.csv`; do not merge the two test CSVs or use PoseLift samples for training or validation. Non-finite confidence values are normalized to `0`; invalid samples and exact duplicates are quarantined. Existing output folders are never overwritten.

## Files and labels

```text
cleaned_data/unified_video_deduplicated/
  videos/normal/             # label 0
  videos/shoplifting/        # label 1
  splits/train.csv
  splits/validation.csv
  splits/test.csv
  manifest.csv               # all usable videos and metadata
  summary.json               # counts and split configuration
  input_duplicates.csv       # duplicate hashes in source inputs
  video_duplicates.csv       # retained/quarantined exact output duplicates
  quarantine.csv             # reasons for quarantined files
  quarantine/                # quarantined files; exclude from modeling
```

Each split CSV is a subset of the manifest and has the same columns:

| Column | Meaning |
| --- | --- |
| `relative_path` | Video path relative to the dataset folder |
| `label` | Binary target: `0` normal, `1` shoplifting |
| `label_name` | Human-readable target name |
| `dataset`, `source_id` | Source dataset and source identifier |
| `sha256` | Exact-content hash; useful for duplicate checks |
| `video_codec`, `width`, `height`, `fps`, `duration_seconds` | Probed video metadata |
| `split` | `train`, `validation`, or `test` |

Resolve each path against the dataset folder, not the CSV's `splits/` directory. For example, for a dataset folder at `D:/shared/cleaned_unified_deduplicated`, join that folder with the row's `relative_path`.

## Modeling handoff notes

- Load `train.csv`, `validation.csv`, and `test.csv` separately. Do not train from `manifest.csv` as one pool; that would mix the held-out splits into training.
- Keep the provided split assignments fixed across experiments. Exact duplicates are grouped to prevent matching copies from crossing splits; the recommended deduplicated folder also removes redundant copies.
- Tune and select models using training and validation data. Reserve the test split for final evaluation.
- The classes are imbalanced (91 normal vs. 116 shoplifting in this dataset). Report per-class precision/recall or F1 and a confusion matrix, not accuracy alone.
- Source resolutions, frame rates, and durations vary. Apply consistent decoding, frame sampling, resizing, and normalization in the model pipeline; the cleaning step intentionally preserves source resolution.
- DCSASS is provided as stitched source videos. A stitched source is labeled shoplifting if any of its 32 labeled segments is positive; otherwise it is labeled normal.

## Current dataset snapshot

These counts describe the `cleaned_data/unified_video_deduplicated/` output. Re-running against changed source archives can produce different counts.

| Split | Videos |
| --- | ---: |
| Train | 144 |
| Validation | 30 |
| Test | 33 |
| Total | 207 |

Class counts: 91 normal, 116 shoplifting. Split seed: 42. Target ratios: 70% train, 15% validation, 15% test; integer counts may vary slightly because exact-duplicate groups stay together.