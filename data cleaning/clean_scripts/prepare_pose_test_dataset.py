from __future__ import annotations

import argparse
import csv
import json
import hashlib
import pickle
import re
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


MANIFEST_FIELDS = [
    "relative_path",
    "label",
    "label_name",
    "dataset",
    "source_id",
    "camera_id",
    "sha256",
    "frame_count",
    "person_count",
    "frames_with_pose",
    "split",
]


class DatasetError(Exception):
    pass


def normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def get_value(record: Mapping[Any, Any], names: set[str]) -> Any:
    for key, value in record.items():
        if normalized_key(key) in names:
            return value
    return None


def is_keypoint_array(value: Any) -> bool:
    try:
        points = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return points.shape in {(17, 2), (17, 3)} or points.size in {34, 51}


def person_records(frame: Any) -> list[tuple[str, Any]]:
    if isinstance(frame, Mapping):
        keypoints = get_value(frame, {"keypoints", "keypoint", "pose"})
        if keypoints is not None:
            person_id = get_value(frame, {"personid", "person", "trackid", "id"})
            return [(str(person_id) if person_id is not None else "person_0", keypoints)]

        people: list[tuple[str, Any]] = []
        for key, value in frame.items():
            if isinstance(value, Mapping):
                nested = person_records(value)
                if nested:
                    fallback_id = str(key)
                    people.extend(
                        (fallback_id if person_id == "person_0" else person_id, points)
                        for person_id, points in nested
                    )
            elif normalized_key(key) in {"people", "persons", "annotations", "instances"}:
                people.extend(person_records(value))
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                if is_keypoint_array(value):
                    people.append((str(key), value))
                elif len(value) >= 2 and is_keypoint_array(value[1]):
                    people.append((str(key), value[1]))
        return people

    if isinstance(frame, Sequence) and not isinstance(frame, (str, bytes)):
        people = []
        for index, value in enumerate(frame):
            if isinstance(value, Mapping):
                nested = person_records(value)
                people.extend(
                    (f"person_{index}" if person_id == "person_0" else person_id, points)
                    for person_id, points in nested
                )
        return people

    return []


def as_keypoints(value: Any, path: Path, frame_key: Any) -> np.ndarray:
    points = np.asarray(value, dtype=np.float32)
    if points.shape == (17, 2):
        points = np.column_stack((points, np.ones(17, dtype=np.float32)))
    elif points.size == 51:
        points = points.reshape(17, 3)
    elif points.size == 34:
        points = points.reshape(17, 2)
        points = np.column_stack((points, np.ones(17, dtype=np.float32)))
    if points.shape != (17, 3):
        raise DatasetError(
            f"Expected 17 keypoints with x, y, confidence in {path.name} "
            f"frame {frame_key}; found shape {points.shape}"
        )
    if not np.isfinite(points[:, :2]).all():
        raise DatasetError(f"Non-finite x/y keypoint values in {path.name} frame {frame_key}")
    if not np.isfinite(points[:, 2]).all():
        points = points.copy()
        points[:, 2] = np.nan_to_num(points[:, 2], nan=0.0, posinf=0.0, neginf=0.0)
    return points


def frame_items(annotation: Any, path: Path) -> list[tuple[int, Any]]:
    if not isinstance(annotation, Mapping):
        raise DatasetError(f"Expected frame-number dictionary in {path.name}")
    items: list[tuple[int, Any]] = []
    for key, value in annotation.items():
        try:
            frame_index = int(key)
        except (TypeError, ValueError) as error:
            raise DatasetError(f"Non-numeric frame key {key!r} in {path.name}") from error
        items.append((frame_index, value))
    if not items:
        raise DatasetError(f"No frames found in {path.name}")
    items.sort(key=lambda item: item[0])
    if len({index for index, _ in items}) != len(items):
        raise DatasetError(f"Duplicate frame numbers in {path.name}")
    return items


def align_frame_labels(frame_indices: list[int], labels: np.ndarray, path: Path) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1)
    if labels.size == len(frame_indices):
        aligned = labels
    elif all(0 <= index < labels.size for index in frame_indices):
        aligned = labels[frame_indices]
    elif all(1 <= index <= labels.size for index in frame_indices):
        aligned = labels[np.asarray(frame_indices) - 1]
    else:
        raise DatasetError(
            f"Cannot align {labels.size} frame labels with "
            f"{len(frame_indices)} annotated frames in {path.name}"
        )
    try:
        numeric = aligned.astype(np.int64)
    except (TypeError, ValueError) as error:
        raise DatasetError(f"Non-integer frame labels in {path.name}") from error
    if not np.isin(numeric, (0, 1)).all():
        raise DatasetError(f"Frame labels must be binary (0 or 1) in {path.name}")
    return numeric.astype(np.uint8)


def load_pose_sample(annotation_path: Path, label_path: Path) -> dict[str, Any]:
    try:
        with annotation_path.open("rb") as annotation_file:
            annotation = pickle.load(annotation_file)
    except (OSError, pickle.PickleError, EOFError) as error:
        raise DatasetError(f"Could not read {annotation_path.name}: {error}") from error

    try:
        frame_labels = np.load(label_path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise DatasetError(f"Could not read {label_path.name}: {error}") from error

    items = frame_items(annotation, annotation_path)
    frame_indices = [index for index, _ in items]
    labels = align_frame_labels(frame_indices, frame_labels, label_path)

    people_by_frame: list[list[tuple[str, np.ndarray]]] = []
    person_ids: set[str] = set()
    for frame_index, frame in items:
        people = [
            (person_id, as_keypoints(points, annotation_path, frame_index))
            for person_id, points in person_records(frame)
        ]
        people_by_frame.append(people)
        person_ids.update(person_id for person_id, _ in people)

    ordered_person_ids = sorted(person_ids)
    person_positions = {person_id: index for index, person_id in enumerate(ordered_person_ids)}
    keypoints = np.zeros((len(items), len(ordered_person_ids), 17, 3), dtype=np.float32)
    person_mask = np.zeros((len(items), len(ordered_person_ids)), dtype=np.bool_)
    for frame_position, people in enumerate(people_by_frame):
        for person_id, points in people:
            person_position = person_positions[person_id]
            keypoints[frame_position, person_position] = points
            person_mask[frame_position, person_position] = True

    return {
        "keypoints": keypoints,
        "person_ids": np.asarray(ordered_person_ids, dtype=np.str_),
        "person_mask": person_mask,
        "frame_labels": labels,
        "frame_indices": np.asarray(frame_indices, dtype=np.int64),
        "label": np.uint8(labels.max()),
    }


def discover_pairs(input_root: Path, labels_root: Path) -> list[tuple[Path, Path]]:
    annotation_files = sorted(input_root.glob("*.pkl"))
    label_files = sorted(labels_root.glob("*.npy"))
    annotations_by_stem = {path.stem: path for path in annotation_files}
    labels_by_stem = {path.stem: path for path in label_files}
    if len(annotations_by_stem) != len(annotation_files):
        raise DatasetError("Multiple .pkl files share a filename; use unique PoseLift filenames")
    if len(labels_by_stem) != len(label_files):
        raise DatasetError("Multiple .npy files share a filename; use unique PoseLift filenames")
    if not annotation_files or not label_files:
        raise DatasetError(
            f"No PoseLift annotations or labels found in {input_root} and {labels_root}"
        )

    missing_labels = sorted(set(annotations_by_stem) - set(labels_by_stem))
    missing_annotations = sorted(set(labels_by_stem) - set(annotations_by_stem))
    if missing_labels or missing_annotations:
        details = []
        if missing_labels:
            details.append(f"missing .npy labels for: {', '.join(missing_labels[:5])}")
        if missing_annotations:
            details.append(f"missing .pkl annotations for: {', '.join(missing_annotations[:5])}")
        raise DatasetError("Unmatched PoseLift files (" + "; ".join(details) + ")")
    return [
        (annotations_by_stem[stem], labels_by_stem[stem])
        for stem in sorted(annotations_by_stem)
    ]


def write_csv(
    path: Path, rows: list[dict[str, Any]], fields: list[str] = MANIFEST_FIELDS
) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sample_sha256(sample: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for field in (
        "keypoints",
        "person_ids",
        "person_mask",
        "frame_labels",
        "frame_indices",
        "label",
    ):
        array = np.ascontiguousarray(sample[field])
        digest.update(field.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def prepare_dataset(args: argparse.Namespace) -> int:
    input_root = args.input.resolve()
    labels_root = args.labels.resolve()
    output_root = args.output.resolve()
    if not input_root.is_dir():
        raise DatasetError(f"Input folder does not exist: {input_root}")
    if not labels_root.is_dir():
        raise DatasetError(f"Label folder does not exist: {labels_root}")
    if (
        output_root == input_root
        or output_root in input_root.parents
        or input_root in output_root.parents
    ):
        raise DatasetError("Output folder must not overlap the PoseLift input folder")
    if output_root.exists():
        raise DatasetError(f"Output already exists: {output_root}; choose a new folder")

    pairs = discover_pairs(input_root, labels_root)
    if args.dry_run:
        labels_count = {"normal": 0, "shoplifting": 0}
        sample_hashes: list[str] = []
        invalid_count = 0
        for annotation_path, label_path in pairs:
            try:
                sample = load_pose_sample(annotation_path, label_path)
            except (DatasetError, OSError, ValueError):
                invalid_count += 1
                continue
            label_name = "shoplifting" if sample["label"] else "normal"
            labels_count[label_name] += 1
            sample_hashes.append(sample_sha256(sample))
        duplicate_count = len(sample_hashes) - len(set(sample_hashes))
        print(f"Paired PoseLift samples: {len(pairs)}")
        print(f"Valid samples: {len(sample_hashes)}; invalid samples to quarantine: {invalid_count}")
        print(f"Exact duplicate samples to quarantine: {duplicate_count}")
        print(
            "Clip labels: "
            f"normal={labels_count['normal']}, shoplifting={labels_count['shoplifting']}"
        )
        print(f"Output would be: {output_root}")
        print("Test-only dry run; no files were written.")
        return 0

    output_root.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    sample_records: list[dict[str, Any]] = []
    quarantine_rows: list[dict[str, str]] = []
    try:
        for annotation_path, label_path in pairs:
            source_id = annotation_path.stem
            try:
                sample = load_pose_sample(annotation_path, label_path)
            except (DatasetError, OSError, ValueError) as error:
                for source_path in (annotation_path, label_path):
                    quarantine_path = (
                        work_dir / "quarantine" / "invalid" / source_id / source_path.name
                    )
                    quarantine_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_path, quarantine_path)
                    quarantine_rows.append(
                        {
                            "quarantined_path": quarantine_path.relative_to(work_dir).as_posix(),
                            "dataset": "PoseLift",
                            "source_id": source_id,
                            "reason": str(error),
                        }
                    )
                continue

            label = int(sample["label"])
            label_name = "shoplifting" if label == 1 else "normal"
            relative_path = Path("samples") / label_name / f"{source_id}.npz"
            sample_records.append(
                {
                    "relative_path": relative_path.as_posix(),
                    "label": label,
                    "label_name": label_name,
                    "dataset": "PoseLift",
                    "source_id": source_id,
                    "camera_id": source_id.split("_", 1)[0],
                    "sha256": sample_sha256(sample),
                    "frame_count": len(sample["frame_labels"]),
                    "person_count": len(sample["person_ids"]),
                    "frames_with_pose": int(sample["person_mask"].any(axis=1).sum()),
                    "split": "test",
                    "sample": sample,
                }
            )

        if not sample_records:
            raise DatasetError("No valid PoseLift samples remain after validation")

        sample_groups: dict[str, list[dict[str, Any]]] = {}
        for record in sample_records:
            sample_groups.setdefault(record["sha256"], []).append(record)

        rows: list[dict[str, Any]] = []
        duplicate_rows: list[dict[str, str]] = []
        label_counts = {"normal": 0, "shoplifting": 0}
        total_frames = 0
        total_people = 0
        for digest, group in sample_groups.items():
            if len({record["label"] for record in group}) != 1:
                raise DatasetError(f"Exact duplicate PoseLift samples have conflicting labels: {digest}")
            group.sort(key=lambda record: record["source_id"])
            canonical = group[0]
            for record in group:
                action = "retained"
                if record is not canonical:
                    quarantine_path = (
                        work_dir / "quarantine" / "exact_duplicates" / record["relative_path"]
                    )
                    quarantine_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(quarantine_path, **record["sample"])
                    action = "quarantined_duplicate"
                    quarantine_rows.append(
                        {
                            "quarantined_path": quarantine_path.relative_to(work_dir).as_posix(),
                            "dataset": "PoseLift",
                            "source_id": record["source_id"],
                            "reason": f"Exact duplicate of {canonical['source_id']}",
                        }
                    )
                else:
                    sample_path = work_dir / record["relative_path"]
                    sample_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(sample_path, **record["sample"])
                    rows.append({key: value for key, value in record.items() if key != "sample"})
                    label_counts[record["label_name"]] += 1
                    total_frames += record["frame_count"]
                    total_people += record["person_count"]
                duplicate_rows.append(
                    {
                        "sha256": digest,
                        "path": record["relative_path"],
                        "source_id": record["source_id"],
                        "retained_path": canonical["relative_path"],
                        "retained_source_id": canonical["source_id"],
                        "action": action,
                    }
                )

        rows.sort(key=lambda row: row["source_id"])
        write_csv(work_dir / "manifest.csv", rows)
        split_dir = work_dir / "splits"
        split_dir.mkdir(parents=True, exist_ok=True)
        write_csv(split_dir / "test.csv", rows)
        write_csv(
            work_dir / "quarantine.csv",
            quarantine_rows,
            ["quarantined_path", "dataset", "source_id", "reason"],
        )
        write_csv(
            work_dir / "pose_duplicates.csv",
            duplicate_rows,
            ["sha256", "path", "source_id", "retained_path", "retained_source_id", "action"],
        )

        summary = {
            "dataset": "PoseLift",
            "role": "external_test_only",
            "samples_written": len(rows),
            "clip_labels": label_counts,
            "split_counts": {"test": len(rows)},
            "frames": total_frames,
            "person_tracks": total_people,
            "quarantined_files": len(quarantine_rows),
            "invalid_files_quarantined": sum(
                not row["reason"].startswith("Exact duplicate of") for row in quarantine_rows
            ),
            "duplicate_samples_quarantined": sum(
                row["action"] == "quarantined_duplicate" for row in duplicate_rows
            ),
            "keypoint_format": "17 joints per person; x, y, confidence",
            "non_finite_confidence_handling": "replaced with 0",
            "clip_label_rule": "shoplifting if any frame label is 1; otherwise normal",
            "training_or_validation_data_created": False,
        }
        (work_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        work_dir.rename(output_root)
    except Exception:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise

    print(f"Created cleaned PoseLift test dataset: {output_root}")
    print(
        f"Samples: {len(rows)}; quarantined files: {len(quarantine_rows)}; "
        f"labels: {label_counts}"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare PoseLift pose sequences as an external test-only dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="PoseLift test annotation folder, such as Pickle_files/Test.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="PoseLift frame-label folder, such as Pickle_files/GT.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output folder; existing folders are never overwritten.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and count input samples without writing output files.",
    )
    return parser.parse_args()


def main() -> int:
    try:
        return prepare_dataset(parse_args())
    except (DatasetError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())