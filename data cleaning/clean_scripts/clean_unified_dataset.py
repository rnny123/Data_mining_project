from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any


CSV_FIELDS = [
    "relative_path",
    "label",
    "label_name",
    "dataset",
    "source_id",
    "sha256",
    "video_codec",
    "width",
    "height",
    "fps",
    "duration_seconds",
    "split",
]
SEGMENT_PATTERN = re.compile(r"(?P<source>.+)_(?P<index>\d+)$")


class DatasetError(Exception):
    pass


@dataclass
class DcsassSource:
    source_id: str
    files: list[Path]
    labels: list[str]

    @property
    def label(self) -> str:
        return "1" if "1" in self.labels else "0"


def read_dcsass_sources(shoplifting_dir: Path, labels_path: Path) -> list[DcsassSource]:
    labels: dict[str, str] = {}
    with labels_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.reader(csv_file)
        for line_number, row in enumerate(reader, start=1):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) != 3:
                raise DatasetError(
                    f"{labels_path.name}:{line_number} must have exactly 3 columns"
                )
            segment_name, category, label = (value.strip() for value in row)
            if category != "Shoplifting" or label not in {"0", "1"}:
                raise DatasetError(
                    f"Unexpected category or binary label at {labels_path.name}:{line_number}"
                )
            segment_key = Path(segment_name).stem
            if segment_key in labels:
                raise DatasetError(f"Duplicate label entry for {segment_key}")
            labels[segment_key] = label

    source_dirs = sorted(path for path in shoplifting_dir.iterdir() if path.is_dir())
    if not source_dirs:
        raise DatasetError(f"No DCSASS source folders found in {shoplifting_dir}")

    sources: list[DcsassSource] = []
    found_keys: set[str] = set()
    for source_dir in source_dirs:
        files = sorted(
            source_dir.glob("*.mp4"),
            key=lambda path: int(SEGMENT_PATTERN.fullmatch(path.stem)["index"])
            if SEGMENT_PATTERN.fullmatch(path.stem)
            else -1,
        )
        if not files:
            continue

        indexed_files: list[tuple[int, Path]] = []
        source_labels: list[str] = []
        for path in files:
            match = SEGMENT_PATTERN.fullmatch(path.stem)
            if match is None or match["source"] != source_dir.stem:
                raise DatasetError(f"Unexpected DCSASS segment name: {path}")
            segment_index = int(match["index"])
            if path.stem not in labels:
                raise DatasetError(f"Missing DCSASS label for {path.stem}")
            indexed_files.append((segment_index, path))
            source_labels.append(labels[path.stem])
            found_keys.add(path.stem)

        indexed_files.sort(key=lambda item: item[0])
        indices = [index for index, _ in indexed_files]
        if len(indices) != 32 or indices != list(range(32)):
            raise DatasetError(
                f"Expected segments 0 through 31 in {source_dir.name}; found {indices}"
            )

        sources.append(
            DcsassSource(
                source_id=source_dir.stem,
                files=[path for _, path in indexed_files],
                labels=source_labels,
            )
        )

    missing_files = sorted(set(labels) - found_keys)
    if missing_files:
        example = ", ".join(missing_files[:5])
        raise DatasetError(
            f"{len(missing_files)} label entries have no selected video segment: {example}"
        )
    return sources


def ffmpeg_decode_error(ffmpeg: str, path: Path) -> str | None:
    result = subprocess.run(
        [ffmpeg, "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return None
    detail = (result.stderr or "Video decoder returned an error").strip()
    return detail[-600:]


def probe_video(ffprobe: str, path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,r_frame_rate",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise DatasetError(f"Could not inspect {path.name}: {result.stderr.strip()[-600:]}")

    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise DatasetError(f"No video stream found in {path.name}")
    stream = streams[0]
    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/0"
    try:
        fps = float(Fraction(rate))
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    duration = float(data.get("format", {}).get("duration", 0.0) or 0.0)
    if duration <= 0 or int(stream.get("width", 0)) <= 0 or int(stream.get("height", 0)) <= 0:
        raise DatasetError(f"Invalid video metadata in {path.name}")
    return {
        "video_codec": stream.get("codec_name", ""),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": round(fps, 4),
        "duration_seconds": round(duration, 4),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as video_file:
        for chunk in iter(lambda: video_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ffconcat_path(path: Path) -> str:
    value = path.resolve().as_posix().replace("'", "'\\''")
    return f"file '{value}'"


def split_records(records: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    randomizer = random.Random(seed)
    hash_labels: dict[str, str] = {}
    for record in records:
        digest = record["sha256"]
        previous_label = hash_labels.setdefault(digest, record["label"])
        if previous_label != record["label"]:
            raise DatasetError(f"Exact duplicate video has conflicting labels: {digest}")

    for label in ("0", "1"):
        label_records = [record for record in records if record["label"] == label]
        groups: dict[str, list[dict[str, Any]]] = {}
        for record in label_records:
            groups.setdefault(record["sha256"], []).append(record)
        grouped_records = list(groups.values())
        randomizer.shuffle(grouped_records)
        total = len(label_records)
        targets = {
            "train": math.floor(total * 0.70),
            "validation": math.floor(total * 0.15),
        }
        targets["test"] = total - targets["train"] - targets["validation"]
        counts = {name: 0 for name in splits}
        for group in grouped_records:
            split_name = max(splits, key=lambda name: targets[name] - counts[name])
            for record in group:
                record["split"] = split_name
            splits[split_name].extend(group)
            counts[split_name] += len(group)
    return splits


def duplicate_hash_rows(items: list[dict[str, str]]) -> list[dict[str, str]]:
    by_hash: dict[str, list[dict[str, str]]] = {}
    for item in items:
        by_hash.setdefault(item["sha256"], []).append(item)
    return [
        {"sha256": digest, **item}
        for digest, group in by_hash.items()
        if len(group) > 1
        for item in group
    ]


def deduplicate_records(
    records: list[dict[str, Any]],
    work_dir: Path,
    quarantine_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record["sha256"], []).append(record)

    retained_records: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, str]] = []
    for digest, group in groups.items():
        labels = {record["label"] for record in group}
        if len(labels) != 1:
            raise DatasetError(f"Exact duplicate video has conflicting labels: {digest}")
        group.sort(key=lambda record: (record["dataset"], record["source_id"]))
        canonical = group[0]
        retained_records.append(canonical)
        for record in group:
            action = "retained"
            if record is not canonical:
                original_path = work_dir / record["relative_path"]
                quarantine_path = (
                    work_dir / "quarantine" / "exact_duplicates" / record["relative_path"]
                )
                quarantine_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(original_path), str(quarantine_path))
                action = "quarantined_duplicate"
                quarantine_rows.append(
                    {
                        "quarantined_path": quarantine_path.relative_to(work_dir).as_posix(),
                        "dataset": record["dataset"],
                        "source_id": record["source_id"],
                        "reason": f"Exact duplicate of {canonical['source_id']}",
                    }
                )
            duplicate_rows.append(
                {
                    "sha256": digest,
                    "path": record["relative_path"],
                    "dataset": record["dataset"],
                    "source_id": record["source_id"],
                    "retained_path": canonical["relative_path"],
                    "retained_source_id": canonical["source_id"],
                    "action": action,
                }
            )
    return retained_records, duplicate_rows


def copy_to_quarantine(
    files: list[Path],
    work_dir: Path,
    quarantine_rows: list[dict[str, str]],
    dataset: str,
    source_id: str,
    reason: str,
) -> None:
    for path in files:
        destination = work_dir / "quarantine" / dataset / source_id / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        quarantine_rows.append(
            {
                "quarantined_path": destination.relative_to(work_dir).as_posix(),
                "dataset": dataset,
                "source_id": source_id,
                "reason": reason,
            }
        )


def make_record(
    output_path: Path,
    work_dir: Path,
    label: str,
    dataset: str,
    source_id: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "relative_path": output_path.relative_to(work_dir).as_posix(),
        "label": label,
        "label_name": "shoplifting" if label == "1" else "normal",
        "dataset": dataset,
        "source_id": source_id,
        **metadata,
        "split": "",
    }


def clean_dataset(args: argparse.Namespace) -> int:
    data_root = args.data_root.resolve()
    staged_root = data_root / "staged_clips"
    dcsass_root = data_root / "dcsass" / "DCSASS Dataset"
    staged_classes = {"normal": "0", "shoplifting": "1"}
    dcsass_shoplifting = dcsass_root / "Shoplifting"
    dcsass_labels = dcsass_root / "Labels" / "Shoplifting.csv"
    output_root = (
        args.output.resolve()
        if args.output
        else data_root.parent / "cleaned_data" / "unified_video_deduplicated"
    )

    required_paths = [
        staged_root / "normal",
        staged_root / "shoplifting",
        dcsass_shoplifting,
        dcsass_labels,
    ]
    for path in required_paths:
        if not path.exists():
            raise DatasetError(f"Required input does not exist: {path}")

    for source_root in (staged_root, dcsass_root):
        if (
            output_root == source_root
            or source_root in output_root.parents
            or output_root in source_root.parents
        ):
            raise DatasetError("Output folder must not overlap either source dataset")
    if output_root.exists():
        raise DatasetError(
            f"Output already exists: {output_root}. Choose a new --output folder; existing files are never overwritten."
        )

    sources = read_dcsass_sources(dcsass_shoplifting, dcsass_labels)
    staged_files = {
        class_name: sorted((staged_root / class_name).glob("*.mp4"))
        for class_name in staged_classes
    }
    if any(not paths for paths in staged_files.values()):
        raise DatasetError("Both staged normal and shoplifting folders must contain MP4 clips")

    if args.dry_run:
        staged_total = sum(len(paths) for paths in staged_files.values())
        dcsass_counts = {
            label: sum(source.label == label for source in sources) for label in ("0", "1")
        }
        combined_counts = {
            "0": len(staged_files["normal"]) + dcsass_counts["0"],
            "1": len(staged_files["shoplifting"]) + dcsass_counts["1"],
        }
        print(f"Staged clips: {staged_total}")
        print(
            f"DCSASS source videos to stitch: {len(sources)} "
            f"({sum(len(source.files) for source in sources)} segments)"
        )
        print(f"Estimated unified videos: {sum(combined_counts.values())}")
        print(f"Estimated labels: normal={combined_counts['0']}, shoplifting={combined_counts['1']}")
        print(f"Output would be: {output_root}")
        print("Dry run only; no files were written and media decoding was not checked.")
        return 0

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        missing = [name for name, value in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe)) if not value]
        raise DatasetError(
            f"Missing required tool(s): {', '.join(missing)}. Install FFmpeg (which includes ffprobe), then rerun."
        )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent))
    records: list[dict[str, Any]] = []
    quarantine_rows: list[dict[str, str]] = []
    input_fingerprints: list[dict[str, str]] = []
    try:
        valid_staged: list[tuple[str, str, Path, str, dict[str, Any], str]] = []
        for class_name, label in staged_classes.items():
            dataset_name = f"staged_{class_name}"
            for source_path in staged_files[class_name]:
                source_id = f"staged/{class_name}/{source_path.stem}"
                digest = sha256_file(source_path)
                input_fingerprints.append(
                    {
                        "sha256": digest,
                        "path": source_path.relative_to(data_root).as_posix(),
                        "dataset": dataset_name,
                        "source_id": source_id,
                    }
                )
                error = ffmpeg_decode_error(ffmpeg, source_path)
                if error:
                    copy_to_quarantine(
                        [source_path], work_dir, quarantine_rows, dataset_name, source_id, error
                    )
                    continue
                try:
                    metadata = probe_video(ffprobe, source_path)
                except (DatasetError, ValueError, json.JSONDecodeError) as error:
                    copy_to_quarantine(
                        [source_path], work_dir, quarantine_rows, dataset_name, source_id, str(error)
                    )
                    continue
                valid_staged.append((class_name, label, source_path, source_id, metadata, digest))

        valid_sources: list[tuple[DcsassSource, dict[str, Any]]] = []
        for source in sources:
            segment_errors: list[str] = []
            segment_metadata: list[dict[str, Any]] = []
            for path in source.files:
                digest = sha256_file(path)
                input_fingerprints.append(
                    {
                        "sha256": digest,
                        "path": path.relative_to(data_root).as_posix(),
                        "dataset": "dcsass_shoplifting",
                        "source_id": f"dcsass/{source.source_id}/{path.stem}",
                    }
                )
                error = ffmpeg_decode_error(ffmpeg, path)
                if error:
                    segment_errors.append(f"{path.name}: {error}")
                    continue
                try:
                    segment_metadata.append(probe_video(ffprobe, path))
                except (DatasetError, ValueError, json.JSONDecodeError) as error:
                    segment_errors.append(f"{path.name}: {error}")

            if segment_errors:
                copy_to_quarantine(
                    source.files,
                    work_dir,
                    quarantine_rows,
                    "dcsass_shoplifting",
                    source.source_id,
                    "Invalid source segment(s): " + " | ".join(segment_errors[:3]),
                )
                continue

            reference_metadata = segment_metadata[0]
            inconsistent_segments = [
                path.name
                for path, metadata in zip(source.files, segment_metadata)
                if any(
                    metadata[key] != reference_metadata[key]
                    for key in ("video_codec", "width", "height", "fps")
                )
            ]
            if inconsistent_segments:
                copy_to_quarantine(
                    source.files,
                    work_dir,
                    quarantine_rows,
                    "dcsass_shoplifting",
                    source.source_id,
                    "Source segments have inconsistent codec or video properties: "
                    + ", ".join(inconsistent_segments[:5]),
                )
                continue
            valid_sources.append((source, reference_metadata))

        input_duplicate_rows = duplicate_hash_rows(input_fingerprints)

        for class_name, label, source_path, source_id, metadata, digest in valid_staged:
            output_path = work_dir / "videos" / class_name / source_path.name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, output_path)
            record = make_record(
                output_path,
                work_dir,
                label,
                f"staged_{class_name}",
                source_id,
                metadata,
            )
            record["sha256"] = digest
            records.append(record)

        for source, _ in valid_sources:
            label_name = "shoplifting" if source.label == "1" else "normal"
            output_path = work_dir / "videos" / label_name / f"{source.source_id}.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            list_path = work_dir / f"{source.source_id}.ffconcat.txt"
            list_path.write_text(
                "\n".join(ffconcat_path(path) for path in source.files) + "\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(list_path),
                    "-c",
                    "copy",
                    "-movflags",
                    "+faststart",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            list_path.unlink(missing_ok=True)
            if result.returncode != 0:
                output_path.unlink(missing_ok=True)
                copy_to_quarantine(
                    source.files,
                    work_dir,
                    quarantine_rows,
                    "dcsass_shoplifting",
                    source.source_id,
                    "Could not stitch source video: "
                    + (result.stderr or "unknown FFmpeg error").strip()[-600:],
                )
                continue

            output_error = ffmpeg_decode_error(ffmpeg, output_path)
            try:
                metadata = probe_video(ffprobe, output_path)
            except (DatasetError, ValueError, json.JSONDecodeError) as error:
                output_error = str(error)
                metadata = {}
            if output_error:
                output_path.unlink(missing_ok=True)
                copy_to_quarantine(
                    source.files,
                    work_dir,
                    quarantine_rows,
                    "dcsass_shoplifting",
                    source.source_id,
                    "Stitched video failed validation: " + output_error,
                )
                continue

            record = make_record(
                output_path,
                work_dir,
                source.label,
                "dcsass_shoplifting",
                f"dcsass/{source.source_id}",
                metadata,
            )
            record["sha256"] = sha256_file(output_path)
            records.append(record)

        if not records:
            raise DatasetError("No valid videos remain after validation")

        records, output_duplicate_rows = deduplicate_records(
            records, work_dir, quarantine_rows
        )
        splits = split_records(records, args.seed)
        records.sort(key=lambda row: (row["dataset"], row["source_id"]))
        write_csv(work_dir / "manifest.csv", records, CSV_FIELDS)
        split_dir = work_dir / "splits"
        split_dir.mkdir(parents=True, exist_ok=True)
        for split_name, split_rows in splits.items():
            split_rows.sort(key=lambda row: (row["dataset"], row["source_id"]))
            write_csv(split_dir / f"{split_name}.csv", split_rows, CSV_FIELDS)
        write_csv(
            work_dir / "quarantine.csv",
            quarantine_rows,
            ["quarantined_path", "dataset", "source_id", "reason"],
        )
        duplicate_fields = [
            "sha256",
            "path",
            "dataset",
            "source_id",
            "retained_path",
            "retained_source_id",
            "action",
        ]
        write_csv(
            work_dir / "input_duplicates.csv",
            input_duplicate_rows,
            ["sha256", "path", "dataset", "source_id"],
        )
        write_csv(work_dir / "video_duplicates.csv", output_duplicate_rows, duplicate_fields)

        summary = {
            "videos_written": len(records),
            "labels": {
                label_name: sum(record["label"] == label for record in records)
                for label, label_name in (("0", "normal"), ("1", "shoplifting"))
            },
            "split_counts": {name: len(rows) for name, rows in splits.items()},
            "quarantined_files": len(quarantine_rows),
            "invalid_files_quarantined": sum(
                not row["reason"].startswith("Exact duplicate of") for row in quarantine_rows
            ),
            "input_duplicate_groups": len({row["sha256"] for row in input_duplicate_rows}),
            "duplicate_samples_quarantined": sum(
                row["action"] == "quarantined_duplicate" for row in output_duplicate_rows
            ),
            "output_duplicate_groups_after_deduplication": len(records) - len(
                {record["sha256"] for record in records}
            ),
            "seed": args.seed,
            "split_ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "resolution_policy": "Preserve source resolution; resize during model preprocessing.",
        }
        (work_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        work_dir.rename(output_root)
    except Exception:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise

    print(f"Created unified dataset: {output_root}")
    print(f"Videos: {len(records)}; quarantined files: {len(quarantine_rows)}")
    print(
        f"Labels: normal={summary['labels']['normal']}, "
        f"shoplifting={summary['labels']['shoplifting']}"
    )
    print(f"Splits: {summary['split_counts']}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a non-destructive unified staged/DCSASS Shoplifting video dataset."
    )
    default_data_root = Path(__file__).resolve().parent.parent / "dataset"
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root,
        help="Folder containing staged_clips and dcsass (default: project dataset folder).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="New output folder (default: <project-root>/cleaned_data/unified_video_deduplicated). It must not already exist.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible splits.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check input names and labels and report counts without writing or decoding videos.",
    )
    return parser.parse_args()


def main() -> int:
    try:
        return clean_dataset(parse_args())
    except (DatasetError, OSError, subprocess.SubprocessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())