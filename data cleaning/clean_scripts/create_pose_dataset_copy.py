from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from prepare_pose_test_dataset import DatasetError, prepare_dataset


def create_dataset_copy(args: argparse.Namespace) -> int:
    video_dataset = args.video_dataset.resolve()
    pose_input = args.pose_input.resolve()
    pose_labels = args.pose_labels.resolve()
    output = args.output.resolve()

    if not video_dataset.is_dir():
        raise DatasetError(f"Video dataset folder does not exist: {video_dataset}")
    if output == video_dataset or output in video_dataset.parents or video_dataset in output.parents:
        raise DatasetError("Output folder must not overlap the video dataset folder")
    if output.exists():
        raise DatasetError(f"Output already exists: {output}; choose a new folder")

    pose_args = argparse.Namespace(
        input=pose_input,
        labels=pose_labels,
        output=output / "pose_lift_test",
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return prepare_dataset(pose_args)

    output.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    )
    staged_output = temp_root / output.name
    try:
        shutil.copytree(video_dataset, staged_output)
        pose_args.output = staged_output / "pose_lift_test"
        result = prepare_dataset(pose_args)
        if result != 0:
            raise DatasetError("PoseLift cleaning did not complete successfully")
        staged_output.rename(output)
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
    temp_root.rmdir()

    print(f"Created dataset copy: {output}")
    print(f"Separate PoseLift test dataset: {output / 'pose_lift_test'}")
    return 0


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Copy the unified video dataset and add a cleaned, separate "
            "PoseLift test-only dataset."
        )
    )
    parser.add_argument(
        "--video-dataset",
        type=Path,
        default=project_root / "cleaned_data" / "unified_video_deduplicated",
        help="Unified video dataset folder to copy.",
    )
    parser.add_argument(
        "--pose-input",
        type=Path,
        default=project_root / "dataset" / "poselift" / "Pickle_files" / "Test",
        help="PoseLift test annotation folder; training annotations are not used.",
    )
    parser.add_argument(
        "--pose-labels",
        type=Path,
        default=project_root / "dataset" / "poselift" / "Pickle_files" / "GT",
        help="PoseLift frame-label folder matching the test annotations.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "cleaned_data" / "unified_video_poselift_test",
        help="New output folder; existing folders are never overwritten.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate PoseLift test samples without copying or writing files.",
    )
    return parser.parse_args()


def main() -> int:
    try:
        return create_dataset_copy(parse_args())
    except (DatasetError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())