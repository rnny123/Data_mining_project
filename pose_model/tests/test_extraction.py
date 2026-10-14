import numpy as np
import pytest

import Data_mining_project.pose_model.pose_utils as pu

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
