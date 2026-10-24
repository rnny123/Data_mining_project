import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Data_mining_project.pose_model.pose_utils as pu  # noqa: E402


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
