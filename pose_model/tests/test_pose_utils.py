import numpy as np
import pytest
import torch

import Data_mining_project.pose_model.pose_utils as pu


def test_cache_path_for_drops_videos_prefix():
    assert pu.cache_path_for("videos/normal/normal-1.mp4") == pu.CACHE_DIR / "normal" / "normal-1.npz"


@pytest.mark.skipif(not (pu.DATA_DIR / "manifest.csv").exists(), reason="dataset not available")
def test_load_manifest_adds_columns():
    m = pu.load_manifest()
    assert len(m) == 207
    assert set(m["source"]) == {"staged", "dcsass"}
    assert m["cache_path"].is_unique
    assert set(m["split"]) == {"train", "validation", "test"}


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


def _hidden(kpts, *names):
    """Copy of a (T,17,3) pose sequence with the named keypoints never confident."""
    kpts = kpts.copy()
    for name in names:
        kpts[:, pu.K[name], 2] = 0.05
    return kpts


@pytest.mark.parametrize("hidden", [
    ("left_ankle", "right_ankle"),     # legs cut off by the frame
    ("left_hip",),                     # one hip never seen
    ("left_hip", "right_hip"),         # no hip ever seen
])
def test_normalize_pose_never_seen_keypoints_are_position_independent(standing_pose, hidden):
    small = _hidden(np.stack([standing_pose(160, 120, 50)] * 4), *hidden)
    big = _hidden(np.stack([standing_pose(500, 400, 200)] * 4), *hidden)
    a = pu.normalize_pose(small, np.full(4, 180.0))
    b = pu.normalize_pose(big, np.full(4, 720.0))
    np.testing.assert_allclose(a["seq"], b["seq"], atol=1e-5)
    for name in hidden:
        assert (a["seq"][:, pu.K[name]] == 0).all() and (a["vel"][:, pu.K[name]] == 0).all()


def test_crouch_feature_does_not_depend_on_position_when_ankles_hidden(standing_pose):
    def crouch(cx, cy):
        kpts = _hidden(np.stack([standing_pose(cx, cy)] * 3), "left_ankle", "right_ankle")
        s = {**pu.normalize_pose(kpts, np.full(3, 360.0)), "n_people": np.ones(3, np.float32), "has_person": True}
        return pu.frame_features(s)["crouch"].to_numpy()

    np.testing.assert_allclose(crouch(160, 120), crouch(500, 400), atol=1e-5)


def test_clip_to_sequence_without_confident_keypoints_has_no_person(make_ext, standing_pose):
    ext = make_ext(T=3)
    for t in range(3):
        ext["kpts"][t, 0] = standing_pose()
        ext["kpts"][t, 0, :, 2] = 0.1  # a box was detected, but no keypoint is reliable
        ext["boxes"][t, 0] = [250, 40, 350, 400]
        ext["track_ids"][t, 0] = 1
    assert pu.clip_to_sequence(ext)["has_person"] is False


def test_single_frame_clip_is_finite(make_ext, standing_pose):
    ext = make_ext(T=1)
    ext["kpts"][0, 0] = standing_pose()
    ext["boxes"][0, 0] = [250, 40, 350, 400]
    ext["track_ids"][0, 0] = 1
    s = pu.clip_to_sequence(ext)
    assert s["has_person"] is True
    assert np.isfinite(pu.clip_features(s).to_numpy()).all()
    assert np.isfinite(pu.window_probs(pu.PoseGRU(), pu.sequence_input(s))).all()


def test_clip_to_sequence_empty_clip_is_finite(make_ext):
    s = pu.clip_to_sequence(make_ext())
    assert s["has_person"] is False
    for key in ("seq", "vel", "conf", "n_people"):
        assert np.isfinite(s[key]).all()
    assert s["seq"].shape == (20, 17, 2)


def test_compute_stride():
    assert pu.compute_stride(30) == 3
    assert pu.compute_stride(60) == 6
    assert pu.compute_stride(25) == 2
    assert pu.compute_stride(5) == 1
    assert pu.compute_stride(0) == 3
    assert pu.compute_stride(float("nan")) == 3


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
