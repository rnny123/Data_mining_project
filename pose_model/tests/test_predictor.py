import numpy as np
import pandas as pd
import pytest

import Data_mining_project.pose_model.pose_utils as pu

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


@pytest.mark.skipif(not (pu.RESULTS_DIR / "clip_features.csv").exists(), reason="run 02_features_ml.ipynb first")
def test_prediction_features_match_training_table():
    """The predictor's feature path must reproduce the table the model was trained on."""
    table = pd.read_csv(pu.RESULTS_DIR / "clip_features.csv").set_index("relative_path")
    m = pu.load_manifest().set_index("relative_path")
    names = [c for c in table.columns if c not in ("label", "label_name", "source", "split")]
    for rp in table.index[::25]:  # every 25th clip, both sources
        live = pu.clip_features(pu.clip_to_sequence(pu.load_extraction(m.loc[rp, "cache_path"])))
        np.testing.assert_allclose(live[names].to_numpy(float), table.loc[rp, names].to_numpy(float), atol=1e-9, err_msg=rp)


def _staged_test_clips():
    """Staged test clips with the primary person's visibility from 01_pose_extraction.ipynb."""
    m = pu.load_manifest()
    quality = pd.read_csv(pu.RESULTS_DIR / "pose_quality.csv").set_index("relative_path")
    test = m[(m.split == "test") & (m.source == "staged")].copy()
    test["primary_rate"] = test["relative_path"].map(quality["primary_rate"])
    return test


@needs_models
def test_predict_clip_on_real_test_videos():
    test = _staged_test_clips()
    predictor = pu.ShopliftingPredictor("best")
    for label in ("shoplifting", "normal"):
        row = test[(test.label_name == label) & (test.primary_rate >= 0.8)].iloc[0]
        out = predictor.predict(row.video_path)
        assert out["label"] in ("normal", "shoplifting") and 0.0 <= out["probability"] <= 1.0
        assert out["no_person_detected"] is False


@needs_models
def test_predict_clip_flags_empty_room_clip():
    test = _staged_test_clips()
    row = test[test.primary_rate == 0].iloc[0]  # some staged "normal" clips show an empty room
    out = pu.ShopliftingPredictor("best").predict(row.video_path)
    assert out["no_person_detected"] is True
    assert 0.0 <= out["probability"] <= 1.0
