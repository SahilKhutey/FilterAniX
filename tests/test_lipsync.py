from pathlib import Path
from src.lipsync.analyzer import (
    classify_mouth,
    extract_mouth_open,
    LipSyncAnalyzer,
    VisemeState,
)
from src.lipsync.smoother import smooth_timeline, LipSyncSmoother
from src.lipsync.timeline import VisemeFrame, LipSyncTimeline
from src.vision.models import FrameVisionData, FaceData, BoundingBox


def test_closed():
    assert classify_mouth(0.02) == "closed"


def test_slightly_open():
    assert classify_mouth(0.15) == "slightly_open"


def test_open():
    assert classify_mouth(0.30) == "open"


def test_wide_open():
    assert classify_mouth(0.60) == "wide_open"


def test_mouth_landmarks():
    result = extract_mouth_open(
        {
            "landmarks": {
                "upper_lip": [0.5, 0.4],
                "lower_lip": [0.5, 0.7],
            }
        }
    )
    assert result > 0


def test_smoothing():
    frames = [
        VisemeFrame(0, 0.0, 0.0, "closed"),
        VisemeFrame(1, 0.033, 0.5, "open"),
        VisemeFrame(2, 0.066, 0.0, "closed"),
    ]
    result = smooth_timeline(frames, window=3)
    assert len(result) == 3
    assert result[1].state == "closed"


def test_lipsync_analyzer_classification():
    """Verifies mouth opening classification into 4 viseme states."""
    analyzer = LipSyncAnalyzer(thresh_closed=0.08, thresh_open=0.22, thresh_wide=0.45)

    assert analyzer.classify_ratio(0.02) == VisemeState.CLOSED
    assert analyzer.classify_ratio(0.15) == VisemeState.SLIGHTLY_OPEN
    assert analyzer.classify_ratio(0.35) == VisemeState.OPEN
    assert analyzer.classify_ratio(0.60) == VisemeState.WIDE_OPEN


def test_lipsync_smoother_noise_rejection():
    """Verifies that an isolated noisy state is smoothed out."""
    analyzer = LipSyncAnalyzer()

    # Sequence: closed, closed, OPEN (noise spike), closed, closed
    ratios = [0.02, 0.03, 0.50, 0.02, 0.03]
    records = []
    for i, r in enumerate(ratios):
        face = FaceData(face_id=0, landmarks=[], bbox=BoundingBox(0, 0, 1, 1), landmark_count=0, mouth_opening=r)
        vision = FrameVisionData(frame_index=i, timestamp=i * 0.033, width=640, height=360, faces=[face])
        records.append(analyzer.analyze_frame(i, i * 0.033, vision))

    smoother = LipSyncSmoother(window_size=3)
    smoothed = smoother.smooth_timeline(records)

    assert len(smoothed) == 5
    assert smoothed[2].viseme == VisemeState.CLOSED.value


def test_lipsync_timeline_save_and_load(tmp_path):
    timeline_file = tmp_path / "test_timeline.jsonl"
    frames = [
        VisemeFrame(0, 0.0, 0.0, "closed"),
        VisemeFrame(1, 0.033, 0.25, "open"),
    ]
    timeline = LipSyncTimeline(fps=30.0, frames=frames)
    timeline.save(str(timeline_file))

    loaded = LipSyncTimeline.load(str(timeline_file), fps=30.0)
    assert loaded.fps == 30.0
    assert len(loaded.frames) == 2
    assert loaded.frames[1].state == "open"
