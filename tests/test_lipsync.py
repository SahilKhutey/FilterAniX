"""Automated Tests for Lip-Sync Extraction and Smoothing."""
from src.lipsync.analyzer import LipSyncAnalyzer, VisemeState
from src.lipsync.smoother import LipSyncSmoother
from src.vision.models import FrameVisionData, FaceData, BoundingBox


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
        vision = FrameVisionData(frame_index=i, timestamp=i*0.033, width=640, height=360, faces=[face])
        records.append(analyzer.analyze_frame(i, i*0.033, vision))

    smoother = LipSyncSmoother(window_size=3)
    smoothed = smoother.smooth_timeline(records)

    assert len(smoothed) == 5
    # The middle spike at index 2 should be smoothed to closed
    assert smoothed[2].viseme == VisemeState.CLOSED.value
