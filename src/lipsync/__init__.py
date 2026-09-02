"""Phase 5 Lip-Sync Package."""
from src.lipsync.timeline import VisemeFrame, LipSyncTimeline
from src.lipsync.analyzer import (
    extract_mouth_open,
    classify_mouth,
    analyze_mouth_frame,
    VisemeState,
    LipSyncRecord,
    LipSyncAnalyzer,
)
from src.lipsync.smoother import smooth_timeline, LipSyncSmoother

__all__ = [
    "VisemeFrame",
    "LipSyncTimeline",
    "extract_mouth_open",
    "classify_mouth",
    "analyze_mouth_frame",
    "smooth_timeline",
    "VisemeState",
    "LipSyncRecord",
    "LipSyncAnalyzer",
    "LipSyncSmoother",
]
