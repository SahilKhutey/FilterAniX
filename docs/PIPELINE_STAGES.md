# FilterAniX Pipeline Stages & Modules

| Stage | Package | Core Responsibilities | Output Artifacts |
|---|---|---|---|
| **Phase 1** | `src.io`, `src.processing` | Frame decoding, zero-config FFmpeg resolution, metadata inspection, audio stream preservation. | `phase1/metadata.json` |
| **Phase 2** | `src.vision` | 468 Face Mesh landmarks, 33 3D Pose joints, 21 Hand keypoints, Person segmentation mask, Optical flow, Props detection. | `phase2/vision.jsonl`, `phase2/annotated.mp4` |
| **Phase 3** | `src.art` | XDoG ink extraction, Kuwahara smoothing, CIELAB stepped cel-shading, warm lighting, reference palette transfer, flow warping. | `phase3/artistic_video.mp4` |
| **Phase 4** | `src.consistency` | Canonical character reference signatures, MSE + 3D histogram scene cut detection, motion keyframe injection, quality auditing. | `phase4/temporal_plan.jsonl`, `phase4/consistency_report.json` |
| **Phase 5** | `src.lipsync`, `src.media` | 4-state viseme classification, temporal sliding-window smoothing, EBU R128 audio normalization, 0.000s A/V synchronization. | `phase5/lipsync.jsonl`, `phase5/youtube_master.mp4` |
| **Phase 6** | `src.core` | Gradio web application, Manifest state machine, Checkpointing, Multi-resolution YouTube export (720p–4K). | `export/youtube_1080p.mp4`, `manifest.json` |
