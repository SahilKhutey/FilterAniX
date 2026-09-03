# FilterAniX Pipeline Stages & Modules

| Stage | Package | Core Responsibilities | Output Artifacts |
|---|---|---|---|
| **Phase 1** | `src.io`, `src.processing` | Frame decoding, zero-config FFmpeg resolution, metadata inspection, audio stream preservation. | `source/video.mp4`, `metadata.json` |
| **Phase 2** | `src.vision` | 468 Face Mesh landmarks, 33 3D Pose joints, 21 Hand keypoints, Person segmentation mask, Optical flow, Scene cuts. | `vision/vision.jsonl` |
| **Phase 4** | `src.consistency` | Canonical character reference signatures, scene cut detection, motion keyframe injection, temporal planning. | `consistency/temporal_plan.jsonl` |
| **Phase 5A** | `src.lipsync` | 4-state viseme classification, temporal sliding-window smoothing for mouth synchronization. | `lipsync/lipsync.jsonl` |
| **Phase 3** | `src.art.mathematical` | Deterministic 9-layer Mathematical Style Engine (MTH-02 Color, MTH-03 Tone, MTH-04 Palette, MTH-05 Edge, MTH-06 Shadow/Highlight, MTH-07 Geometry, MTH-08 Face, MTH-09 Lighting, MTH-10 Temporal). | `artistic/animated.mp4` |
| **Phase 5B** | `src.media` | Audio multiplexing, EBU R128 loudness normalization (-14.0 LUFS), 0.000s A/V synchronization. | `output/youtube_master.mp4` |
| **Phase 6** | `src.core.export` | Multi-resolution YouTube export (720p, 1080p, 1440p, 4K UHD), broadcast validation. | `export/youtube_1080p.mp4`, `validation.json` |
