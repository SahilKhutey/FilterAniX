# FilterAniX Architecture Specification

FilterAniX is structured as a 6-phase sequential production pipeline with independent workspaces, checkpointing, and source timing preservation.

```
                           REAL CREATOR VIDEO
                                   │
                                   ▼
          ┌─────────────────────────────────────────────────┐
          │  PHASE 1: Video Input & Metadata Infrastructure │
          └────────────────────────┬────────────────────────┘
                                   ▼
          ┌─────────────────────────────────────────────────┐
          │  PHASE 2: Vision & Scene Understanding Engine   │
          │  (Face Mesh, 3D Pose, Hands, Mask, Flow, Props) │
          └────────────────────────┬────────────────────────┘
                                   │
                  ┌────────────────┴────────────────┐
                  ▼                                 ▼
┌───────────────────────────────────┐ ┌───────────────────────────────────┐
│ PHASE 4: Consistency & Plan       │ │ PHASE 5A: Lip-Sync Timeline       │
│ (Keyframe Decisions, Drift Audit) │ │ (4-State Smoothed Visemes)        │
└─────────────────┬─────────────────┘ └─────────────────┬─────────────────┘
                  └────────────────┬────────────────────┘
                                   ▼
          ┌─────────────────────────────────────────────────┐
          │  PHASE 3: Mathematical Style Engine (MTH-01..10)│
          │  (Continuous Color, Tone, Palette, Edge, Face,  │
          │   Geometry, Lighting & Temporal Flow Warping)   │
          └────────────────────────┬────────────────────────┘
                                   ▼
          ┌─────────────────────────────────────────────────┐
          │  PHASE 5B: Multi-Track Composition & Audio Mux  │
          │  (Voice Preservation, EBU R128 Loudnorm Muxing) │
          └────────────────────────┬────────────────────────┘
                                   ▼
          ┌─────────────────────────────────────────────────┐
          │  PHASE 6: Production Studio & YouTube Exporter  │
          │  (Manifest Engine, Gradio Web UI, Multi-Res MP4)│
          └────────────────────────┬────────────────────────┘
                                   ▼
                         🎬 YOUTUBE MASTER VIDEO
```

---

## Canonical Phase Sequence & Responsibilities

1. **Phase 1: Video Input & Media Infrastructure (`src.io`, `src.processing`)**
   - Ingests MP4/camera streams, probes metadata (resolution, duration, FPS), extracts audio stream to uncompressed WAV.
2. **Phase 2: Vision & Scene Understanding (`src.vision`)**
   - MediaPipe Face Mesh (468 landmarks), 33 3D Pose keypoints, 21-joint Hand tracking, person segmentation, Farneback optical flow, and shot-boundary detection. Outputs aligned `vision.jsonl`.
3. **Phase 4: Character Consistency & Temporal Planning (`src.consistency`)**
   - Establishes canonical reference banks, detects scene cuts, identifies motion keyframes, and builds `temporal_plan.jsonl`.
4. **Phase 5A: Lip-Sync Analysis (`src.lipsync`)**
   - Classifies mouth openness/width into 4-state smoothed visemes (`lipsync.jsonl`).
5. **Phase 3: Mathematical Style Engine (`src.art.mathematical`)**
   - **MTH-01**: Validated, immutable style configuration (`MathematicalAnimeStyle`).
   - **MTH-02**: Color-field decomposition, Lab color quantization, and bilateral smoothing ($I \to C$).
   - **MTH-03**: Multi-scale tone/luminance separation with S-curve contrast mapping ($C \to T$).
   - **MTH-04**: Continuous softmax projection onto anime palette anchors ($T \to P$).
   - **MTH-05**: Multi-scale Sobel and Laplacian ink line-field extraction ($P \to E$).
   - **MTH-06**: Sigmoidal cel-shadow and specular highlight modulation ($E \to S$).
   - **MTH-07**: Character vs. background geometry fields ($G_C, B_G$).
   - **MTH-08**: Facial landmark Gaussian influence fields protecting eyes, nose, and lips.
   - **MTH-09**: Directional cinematic warm key light and cool shadow tinting.
   - **MTH-10**: Optical-flow stabilized inter-frame temporal warping:
     $$A_t = (1 - \lambda_t) F(I_t) + \lambda_t \mathcal{W}(A_{t-1}, \Phi_t)$$
6. **Phase 5B: Media Composition & Audio Multiplexing (`src.media`)**
   - Muxes stylized frames with original voice track, applying EBU R128 loudness normalization (-14.0 LUFS, -1.5 dBTP, 11 LU LRA). Guarantees `duration_delta = 0.000s`.
7. **Phase 6: YouTube Master Export (`src.core.export`)**
   - Broadcast-grade H.264/AAC encoding across 720p, 1080p, 1440p, and 4K UHD presets with faststart MP4 containers.
