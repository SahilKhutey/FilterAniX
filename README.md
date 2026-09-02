# FilterAniX — Animated Creator Engine

[![License: Commercial](https://img.shields.io/badge/License-Commercial%20Friendly-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-brightgreen.svg)](https://www.python.org/)
[![Tests: 21 Passed](https://img.shields.io/badge/tests-21%20passed-success.svg)](tests/)
[![Gradio Studio UI](https://img.shields.io/badge/UI-Gradio%206.26-orange.svg)](app.py)
[![FFmpeg Broadcast](https://img.shields.io/badge/Audio-EBU%20R128%20(-14%20LUFS)-purple.svg)](src/media/)

> **FilterAniX (Animated Creator)** is a state-of-the-art vision and generative styling engine that converts real-world creator videos into temporally stable, character-consistent 2D anime/illustrated YouTube master productions.

---

## 🌟 Key Capabilities

* **🎬 Complete Real-to-Anime Transformation**: Preserves real camera composition, facial expressions, hand gestures, posture, microphone, laptop, and background props while transforming the scene into an anime universe.
* **🧠 Full Vision & Scene Understanding**: MediaPipe-powered 468 Face Mesh landmarks, 33 3D skeletal pose keypoints, 21-joint hand tracking, person segmentation, and Farneback optical flow.
* **🎨 Procedural & Generative Cel-Shading**: Extended Difference of Gaussians (XDoG) ink lines, anisotropic Kuwahara smoothing, CIELAB stepped cel-shading, and Reinhard palette color transfer.
* **🔒 Character Identity & Temporal Consistency**: Canonical reference profiling, MSE + 3D histogram shot-cut detection, motion-aware keyframing, and automated identity drift auditing.
* **🎙️ Voice Preservation & Lip-Sync**: Retains original microphone voice with EBU R128 (-14.0 LUFS) broadcast loudness normalization and 4-state smoothed viseme tracking.
* **⚡ Interactive Gradio Studio & CLI**: Features a modern web UI (`app.py`), live webcam streaming mode, and headless CLI orchestrator with stage checkpointing.
* **📤 YouTube Multi-Resolution Export**: Broadcast H.264/AAC MP4 encoding at 720p, 1080p, 1440p, and 4K UHD with `faststart` container flags.

---

## 🏛️ System Architecture

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
                                    ▼
           ┌─────────────────────────────────────────────────┐
           │  PHASE 3: Artistic Style & Cel-Shading Engine   │
           │  (Procedural Kuwahara, CIELAB Cel, Inking, Flow)│
           └────────────────────────┬────────────────────────┘
                                    ▼
           ┌─────────────────────────────────────────────────┐
           │  PHASE 4: Character Identity & Temporal Engine  │
           │  (Reference Signatures, Scene Cuts, Keyframes)  │
           └────────────────────────┬────────────────────────┘
                                    ▼
           ┌─────────────────────────────────────────────────┐
           │  PHASE 5: Lip-Sync & Multi-Track Composition    │
           │  (4-State Visemes, EBU R128 Loudnorm, Voice Mux)│
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

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/SahilKhutey/FilterAniX.git
cd FilterAniX
pip install -r requirements.txt
```

### 2. Launch Interactive Web Studio
```bash
python app.py
```
Open **http://127.0.0.1:7860** in your web browser.

### 3. Run Pipeline via Command Line
```bash
python run_pipeline.py --input "samples/test_phase1_input.mp4" --style anime_creator --project-dir "projects/my_video"
```

### 4. Export YouTube 1080p Master
```bash
python export_youtube.py --input "projects/my_video/phase5/youtube_master.mp4" --output "projects/my_video/export/youtube_1080p.mp4" --preset 1080p
```

---

## 🧪 Testing

The repository includes a comprehensive 21-test automated suite covering all 6 phases:

```bash
python -m pytest tests/
```

```
collected 21 items
tests/test_consistency.py ...                                            [ 14%]
tests/test_export.py .                                                   [ 19%]
tests/test_imports.py .                                                  [ 23%]
tests/test_lipsync.py ..                                                 [ 33%]
tests/test_phase2_vision.py ....                                         [ 52%]
tests/test_phase3.py ...                                                 [ 66%]
tests/test_pipeline_smoke.py .                                           [ 71%]
tests/test_project.py .                                                  [ 76%]
tests/test_validate.py .                                                 [ 80%]
tests/test_video_io.py ....                                              [100%]
======================= 21 passed in 6.02s =======================
```

---

## 📁 Repository Structure

```
FilterAniX/
├── app.py                      # Interactive Web Studio Application (Gradio)
├── run_pipeline.py             # Master Production Pipeline CLI Runner
├── export_youtube.py           # Multi-Resolution YouTube Exporter CLI
├── system_info.py              # Hardware & Capability Diagnostics CLI
├── config.json                 # Core Application Configuration
├── styles.json                 # Configurable Artistic Style Presets
├── requirements.txt            # Python Dependencies
├── LICENSE                     # Commercial & Open Software License
├── README.md                   # Project Documentation
│
├── docs/                       # Technical Documentation
│   ├── QUICKSTART.md           # 2-Minute Onboarding Guide
│   ├── ARCHITECTURE.md         # Mathematical & System Specification
│   ├── PIPELINE_STAGES.md      # Detailed Phase Breakdown
│   ├── STYLE_GUIDE.md          # Customizing Style Presets
│   └── API_REFERENCE.md        # Python API Guide
│
├── src/
│   ├── core/                   # Project Manager, Manifest, Exporter, Hardware
│   ├── io/                     # Phase 1: Video I/O & Metadata
│   ├── vision/                 # Phase 2: MediaPipe Face, Pose, Hands, Flow
│   ├── art/                    # Phase 3: Cel-Shading, Inking, Temporal Warp
│   ├── consistency/            # Phase 4: Identity Profiling & Temporal Plan
│   ├── lipsync/                # Phase 5: Viseme Extraction & Smoothing
│   └── media/                  # Phase 5: EBU R128 Normalizer & Muxer
│
└── tests/                      # Automated Pytest Suite
```

---

## 📄 License & Commercial Rights

Copyright (c) 2026 **Sahil Khutey**. All rights reserved.
Licensed under the [FilterAniX Commercial Software License](LICENSE). Creators retain 100% intellectual property and commercial monetization rights over all generated video outputs.
