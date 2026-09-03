# FilterAniX — Animated Creator Engine

[![License: Commercial](https://img.shields.io/badge/License-Commercial%20Friendly-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-brightgreen.svg)](https://www.python.org/)
[![Tests: 149+ Passed](https://img.shields.io/badge/tests-149%2B%20passed-success.svg)](tests/)
[![Gradio Studio UI](https://img.shields.io/badge/UI-Gradio%206.26-orange.svg)](app.py)
[![FFmpeg Broadcast](https://img.shields.io/badge/Audio-EBU%20R128%20(-14%20LUFS)-purple.svg)](src/media/)

> **FilterAniX (Animated Creator)** is a state-of-the-art vision and mathematical styling engine that converts real-world creator videos into temporally stable, character-consistent 2D anime/illustrated YouTube master productions.

---

## 🌟 Key Capabilities

* **🎬 Complete Real-to-Anime Transformation**: Preserves real camera composition, facial expressions, hand gestures, posture, microphone, laptop, and background props while transforming the scene into an anime universe.
* **📐 Mathematical Style Engine v1.0 (MTH-01..10)**: 100% deterministic continuous image-field engine performing per-pixel transformations across color, tone, palette, edge line-art, shadows, geometry, facial features, cinematic lighting, and temporal flow warping without diffusion or AI keyframes.
* **🧠 Full Vision & Scene Understanding**: MediaPipe-powered 468 Face Mesh landmarks, 33 3D skeletal pose keypoints, 21-joint hand tracking, person segmentation, and Farneback optical flow.
* **🎨 Procedural & Generative Cel-Shading**: Extended Difference of Gaussians (XDoG) ink lines, anisotropic Kuwahara smoothing, CIELAB stepped cel-shading, and Reinhard palette color transfer.
* **🔒 Character Identity & Temporal Consistency**: Canonical reference profiling, MSE + 3D histogram shot-cut detection, motion-aware keyframing, and automated identity drift auditing.
* **🎙️ Voice Preservation & Lip-Sync**: Retains original microphone voice with EBU R128 (-14.0 LUFS) broadcast loudness normalization and 4-state smoothed viseme tracking driving character mouth animation.
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
                                     │
                   ┌─────────────────┴─────────────────┐
                   ▼                                   ▼
 ┌───────────────────────────────────┐ ┌───────────────────────────────────┐
 │ PHASE 4: Consistency & Plan       │ │ PHASE 5A: Lip-Sync Timeline       │
 │ (Keyframe Decisions, Drift Audit) │ │ (4-State Smoothed Visemes)        │
 └─────────────────┬─────────────────┘ └─────────────────┬─────────────────┘
                   └─────────────────┬───────────────────┘
                                     ▼
            ┌─────────────────────────────────────────────────┐
            │  PHASE 3: Mathematical Style Engine (MTH-01..10)│
            │  (Color, Tone, Palette, Edge, Geometry, Face,   │
            │   Shadow/Highlight, Lighting & Temporal Warp)   │
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

## 📐 Mathematical Anime Style Engine (MTH-01 → MTH-10)

The core Phase-3 rendering engine performs deterministic, per-pixel transformation across 9 mathematical stages:

| Stage | Module | Description |
|---|---|---|
| **MTH-01** | `config.py` | Validated, immutable style parameters (`MathematicalAnimeStyle`) and color palette |
| **MTH-02** | `color_field.py` | $I \to C$: CIELAB color quantization, bilateral tone mapping, saturation control |
| **MTH-03** | `tone_field.py` | $C \to T$: Multi-scale Gaussian luminance, S-curve contrast, smooth quantization |
| **MTH-04** | `palette_field.py` | $T \to P$: Softmax distance projection onto warm anime palette anchors |
| **MTH-05** | `edge_field.py` | $P \to E$: Multi-scale Sobel gradients, Laplacian structure, dark anime ink lines |
| **MTH-06** | `shadow_highlight_field.py` | $E \to S$: Sigmoidal cel-shadows and warm specular highlight modulation |
| **MTH-07** | `geometry_field.py` | Character ($G_C$) vs. background ($B_G$) spatial segmentation & bounding boxes |
| **MTH-08** | `face_field.py` | MediaPipe 468 landmark Gaussian fields protecting eyes, nose, and mouth |
| **MTH-09** | `lighting_field.py` | Cinematic warm key light and cool shadow tinting |
| **MTH-10** | `temporal_field.py` | Optical-flow stabilized inter-frame temporal warping ($A_t = (1 - \lambda_t) F(I_t) + \lambda_t \mathcal{W}(A_{t-1}, \Phi_t)$) |

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
python run_pipeline.py --input "samples/A_pose_explaingn_Something_.mp4" --project "projects/my_video"
```

### 4. Export YouTube 1080p Master
```bash
python export_youtube.py --input "projects/my_video/output/youtube_master.mp4" --output "projects/my_video/export/youtube_1080p.mp4" --preset 1080p
```

---

## 🧪 Testing

The repository includes a comprehensive 149+ automated test suite covering all mathematical stages, core contracts, and full system integration:

```bash
# Run all mathematical and system integration tests
python -m pytest tests/test_mth*.py tests/test_system_integration.py tests/test_mathematical_fields.py -v

# Run the complete test suite
python -m pytest tests/ -v
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
├── configs/
│   └── mathematical_anime.yaml # Canonical MTH-01..10 YAML Configuration
├── docs/                       # Technical Documentation
│   ├── ARCHITECTURE.md
│   ├── PIPELINE_STAGES.md
│   ├── QUICKSTART.md
│   └── STYLE_GUIDE.md
├── src/
│   ├── core/                   # FramePacket, error hierarchy, logging, pipeline
│   ├── vision/                 # MediaPipe Face, Pose, Hands, Scene cuts
│   ├── consistency/            # Character reference signatures & temporal planning
│   ├── lipsync/                # 4-state viseme classification & timeline smoothing
│   ├── art/mathematical/       # MTH-01..10 Mathematical Style Engine & Renderer
│   ├── media/                  # Audio multiplexing, EBU R128 loudness, validation
│   └── io/                     # Video decode, inspection, and writer wrappers
├── tests/                      # Automated test suite (149+ unit & integration tests)
└── tools/                      # Diagnostic inspection tools (inspect_mth01..10)
```
