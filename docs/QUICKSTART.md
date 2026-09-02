# FilterAniX Quickstart Guide

Get up and running with FilterAniX in under 2 minutes.

---

## 1. Prerequisites & Installation

Ensure you have Python 3.10+ installed.

```bash
# Clone the repository
git clone https://github.com/SahilKhutey/FilterAniX.git
cd FilterAniX

# Install required dependencies
pip install -r requirements.txt
```

---

## 2. Launching the Interactive Web Studio (UI)

FilterAniX includes a modern, high-performance web interface built for creators:

```bash
python app.py
```
Open **http://127.0.0.1:7860** in your web browser:
1. **Process Creator Video Tab**: Upload any recorded MP4/MOV, select an artistic style (e.g. `anime_creator`), and click **Render Full Pipeline**.
2. **Live Camera Tab**: Stream real-time webcam video with fast GPU/CPU stylization preview.
3. **YouTube Export Tab**: Export multi-resolution YouTube-ready masters (720p, 1080p, 1440p, 4K UHD).
4. **System Diagnostics Tab**: Real-time hardware, GPU VRAM, and FFmpeg capability status.

---

## 3. CLI Quickstart (Command-Line Workflows)

### A. Run Full End-to-End Pipeline
```bash
python run_pipeline.py --input "samples/test_phase1_input.mp4" --style anime_creator --project-dir "projects/my_first_video"
```

### B. Export YouTube Master MP4
```bash
python export_youtube.py --input "projects/my_first_video/phase5/youtube_master.mp4" --output "projects/my_first_video/export/youtube_1080p.mp4" --preset 1080p
```

### C. Check System Diagnostics
```bash
python system_info.py
```

### D. Run Automated Test Suite
```bash
python -m pytest tests/
```
