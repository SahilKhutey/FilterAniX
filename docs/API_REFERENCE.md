# FilterAniX API Reference

## 1. Project Management
```python
from src.core.project import Project, ProjectStatus

project = Project("projects/my_video")
project.set_status(ProjectStatus.PHASE_1)
```

## 2. Full Pipeline Execution
```python
from src.core.project import Project
from src.core.pipeline import ProductionPipelineController

project = Project("projects/my_video")
controller = ProductionPipelineController(project=project, style_key="anime_creator")

master_mp4 = controller.run(
    input_video_path="input.mp4",
    max_frames=None,
    resume=True,
)
```

## 3. YouTube Multi-Resolution Export
```python
from src.core.export import YouTubeExporter

exporter = YouTubeExporter(preset_name="1080p")
exporter.export("projects/my_video/phase5/youtube_master.mp4", "export/youtube_1080p.mp4")
```

## 4. Hardware & Capability Diagnostics
```python
from src.core.hardware import get_hardware_report

report = get_hardware_report()
print(report.to_dict())
```
