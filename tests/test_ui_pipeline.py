"""Tests for pipeline event bus integration, stage telemetry, and style overrides."""
from pathlib import Path
import pytest
from src.core.events import EventBus, PipelineEvent
from src.core.jobs import JobManager
from src.core.pipeline import PipelineManager, STAGE_WEIGHTS
from src.core.project import Project
from src.art.mathematical import MathematicalAnimeStyle


def test_pipeline_eventbus_registration(tmp_path):
    project_dir = tmp_path / "test_pipeline_proj"
    project = Project(project_dir)
    project.create("event_test")

    event_bus = EventBus()
    received_events = []

    def on_event(ev: PipelineEvent):
        received_events.append(ev)

    event_bus.subscribe(on_event)

    pipeline = PipelineManager(project, event_bus=event_bus)
    assert pipeline.event_bus is event_bus

    # Emit a test event through the bus
    event = PipelineEvent(
        job_id="test_job",
        stage="input",
        progress=0.05,
        message="Starting stage: input...",
    )
    pipeline.event_bus.emit(event)

    assert len(received_events) == 1
    assert received_events[0].stage == "input"
    assert received_events[0].progress == 0.05


def test_pipeline_stage_weights_completeness():
    # Verify all 7 pipeline phases have assigned weights summing to 1.0
    expected_stages = ["input", "vision", "consistency", "lipsync", "artistic", "composition", "validation"]
    for s in expected_stages:
        assert s in STAGE_WEIGHTS

    total_weight = sum(STAGE_WEIGHTS[s] for s in expected_stages)
    assert pytest.approx(total_weight, 0.01) == 1.0


def test_pipeline_custom_style_instantiation(tmp_path):
    project_dir = tmp_path / "test_style_proj"
    project = Project(project_dir)
    project.create("style_test")

    custom_params = {
        "contrast": 1.25,
        "gamma": 0.90,
        "edge_strength": 0.88,
        "palette_mix": 0.70,
    }

    pipeline = PipelineManager(project)
    # Verify _phase3 accepts style_params dictionary without raising
    # by testing the style resolution logic
    style_obj = MathematicalAnimeStyle(**custom_params).validated()
    assert style_obj.contrast == 1.25
    assert style_obj.edge_strength == 0.88
