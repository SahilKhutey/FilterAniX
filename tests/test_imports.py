def test_project_modules_importable():
    from src.vision.types import FrameVision, MotionData
    from src.vision.motion import OpticalFlowMotion

    assert FrameVision is not None
    assert MotionData is not None
    assert OpticalFlowMotion is not None
