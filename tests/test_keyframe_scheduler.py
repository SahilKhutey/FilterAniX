from src.art.keyframe_scheduler import KeyframeScheduler


def test_scene_cut_forces_keyframe():
    scheduler = KeyframeScheduler(minimum_interval=12)

    plan = [
        {
            "frame_index": 0,
            "scene_id": 0,
            "is_keyframe": True,
            "is_scene_cut": True,
            "motion_score": 0.0,
        },
        {
            "frame_index": 1,
            "scene_id": 0,
            "is_keyframe": False,
            "is_scene_cut": False,
            "motion_score": 0.0,
        },
        {
            "frame_index": 20,
            "scene_id": 1,
            "is_keyframe": False,
            "is_scene_cut": True,
            "motion_score": 0.0,
        },
    ]

    result = scheduler.build(plan)

    assert result[0].is_keyframe
    assert result[2].is_keyframe


def test_high_motion_creates_keyframe():
    scheduler = KeyframeScheduler(
        minimum_interval=8,
        motion_threshold=0.22,
    )

    plan = [
        {
            "frame_index": 0,
            "scene_id": 0,
            "is_keyframe": True,
            "is_scene_cut": True,
            "motion_score": 0.0,
        },
        {
            "frame_index": 10,
            "scene_id": 0,
            "is_keyframe": False,
            "is_scene_cut": False,
            "motion_score": 0.8,
        },
    ]

    result = scheduler.build(plan)

    assert result[1].is_keyframe
