from src.art.keyframe_scheduler import KeyframeScheduler


def test_first_frame_is_keyframe():

    scheduler = KeyframeScheduler()

    result = scheduler.build([
        {
            "frame_index": 0,
            "scene_id": 0,
            "is_keyframe": False,
            "is_scene_cut": False,
            "motion_score": 0.0,
            "reference_strength": 0.55,
        }
    ])

    assert len(result) == 1
    assert result[0].is_keyframe
    assert result[0].reason == "initial_keyframe"


def test_scene_cut_forces_keyframe():

    scheduler = KeyframeScheduler()

    result = scheduler.build([
        {
            "frame_index": 0,
            "scene_id": 0,
            "is_keyframe": True,
            "is_scene_cut": False,
            "motion_score": 0.0,
        },
        {
            "frame_index": 1,
            "scene_id": 1,
            "is_keyframe": False,
            "is_scene_cut": True,
            "motion_score": 0.0,
        },
    ])

    assert result[1].is_keyframe
    assert result[1].is_scene_cut


def test_phase4_keyframe_is_preserved():

    scheduler = KeyframeScheduler()

    result = scheduler.build([
        {
            "frame_index": 0,
            "scene_id": 0,
            "is_keyframe": True,
        },
        {
            "frame_index": 5,
            "scene_id": 0,
            "is_keyframe": True,
            "reason": "interval_keyframe",
        },
    ])

    assert result[1].is_keyframe
