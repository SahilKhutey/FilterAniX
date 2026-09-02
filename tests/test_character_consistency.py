import pytest
import numpy as np
from pathlib import Path

from src.consistency.controller import (
    IdentityRenderController,
)
from src.consistency.retry import (
    IdentityRetryPolicy,
)
from src.consistency.reference_bank import (
    ReferenceBank,
    ReferenceImage,
)
from src.consistency.identity import (
    build_identity_profile,
    histogram_similarity,
)
from src.consistency.temporal import (
    CharacterTemporalState,
)


def test_first_frame_is_keyframe():
    controller = IdentityRenderController()

    decision = controller.decide(
        frame_index=0,
        scene_id=0,
        scene_cut=False,
        motion_score=0.0,
    )

    assert decision.keyframe
    assert decision.reason == "first_frame"


def test_scene_cut_forces_keyframe():
    controller = IdentityRenderController()

    decision = controller.decide(
        frame_index=7,
        scene_id=1,
        scene_cut=True,
        motion_score=0.0,
    )

    assert decision.keyframe
    assert decision.scene_cut
    assert decision.reason == "scene_cut"


def test_high_motion_forces_keyframe():
    controller = IdentityRenderController(
        keyframe_interval=100
    )

    decision = controller.decide(
        frame_index=7,
        scene_id=0,
        scene_cut=False,
        motion_score=0.8,
    )

    assert decision.keyframe
    assert decision.reason == "high_motion"


def test_intermediate_frame():
    controller = IdentityRenderController(
        keyframe_interval=12
    )

    decision = controller.decide(
        frame_index=5,
        scene_id=0,
        scene_cut=False,
        motion_score=0.01,
    )

    assert not decision.keyframe
    assert decision.reason == "intermediate"


def test_retry_when_identity_drifts():
    policy = IdentityRetryPolicy(
        threshold=0.62,
        max_attempts=2,
    )

    assert policy.should_retry(
        similarity=0.40,
        attempt=0,
    )


def test_no_retry_after_limit():
    policy = IdentityRetryPolicy(
        threshold=0.62,
        max_attempts=2,
    )

    assert not policy.should_retry(
        similarity=0.40,
        attempt=2,
    )


def test_reference_strength_increases():
    policy = IdentityRetryPolicy()

    assert (
        round(policy.next_strength(0.70), 2)
        == 0.80
    )


def test_reference_bank_selection():
    bank = ReferenceBank("references")
    assert len(bank.references) >= 8

    # Expression match
    ref_smile = bank.select(expression="smile")
    assert ref_smile is not None
    assert "smile" in ref_smile.tags

    # Pose match
    ref_gesture = bank.select(pose="gesture")
    assert ref_gesture is not None
    assert "gesture" in ref_gesture.tags

    # Default fallback
    ref_default = bank.select()
    assert ref_default is not None


def test_identity_profile_and_similarity():
    p1 = build_identity_profile("references/neutral.jpg")
    p2 = build_identity_profile("references/neutral.jpg")
    p3 = build_identity_profile("references/smile.jpg")

    assert p1.width == 512
    assert p1.height == 512
    assert p1.edge_density >= 0.0

    # Self similarity is near 1.0
    sim_self = histogram_similarity(p1, p2)
    assert sim_self > 0.95

    # Cross similarity is computed
    sim_cross = histogram_similarity(p1, p3)
    assert 0.0 <= sim_cross <= 1.0


def test_character_temporal_state_propagation():
    state = CharacterTemporalState(blend_strength=0.20)

    f1 = np.full((100, 100, 3), 100, dtype=np.uint8)
    f2 = np.full((100, 100, 3), 150, dtype=np.uint8)
    f_cut = np.full((100, 100, 3), 20, dtype=np.uint8)

    # Set initial keyframe
    state.set_keyframe(f1, scene_id=0)
    assert state.keyframe is not None

    # Propagate frame 2 in same scene
    res2 = state.propagate(f2, scene_id=0)
    assert res2.shape == (100, 100, 3)
    assert not np.array_equal(res2, f2)  # Blended with previous

    # Scene cut reset
    res_cut = state.propagate(f_cut, scene_id=1)
    assert np.array_equal(res_cut, f_cut)  # No cross-scene blending
