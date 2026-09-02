"""Phase 4 Consistency Package."""
from src.consistency.types import (
    ReferenceProfile,
    TemporalState,
    ConsistencyMetrics,
    ConsistencyReport,
)
from src.consistency.identity import (
    IdentityProfile,
    build_identity_profile,
    histogram_similarity,
    IdentityProfileBuilder,
    IdentityScorer,
)
from src.consistency.reference_bank import (
    ReferenceBank,
    ReferenceImage,
)
from src.consistency.controller import (
    RenderDecision,
    IdentityRenderController,
    TemporalController,
)
from src.consistency.retry import IdentityRetryPolicy
from src.consistency.temporal import CharacterTemporalState
from src.consistency.scene import SceneDetector
from src.consistency.motion import MotionAnalyzer
from src.consistency.planner import TemporalPlanner
from src.consistency.report import ConsistencyAuditor

__all__ = [
    "IdentityProfile",
    "build_identity_profile",
    "histogram_similarity",
    "ReferenceProfile",
    "TemporalState",
    "RenderDecision",
    "ConsistencyMetrics",
    "ConsistencyReport",
    "IdentityProfileBuilder",
    "IdentityScorer",
    "ReferenceBank",
    "ReferenceImage",
    "IdentityRenderController",
    "TemporalController",
    "IdentityRetryPolicy",
    "CharacterTemporalState",
    "SceneDetector",
    "MotionAnalyzer",
    "TemporalPlanner",
    "ConsistencyAuditor",
]
