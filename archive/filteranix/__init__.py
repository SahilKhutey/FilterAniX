"""FilterAniX - AI Video Stylization and Animation Transformation Engine."""
from filteranix.core.config import FilterAniXConfig, load_config
from filteranix.pipeline.offline_pipeline import OfflineVideoPipeline

__version__ = "0.1.0"
__all__ = ["FilterAniXConfig", "load_config", "OfflineVideoPipeline"]
