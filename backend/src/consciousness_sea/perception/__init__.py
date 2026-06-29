"""感知层：感知管理、视觉锚点、听觉锚点、体感锚点、Hebbian关联、多模态对齐"""

from .perception import (
    PerceptionManager, PerceptionManagerStatus,
    PerceptActivationEvent, ConceptActivationEvent,
    PerceptionChannel, PerceptualSeedStatus,
)
from .visual_anchor import VisualAnchor, VisualFeatures
from .audio_anchor import AudioAnchor, AudioFeatures
from .somatic_anchor import SomaticAnchor, SomaticFeatures
from .hebbian_binder import HebbianBinder, HebbianBinderStatus
from .multimodal_aligner import MultimodalAligner, AlignmentResult

__all__ = [
    'PerceptionManager', 'PerceptionManagerStatus',
    'PerceptActivationEvent', 'ConceptActivationEvent',
    'PerceptionChannel', 'PerceptualSeedStatus',
    'VisualAnchor', 'VisualFeatures',
    'AudioAnchor', 'AudioFeatures',
    'SomaticAnchor', 'SomaticFeatures',
    'HebbianBinder', 'HebbianBinderStatus',
    'MultimodalAligner', 'AlignmentResult',
]