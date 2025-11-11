"""
TurboReg Python Implementation
Point Cloud Registration using GPU acceleration with PyTorch
"""

from .turboreg import TurboRegGPU
from .turboregPlus import TurboRegPlus
from .turboregCopCons import TurboRegCopCons
from .model_selection import ModelSelection, MetricType
from .rigid_transform import RigidTransform, rigid_transform_3d
from .core_turboreg import (
    verification,
    verification_v2_metric,
    post_refinement,
    transform
)
from .utils_pcr import coplanar_constraint

__all__ = [
    'TurboRegGPU',
    'ModelSelection',
    'MetricType',
    'RigidTransform',
    'rigid_transform_3d',
    'verification',
    'verification_v2_metric',
    'post_refinement',
    'transform',
    'coplanar_constraint'
]

__version__ = '1.0.0'

