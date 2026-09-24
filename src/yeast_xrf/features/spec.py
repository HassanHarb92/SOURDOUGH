"""Feature specification types.

No scientific transforms are implemented in the bootstrap.  This is the contract later
intensity, gradient, Hessian, texture, multichannel and multiscale features will register
against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class FeatureKind(StrEnum):
    VISUALIZATION = "visualization"
    FEATURE = "feature"
    REPRESENTATION = "representation"


class FeatureDimensionality(StrEnum):
    TWO_D = "2d"
    THREE_D = "3d"
    EITHER = "either"


@dataclass(frozen=True)
class FeatureSpec:
    key: str
    label: str
    family: str
    kind: FeatureKind
    dimensionality: FeatureDimensionality
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    units: str = "derived"
    source_role: str = "element_map"
    scale_aware: bool = False
    signed: bool = False
