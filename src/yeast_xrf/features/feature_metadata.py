'''Metadata shared by derived XRF feature maps.'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

FeatureKind = Literal["display", "derived_feature", "normalized_product"]


@dataclass(frozen=True)
class FeatureMetadata:
    name: str
    kind: FeatureKind
    description: str
    input_label: str
    output_unit: str
    parameters: dict[str, Any]
    provenance: dict[str, Any]


def make_feature_metadata(
    *,
    name: str,
    kind: FeatureKind,
    description: str,
    input_label: str,
    output_unit: str,
    parameters: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> FeatureMetadata:
    return FeatureMetadata(
        name=name,
        kind=kind,
        description=description,
        input_label=input_label,
        output_unit=output_unit,
        parameters=dict(parameters or {}),
        provenance=dict(provenance or {}),
    )
