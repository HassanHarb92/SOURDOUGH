from yeast_xrf.features.registry import FeatureRegistry
from yeast_xrf.features.spec import FeatureDimensionality, FeatureKind, FeatureSpec


def test_registry_round_trip() -> None:
    registry = FeatureRegistry()
    spec = FeatureSpec(
        key="example",
        label="Example",
        family="testing",
        kind=FeatureKind.FEATURE,
        dimensionality=FeatureDimensionality.TWO_D,
        description="test",
    )
    registry.register(spec)
    assert registry.get("example") == spec
    assert registry.families() == ("testing",)
