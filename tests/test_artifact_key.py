from yeast_xrf.features.manifest import FeatureArtifactKey


def test_feature_key_changes_with_parameters() -> None:
    a = FeatureArtifactKey("abc", "/xrf/Fe", "gradient", {"sigma": 1.0}, "2d")
    b = FeatureArtifactKey("abc", "/xrf/Fe", "gradient", {"sigma": 2.0}, "2d")
    assert a.digest() != b.digest()
