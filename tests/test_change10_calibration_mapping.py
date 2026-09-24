from yeast_xrf.io.maps_calibration_semantics import inspect_calibration_semantics


def test_change10_mapping_probe_import_contract():
    # The focused mapping probe intentionally builds on the read-only
    # semantics inspector installed by the first half of Change 10.
    assert callable(inspect_calibration_semantics)
