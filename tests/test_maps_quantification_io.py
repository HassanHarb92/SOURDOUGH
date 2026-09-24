import h5py
import numpy as np

from yeast_xrf.io.maps_quantification import inspect_maps_quantification


def test_quantification_reader_is_conservative(tmp_path):
    path = tmp_path / "sample.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("/MAPS/Quantification/Number_Of_Standards", data=[1])
        h5.create_dataset(
            "/MAPS/Quantification/Standard0/Element_Weights_Names",
            data=np.array([b"Fe", b"Zn"]),
        )
        h5.create_dataset(
            "/MAPS/Quantification/Standard0/Element_Weights",
            data=np.array([12.0, 5.0]),
        )
        h5.create_dataset(
            "/MAPS/Quantification/Calibration/Fitted/Calibration_Curve_Labels",
            data=np.array(
                [
                    [b"energy"] * 4,
                    [b"response"] * 4,
                    [b"ug/cm2"] * 4,
                ]
            ),
        )
        h5.create_dataset(
            "/MAPS/Quantification/Calibration/Fitted/Calibration_Curve_US_IC",
            data=np.ones((3, 4)),
        )
        h5.create_dataset(
            "/MAPS/Quantification/Calibration/Fitted/US_IC_Element_Info_Names",
            data=np.array([b"name", b"factor"]),
        )
        h5.create_dataset(
            "/MAPS/Quantification/Calibration/Fitted/US_IC_Element_Info_Values",
            data=np.ones((4, 2)),
        )

    report = inspect_maps_quantification(
        path,
        method="Fitted",
        reference="US_IC",
    )
    assert report["read_only"] is True
    assert report["readiness"]["standard_metadata_present"] is True
    assert report["readiness"]["calibration_curve_present"] is True
    assert report["readiness"]["element_info_present"] is True
    assert report["readiness"]["areal_density_unit_detected"] is True
    assert report["readiness"]["automatic_areal_density_conversion_ready"] is False
