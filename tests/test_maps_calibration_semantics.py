import h5py
import numpy as np

from yeast_xrf.io.maps_calibration_semantics import (
    compact_signature,
    inspect_calibration_semantics,
)


def test_semantics_reader_is_read_only_and_preserves_labels(tmp_path):
    path = tmp_path / "sample.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset(
            "/MAPS/XRF_Analyzed/Fitted/Channel_Names",
            data=np.array([b"P", b"Zn"]),
        )
        h5.create_dataset(
            "/MAPS/XRF_Analyzed/Fitted/Channel_Units",
            data=np.array([b"cts/s", b"cts/s"]),
        )
        h5.create_dataset(
            "/MAPS/Quantification/Calibration/Fitted/Calibration_Curve_Labels",
            data=np.array(
                [
                    [b"P", b"Zn"],
                    [b"P", b"Zn"],
                    [b"P", b"Zn"],
                ]
            ),
        )
        h5.create_dataset(
            "/MAPS/Quantification/Calibration/Fitted/Calibration_Curve_US_IC",
            data=np.array(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                    [5.0, 6.0],
                ]
            ),
        )
        h5.create_dataset(
            "/MAPS/Quantification/Calibration/Fitted/US_IC_Element_Info_Index",
            data=np.array([0, 1, 2, 3]),
        )
        h5.create_dataset(
            "/MAPS/Quantification/Calibration/Fitted/US_IC_Element_Info_Names",
            data=np.array([b"name", b"weight", b"signal", b"factor"]),
        )
        h5.create_dataset(
            "/MAPS/Quantification/Calibration/Fitted/US_IC_Element_Info_Values",
            data=np.arange(16, dtype=float).reshape(4, 4),
        )
        h5.create_dataset(
            "/MAPS/XRF_fits_quant_names",
            data=np.array([b"standard"]),
        )
        h5.create_dataset(
            "/MAPS/XRF_fits_quant",
            data=np.arange(12, dtype=float).reshape(3, 2, 2),
        )

    report = inspect_calibration_semantics(
        path,
        method="Fitted",
        reference="US_IC",
    )
    assert report["read_only"] is True
    assert report["channel_names"]["values"] == ["P", "Zn"]
    assert report["calibration"]["curve_labels"]["values"] == [
        ["P", "Zn"],
        ["P", "Zn"],
        ["P", "Zn"],
    ]
    assert report["calibration"]["element_info_names"]["values"] == [
        "name",
        "weight",
        "signal",
        "factor",
    ]
    assert report["calibration"]["curve"]["shape"] == [3, 2]
    assert report["calibration"]["element_info_values"]["shape"] == [4, 4]
    assert report["quantification"]["values"]["shape"] == [3, 2, 2]

    sig = compact_signature(report)
    assert sig["method"] == "Fitted"
    assert sig["reference"] == "US_IC"
    assert sig["channel_names"] == ["P", "Zn"]


def test_reader_does_not_modify_hdf5(tmp_path):
    path = tmp_path / "sample.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset(
            "/MAPS/XRF_Analyzed/Fitted/Channel_Names",
            data=np.array([b"P"]),
        )

    before = path.read_bytes()
    inspect_calibration_semantics(
        path,
        method="Fitted",
        reference="US_IC",
    )
    after = path.read_bytes()
    assert before == after
