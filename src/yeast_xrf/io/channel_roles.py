'''Channel-role metadata for the currently observed MAPS XRF channels.

Roles are descriptive only. They never alter numeric arrays or imply biological meaning.
'''

from __future__ import annotations

from enum import StrEnum


class ChannelRole(StrEnum):
    ELEMENTAL_OR_LINE = "elemental_or_line"
    FLUORESCENCE = "fluorescence"
    SCATTER = "scatter"
    FIT_DIAGNOSTIC = "fit_diagnostic"
    AUXILIARY = "auxiliary"


OBSERVED_ELEMENTAL_OR_LINE_CHANNELS = frozenset(
    {
        "Al", "Si", "P", "S", "Cl", "K", "Ca", "Ti", "Cr", "Mn",
        "Fe", "Ni", "Cu", "Zn", "La_L", "Si_Si",
    }
)
FLUORESCENCE_CHANNELS = frozenset({"Total_Fluorescence_Yield"})
SCATTER_CHANNELS = frozenset(
    {"COMPTON_AMPLITUDE", "COHERENT_SCT_AMPLITUDE", "Sum_Elastic_Inelastic"}
)
FIT_DIAGNOSTIC_CHANNELS = frozenset({"Num_Iter", "Fit_Residual"})


def channel_role(name: str) -> ChannelRole:
    if name in OBSERVED_ELEMENTAL_OR_LINE_CHANNELS:
        return ChannelRole.ELEMENTAL_OR_LINE
    if name in FLUORESCENCE_CHANNELS:
        return ChannelRole.FLUORESCENCE
    if name in SCATTER_CHANNELS:
        return ChannelRole.SCATTER
    if name in FIT_DIAGNOSTIC_CHANNELS:
        return ChannelRole.FIT_DIAGNOSTIC
    return ChannelRole.AUXILIARY
