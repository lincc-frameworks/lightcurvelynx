import matplotlib

matplotlib.use("Agg")  # Suppress the plots

import numpy as np
import pytest
from lightcurvelynx.astro_utils.passbands import Passband, PassbandGroup
from lightcurvelynx.astro_utils.sed_basis_models import SEDBasisModel


def test_sed_basis_model() -> None:
    """Test that we can create and query a simple SEDBasisModel object."""
    # Create a simple SED basis model with three filters that have overlapping ranges.
    wavelengths = np.array([3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000], dtype=float)
    filter_data = {
        "u": np.array([0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
        "g": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.8, 0.8, 0.0, 0.0], dtype=float),
        "r": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.6, 0.6, 0.0], dtype=float),
    }
    sed_basis = SEDBasisModel(wavelengths, filter_data)
    assert len(sed_basis) == 3
    assert sed_basis.filters == ["u", "g", "r"]
    assert np.allclose(sed_basis.wavelengths, wavelengths)

    # Query the SED values for the bands at specific wavelengths.
    query_waves = np.array([3500, 4500, 5500, 6500, 7500, 8500, 9250])
    assert np.allclose(sed_basis.compute_sed("u", query_waves), [0.25, 0.5, 0.5, 0.25, 0.0, 0.0, 0.0])
    assert np.allclose(sed_basis.compute_sed("g", query_waves), [0.0, 0.0, 0.0, 0.0, 0.4, 0.8, 0.8])
    assert np.allclose(sed_basis.compute_sed("r", query_waves), [0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.6])

    # We fall back to all internal wavelengths if none are provided.
    assert np.allclose(sed_basis.compute_sed("u"), filter_data["u"])
    assert np.allclose(sed_basis.compute_sed("g"), filter_data["g"])
    assert np.allclose(sed_basis.compute_sed("r"), filter_data["r"])

    # We can compute the flux densities over time.
    flux_density = sed_basis.compute_sed_from_bandfluxes(
        ["r", "g", "g", "r"], bandfluxes=[2.0, 1.0, 1.5, 2.5]
    )
    expected = np.array(
        [
            sed_basis.compute_sed("r") * 2.0,
            sed_basis.compute_sed("g") * 1.0,
            sed_basis.compute_sed("g") * 1.5,
            sed_basis.compute_sed("r") * 2.5,
        ]
    )
    assert np.allclose(flux_density, expected)

    # SED creation fails if the lengths do not match.
    with pytest.raises(ValueError):
        _ = SEDBasisModel(np.array([300, 400, 500, 600, 700]), filter_data)

    # Test plot doesn't crash
    sed_basis.plot()


def test_create_from_passbands():
    """Test that we can create a SEDBasisModel from a PassbandGroup."""
    a_band = Passband(np.array([[4000, 0.5], [5000, 0.5], [6000, 0.5]]), "LSST", "u")
    b_band = Passband(np.array([[8000, 0.8], [9000, 0.8], [10000, 0.8]]), "LSST", "g")
    c_band = Passband(np.array([[9000, 0.6], [10000, 0.6], [11000, 0.6]]), "LSST", "r")
    passband_group = PassbandGroup(given_passbands=[a_band, b_band, c_band])
    sed_basis = SEDBasisModel.from_box_approximation(passband_group)

    assert np.allclose(sed_basis.wavelengths, passband_group.waves)
    assert sed_basis.filters == ["u", "g", "r"]

    # Check that no two SED basis functions overlap.
    for f1 in sed_basis.filters:
        for f2 in sed_basis.filters:
            if f1 != f2:
                assert np.count_nonzero(sed_basis.sed_values[f1] * sed_basis.sed_values[f2]) == 0

    # Check that they are normalized correctly.
    for filter_name in sed_basis.filters:
        sed = sed_basis.sed_values[filter_name]
        flux_density = np.array([sed])
        bandflux = passband_group.fluxes_to_bandflux(flux_density, filter_name)
        assert np.isclose(bandflux, 1.0)
