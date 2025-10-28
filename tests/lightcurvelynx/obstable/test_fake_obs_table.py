import numpy as np
import pandas as pd
import pytest
from lightcurvelynx.consts import GAUSS_EFF_AREA2FWHM_SQ
from lightcurvelynx.obstable.fake_obs_table import FakeObsTable


def test_create_fake_obs_table_consts():
    """Create a minimal FakeObsTable object with given defaults."""
    values = {
        "time": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "filter": np.array(["r", "g", "r", "i", "g"]),
    }
    pdf = pd.DataFrame(values)

    zp_per_band = {"g": 26.0, "r": 27.0, "i": 28.0}
    ops_data = FakeObsTable(
        pdf,
        noise_strategy="exhaustive",
        zp_per_band=zp_per_band,
        fwhm_px=2.0,
        sky_bg_electrons=100.0,
    )
    assert len(ops_data) == 5

    # We use the defaults when we do not provide values in the table. Not all of these
    # will be added as columns, but we can still retrieve an array of values.
    assert np.allclose(ops_data["ra"], values["ra"])
    assert np.allclose(ops_data["dec"], values["dec"])
    assert np.allclose(ops_data["time"], values["time"])
    assert np.array_equal(ops_data["filter"], values["filter"])
    assert np.allclose(ops_data["zp"], [27.0, 26.0, 27.0, 28.0, 26.0])
    assert np.allclose(ops_data.get_value_per_row("fwhm_px"), [2.0] * 5)
    assert np.allclose(ops_data.get_value_per_row("sky_bg_electrons"), [100.0] * 5)
    assert np.allclose(ops_data.get_value_per_row("exptime"), [30.0] * 5)
    assert np.allclose(ops_data.get_value_per_row("nexposure"), [1] * 5)

    # Derived from fwhm_px.
    assert np.allclose(
        ops_data.get_value_per_row("psf_footprint"),
        [GAUSS_EFF_AREA2FWHM_SQ * (2.0) ** 2] * 5,
    )

    assert ops_data.survey_values["dark_current"] == 0
    assert ops_data.survey_values["nexposure"] == 1
    assert ops_data.survey_values["sky_bg_electrons"] == 100
    assert ops_data.survey_values["radius"] is None
    assert ops_data.survey_values["read_noise"] == 0
    assert ops_data.survey_values["survey_name"] == "FAKE_SURVEY"

    # Successfully derived a zp column.
    assert "zp" in ops_data

    # We can compute noise.
    flux_error = ops_data.bandflux_error_point_source(
        np.array([100.0, 200.0, 300.0, 400.0, 500.0]),  # Fluxes in nJy
        np.arange(5),  # Indices of the observations
    )
    assert len(flux_error) == 5
    assert np.all(flux_error > 0)
    assert len(np.unique(flux_error)) > 1  # Not all the same

    # If we give psf_footprint, we use that instead of fwhm_px.
    ops_data = FakeObsTable(
        pdf,
        noise_strategy="exhaustive",
        zp_per_band=zp_per_band,
        fwhm_px=2.0,
        psf_footprint=1.0,
        sky_bg_electrons=100.0,
    )
    assert np.allclose(ops_data.get_value_per_row("psf_footprint"), [1.0] * 5)

    # We can override the defaults, using dictionaries of values for fwhm_px and sky.
    ops_data = FakeObsTable(
        pdf,
        noise_strategy="exhaustive",
        zp_per_band=zp_per_band,
        exptime=60.0,
        fwhm_px={"g": 2.5, "r": 3.1, "i": 1.9},
        nexposure=2,
        radius=1.0,
        read_noise=5.0,
        sky_bg_electrons={"g": 150.0, "r": 140.0, "i": 155.0},
        survey_name="MY_SURVEY",
    )
    assert len(ops_data) == 5

    # We use the defaults when we do not provide values in the table. Not all of these
    # will be added as columns, but we can still retrieve an array of values.
    assert np.allclose(ops_data["ra"], values["ra"])
    assert np.allclose(ops_data["dec"], values["dec"])
    assert np.allclose(ops_data["time"], values["time"])
    assert np.array_equal(ops_data["filter"], values["filter"])
    assert np.allclose(ops_data["zp"], [27.0, 26.0, 27.0, 28.0, 26.0])
    assert np.allclose(ops_data.get_value_per_row("fwhm_px"), [3.1, 2.5, 3.1, 1.9, 2.5])
    assert np.allclose(ops_data.get_value_per_row("sky_bg_electrons"), [140.0, 150.0, 140.0, 155.0, 150.0])
    assert np.allclose(ops_data.get_value_per_row("exptime"), [60.0] * 5)
    assert np.allclose(ops_data.get_value_per_row("nexposure"), [2] * 5)
    assert np.allclose(
        ops_data.get_value_per_row("psf_footprint"),
        [
            GAUSS_EFF_AREA2FWHM_SQ * (3.1) ** 2,
            GAUSS_EFF_AREA2FWHM_SQ * (2.5) ** 2,
            GAUSS_EFF_AREA2FWHM_SQ * (3.1) ** 2,
            GAUSS_EFF_AREA2FWHM_SQ * (1.9) ** 2,
            GAUSS_EFF_AREA2FWHM_SQ * (2.5) ** 2,
        ],
    )

    assert ops_data.survey_values["dark_current"] == 0
    assert ops_data.survey_values["nexposure"] == 2
    assert ops_data.survey_values["sky_bg_electrons"] == {"g": 150.0, "r": 140.0, "i": 155.0}
    assert ops_data.survey_values["radius"] == 1.0
    assert ops_data.survey_values["read_noise"] == 5.0
    assert ops_data.survey_values["survey_name"] == "MY_SURVEY"

    # We can compute noise.
    flux_error2 = ops_data.bandflux_error_point_source(
        np.array([100.0, 200.0, 300.0, 400.0, 500.0]),  # Fluxes in nJy
        np.arange(5),  # Indices of the observations
    )
    assert len(flux_error2) == 5
    assert np.all(flux_error2 > 0)
    assert len(np.unique(flux_error2)) > 1  # Not all the same
    assert np.any(flux_error2 != flux_error)  # Different from before

    # We fail if we use a deriver that cannot compute required parameters.
    with pytest.raises(ValueError):
        _ = FakeObsTable(
            pdf,
            noise_strategy="given_only",
            zp_per_band=zp_per_band,
            fwhm_px=2.0,
            psf_footprint=1.0,
            sky_bg_electrons=100.0,
        )


def test_create_fake_obs_table_non_consts():
    """Test that if we specify values in columns, we use those instead of the defaults."""
    values = {
        "time": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "filter": np.array(["r", "g", "r", "i", "g"]),
        "fwhm_px": np.array([2.0, 2.5, 3.0, 2.0, 1.5]),
        "sky_bg_electrons": np.array([100.0, 150.0, 200.0, 100.0, 50.0]),
        "exptime": np.array([20.0, 30.0, 40.0, 20.0, 10.0]),
        "nexposure": np.array([1, 2, 3, 1, 1]),
        "zp": np.array([10.0, 11.0, 12.0, 13.0, 14.0]),
    }
    pdf = pd.DataFrame(values)

    zp_per_band = {"g": 26.0, "r": 27.0, "i": 28.0}
    ops_data = FakeObsTable(
        pdf,
        noise_strategy="exhaustive",
        zp_per_band=zp_per_band,
        fwhm_px=2.0,
        sky=100.0,
    )
    assert len(ops_data) == 5
    assert np.allclose(ops_data["ra"], values["ra"])
    assert np.allclose(ops_data["dec"], values["dec"])
    assert np.allclose(ops_data["time"], values["time"])
    assert np.array_equal(ops_data["filter"], values["filter"])
    assert np.allclose(ops_data["fwhm_px"], values["fwhm_px"])
    assert np.allclose(ops_data["sky_bg_electrons"], values["sky_bg_electrons"])
    assert np.allclose(ops_data["exptime"], values["exptime"])
    assert np.allclose(ops_data["nexposure"], values["nexposure"])
    assert np.allclose(ops_data["zp"], values["zp"])

    # We can compute noise.
    flux_error = ops_data.bandflux_error_point_source(
        np.array([100.0, 200.0, 300.0, 400.0, 500.0]),  # Fluxes in nJy
        np.arange(5),  # Indices of the observations
    )
    assert len(flux_error) == 5
    assert np.all(flux_error > 0)
    assert len(np.unique(flux_error)) > 1  # Not all the same


def test_create_fake_obs_table_cols_fail():
    """Test that we raise errors when we do not provide required values."""
    values = {
        "time": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "filter": np.array(["r", "g", "r", "i", "g"]),
    }
    pdf = pd.DataFrame(values)

    # Missing fwhm_px.
    zp_per_band = None
    with pytest.raises(ValueError):
        _ = FakeObsTable(
            pdf,
            noise_strategy="exhaustive",
            zp_per_band=zp_per_band,
            sky_bg_electrons=100.0,
        )

    # Missing sky_bg_electrons.
    with pytest.raises(ValueError):
        _ = FakeObsTable(
            pdf,
            noise_strategy="exhaustive",
            zp_per_band=zp_per_band,
            fwhm_px=2.0,
        )

    # Missing or invalid exptime.
    with pytest.raises(ValueError):
        _ = FakeObsTable(
            pdf,
            noise_strategy="exhaustive",
            zp_per_band=zp_per_band,
            fwhm_px=2.0,
            sky_bg_electrons=100.0,
            exptime=None,
        )
    with pytest.raises(ValueError):
        _ = FakeObsTable(
            pdf,
            noise_strategy="exhaustive",
            zp_per_band=zp_per_band,
            fwhm_px=2.0,
            sky_bg_electrons=100.0,
            exptime=-10.0,
        )

    # Missing or invalid nexposure.
    with pytest.raises(ValueError):
        _ = FakeObsTable(
            pdf,
            noise_strategy="exhaustive",
            zp_per_band=zp_per_band,
            fwhm_px=2.0,
            sky_bg_electrons=100.0,
            nexposure=None,
        )
    with pytest.raises(ValueError):
        _ = FakeObsTable(
            pdf,
            noise_strategy="exhaustive",
            zp_per_band=zp_per_band,
            fwhm_px=2.0,
            sky_bg_electrons=100.0,
            nexposure=-1,
        )


def test_create_fake_obs_table_zp_fail():
    """Test that we raise errors when we do not provide required values."""
    values = {
        "time": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
    }
    pdf = pd.DataFrame(values)

    # No filters from which to computer zp.
    zp_per_band = {"g": 26.0, "r": 27.0, "i": 28.0}
    with pytest.raises(KeyError):
        _ = FakeObsTable(
            pdf,
            noise_strategy="exhaustive",
            zp_per_band=zp_per_band,
            fwhm_px=2.0,
            sky=100.0,
        )

    # Mismatched filters.
    values["filter"] = np.array(["r", "g", "r", "i", "z"])
    pdf = pd.DataFrame(values)

    zp_per_band = {"g": 26.0, "r": 27.0}
    with pytest.raises(ValueError):
        _ = FakeObsTable(
            pdf,
            noise_strategy="exhaustive",
            zp_per_band=zp_per_band,
            fwhm_px=2.0,
            sky=100.0,
        )


def test_create_fake_obs_table_noise_free():
    """Create a noise free FakeObsTable object."""
    values = {
        "time": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "filter": np.array(["r", "g", "r", "i", "g"]),
    }
    pdf = pd.DataFrame(values)

    ops_data = FakeObsTable(pdf, const_flux_error=0.0)
    assert len(ops_data) == 5

    # Compute the bandflux error for each observation (should all be zero).
    fluxes = np.array([100.0, 200.0, 300.0, 400.0, 500.0])  # Fluxes in nJy
    flux_error = ops_data.bandflux_error_point_source(fluxes, np.arange(5))
    assert np.array_equal(flux_error, np.zeros(5))
