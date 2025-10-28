import numpy as np
import pandas as pd
from lightcurvelynx.astro_utils.zeropoint import calculate_zp_from_maglim
from lightcurvelynx.consts import GAUSS_EFF_AREA2FWHM_SQ
from lightcurvelynx.obstable.obs_table import ObsTable
from lightcurvelynx.obstable.obs_table_params import (
    FiveSigmaDepthDeriver,
    FullParamDeriver,
    NoopParamDeriver,
)


def test_noop_param_deriver():
    """Use the NoopParamDeriver object to fill in missing ObsTable parameters."""
    values = {
        "time": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "filter": np.array(["r", "g", "r", "i", "g"]),
    }
    pdf = pd.DataFrame(values)

    ops_data = ObsTable(pdf, seeing=0.5, pixel_scale=0.2, sky_bg_adu=100.0, gain=2.0)
    assert len(ops_data) == 5

    # The table contains only the information provided.
    given_keys = ["time", "ra", "dec", "filter", "seeing", "pixel_scale", "sky_bg_adu", "gain"]
    not_given = ["zp_per_band", "exptime", "nexposure", "zp", "psf_footprint", "fwhm_px", "sky_bg_electrons"]
    assert np.all([key in ops_data for key in given_keys])
    assert np.all([key not in ops_data for key in not_given])

    # We can derive additional parameters.
    deriver = NoopParamDeriver()
    deriver.derive_parameters(ops_data)

    # Original keys
    assert np.all([key in ops_data for key in given_keys])

    # No new keys should be added.
    assert np.all([key not in ops_data for key in not_given])


def test_full_param_deriver():
    """Use the FullParamDeriver object to fill in missing ObsTable parameters."""
    values = {
        "time": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "filter": np.array(["r", "g", "r", "i", "g"]),
    }
    pdf = pd.DataFrame(values)

    zp_per_band = {"g": 26.0, "r": 27.0, "i": 28.0}
    ops_data = ObsTable(pdf, zp_per_band=zp_per_band, seeing=0.5, pixel_scale=0.2, sky_bg_adu=100.0, gain=2.0)
    assert len(ops_data) == 5

    # The table contains only the information provided.
    given_keys = ["time", "ra", "dec", "filter", "zp_per_band", "seeing", "pixel_scale", "sky_bg_adu", "gain"]
    assert np.all([key in ops_data for key in given_keys])
    assert np.all([key not in ops_data for key in ["exptime", "nexposure", "zp", "psf_footprint", "fwhm_px"]])

    # We can derive additional parameters.
    deriver = FullParamDeriver()
    deriver.derive_parameters(ops_data)

    # Original keys
    assert np.all([key in ops_data for key in given_keys])

    # Derived keys (one step of derivation)
    assert "zp" in ops_data
    assert np.allclose(ops_data["zp"], np.array([27.0, 26.0, 27.0, 28.0, 26.0]))

    assert "sky_bg_electrons" in ops_data
    assert np.allclose(ops_data["sky_bg_electrons"], np.array([200.0, 200.0, 200.0, 200.0, 200.0]))

    assert "fwhm_px" in ops_data
    assert np.allclose(ops_data["fwhm_px"], np.array([2.5] * 5))

    # Derived keys (two steps of derivation)
    assert "psf_footprint" in ops_data
    assert np.allclose(ops_data["psf_footprint"], np.array([GAUSS_EFF_AREA2FWHM_SQ * (2.5) ** 2] * 5))


def test_full_param_deriver_zp():
    """Use the FullParamDeriver object to compute a non-trivial zero point."""
    values = {
        "time": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "filter": np.array(["r", "g", "r", "i", "g"]),
        "maglim": np.array([22.0] * 5),
    }
    pdf = pd.DataFrame(values)

    ops_data = ObsTable(
        pdf,
        maglim=22.0,
        sky_bg_electrons=1000.0,
        fwhm_px=3.0,
        read_noise=5.0,
        dark_current=0.01,
        nexposure=1,
        exptime=30.0,
    )
    assert len(ops_data) == 5

    # The table contains only the information provided.
    given_cols = ["time", "ra", "dec", "filter", "maglim"]
    assert np.all([col in ops_data for col in given_cols])
    given_params = ["sky_bg_electrons", "fwhm_px", "read_noise", "dark_current", "exptime", "nexposure"]
    assert np.all([param in ops_data for param in given_params])
    assert np.all([key not in ops_data for key in ["zp", "psf_footprint", "seeing"]])

    # We can derive additional parameters.
    deriver = FullParamDeriver()
    deriver.derive_parameters(ops_data)

    # Derived keys (one step of derivation)
    assert "zp" in ops_data
    expected_zp = calculate_zp_from_maglim(
        maglim=22.0,
        sky_bg_electrons=1000.0,
        fwhm_px=3.0,
        read_noise=5.0,
        dark_current=0.01,
        exptime=30.0,
        nexposure=1,
    )
    assert np.allclose(ops_data["zp"], expected_zp)


def test_five_sigma_depth_deriver():
    """Use the FiveSigmaDepthDeriver object to compute the five-sigma depth."""
    values = {
        "time": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "filter": np.array(["r", "g", "r", "i", "g"]),
        "five_sigma_depth": 20.0 + np.arange(5),
    }
    pdf = pd.DataFrame(values)

    bandflux_ref = {"g": 3630e6, "r": 3635e6, "i": 3625e6}
    ops_data = ObsTable(pdf, bandflux_ref=bandflux_ref)
    assert len(ops_data) == 5

    # The table contains only the information provided.
    given_data = ["time", "ra", "dec", "filter", "five_sigma_depth", "bandflux_ref"]
    assert np.all([col in ops_data for col in given_data])
    assert np.all([key not in ops_data for key in ["zp", "bandflux_error", "seeing"]])

    # We can derive additional parameters.
    deriver = FiveSigmaDepthDeriver()
    deriver.derive_parameters(ops_data)

    # Derived keys (one step of derivation)
    assert "bandflux_error" in ops_data
    bandflux_ref_arr = np.array([bandflux_ref[filt] for filt in ops_data["filter"]])
    expected_bandflux_error = (
        bandflux_ref_arr * np.power(10.0, -0.4 * ops_data["five_sigma_depth"].to_numpy()) / 5.0
    )
    assert np.allclose(ops_data["bandflux_error"], expected_bandflux_error)
