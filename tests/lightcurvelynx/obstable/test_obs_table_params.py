import numpy as np
import pandas as pd
from lightcurvelynx.astro_utils.zeropoint import flux_electron_zeropoint
from lightcurvelynx.consts import GAUSS_EFF_AREA2FWHM_SQ
from lightcurvelynx.obstable.obs_table import ObsTable
from lightcurvelynx.obstable.obs_table_params import _ParamDeriver


def test_param_deriver():
    """Use the _ParamDeriver object to fill in missing ObsTable parameters."""
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
    deriver = _ParamDeriver()
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


def test_param_deriver_zp():
    """Use the _ParamDeriver object to compute a non-trivial zero point."""
    values = {
        "time": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "filter": np.array(["r", "g", "r", "i", "g"]),
    }
    pdf = pd.DataFrame(values)

    instr_zp_mag = {"g": 26.0, "r": 27.0, "i": 28.0}
    ops_data = ObsTable(pdf, instr_zp_mag=instr_zp_mag, airmass=0.01, ext_coeff=0.1, exptime=30.0)
    assert len(ops_data) == 5

    # The table contains only the information provided.
    assert np.all(
        [
            key in ops_data
            for key in ["time", "ra", "dec", "filter", "instr_zp_mag", "airmass", "ext_coeff", "exptime"]
        ]
    )
    assert np.all([key not in ops_data for key in ["nexposure", "zp", "psf_footprint", "seeing"]])

    # We can derive additional parameters.
    deriver = _ParamDeriver()
    deriver.derive_parameters(ops_data)

    # Derived keys (one step of derivation)
    assert "zp" in ops_data
    expected_zp = flux_electron_zeropoint(
        instr_zp_mag=instr_zp_mag,
        ext_coeff=0.1,
        filter=ops_data["filter"],
        airmass=0.01,
        exptime=30.0,
    )
    assert np.allclose(ops_data["zp"], expected_zp)
