import numpy as np
from lightcurvelynx.astro_utils.mag_flux import flux2mag
from lightcurvelynx.astro_utils.zeropoint import (
    calculate_zp_from_maglim,
    flux_electron_zeropoint,
    magnitude_electron_zeropoint,
    sky_bg_adu_to_electrons,
)
from lightcurvelynx.obstable.opsim import LSSTCAM_PIXEL_SCALE, _lsstcam_dark_current, _lsstcam_readout_noise
from scipy.optimize import fsolve


def test_magnitude_electron_zeropoint():
    """Test that instrumental zeropoints are correct"""
    # Reproducing magnitude corresponding to S/N=5 from
    # https://smtn-002.lsst.io/v/OPSIM-1171/index.html
    fwhm_eff = {
        "u": 0.92,
        "g": 0.87,
        "r": 0.83,
        "i": 0.80,
        "z": 0.78,
        "y": 0.76,
    }
    fwhm_eff_getter = np.vectorize(fwhm_eff.get)
    sky_brightness = {
        "u": 23.05,
        "g": 22.25,
        "r": 21.20,
        "i": 20.46,
        "z": 19.61,
        "y": 18.60,
    }
    sky_brightness_getter = np.vectorize(sky_brightness.get)

    # Use the LSST extinction coefficients and zeropoints per second at zenith.
    extinction_coeff = {
        "u": -0.458,
        "g": -0.208,
        "r": -0.122,
        "i": -0.074,
        "z": -0.057,
        "y": -0.095,
    }
    zp_per_sec_zenith = {
        "u": 26.524,
        "g": 28.508,
        "r": 28.361,
        "i": 28.171,
        "z": 27.782,
        "y": 26.818,
    }

    bands = list(fwhm_eff.keys())
    exptime = 30
    airmass = 1
    s2n = 5
    zp = magnitude_electron_zeropoint(
        filter=bands,
        airmass=airmass,
        exptime=exptime,
        instr_zp_mag=zp_per_sec_zenith,
        ext_coeff=extinction_coeff,
    )
    sky_count_per_arcsec_sq = np.power(10.0, -0.4 * (sky_brightness_getter(bands) - zp))
    readout_per_arcsec_sq = _lsstcam_readout_noise**2 / LSSTCAM_PIXEL_SCALE**2
    dark_per_arcsec_sq = _lsstcam_dark_current * exptime / LSSTCAM_PIXEL_SCALE**2
    count_per_arcsec_sq = sky_count_per_arcsec_sq + readout_per_arcsec_sq + dark_per_arcsec_sq

    area = 2.266 * fwhm_eff_getter(bands) ** 2  # effective seeing area in arcsec^2
    n_background = count_per_arcsec_sq * area
    # sky-dominated regime would be n_signal = s2n * np.sqrt(n_background)
    n_signal = 0.5 * (s2n**2 + np.sqrt(s2n**4 + 4 * s2n**2 * n_background))
    mag_signal = zp - 2.5 * np.log10(n_signal)

    m5_desired = {
        "u": 23.70,
        "g": 24.97,
        "r": 24.52,
        "i": 24.13,
        "z": 23.56,
        "y": 22.55,
    }
    assert list(m5_desired) == bands
    m5_desired_getter = np.vectorize(m5_desired.get)

    np.testing.assert_allclose(mag_signal, m5_desired_getter(bands), atol=0.1)

    # Test that the same result is obtained if we pass in the full arrays.
    # The arrays must be in the same order as the bands.
    extinction_coeff_array = np.array([extinction_coeff[b] for b in bands])
    zp_per_sec_zenith_array = np.array([zp_per_sec_zenith[b] for b in bands])
    zp2 = magnitude_electron_zeropoint(
        filter=bands,
        airmass=airmass,
        exptime=exptime,
        instr_zp_mag=zp_per_sec_zenith_array,
        ext_coeff=extinction_coeff_array,
    )
    np.testing.assert_allclose(zp, zp2, rtol=1e-10)


def test_flux_electron_zeropoint():
    """Test that bandflux zeropoints are correct"""
    extinction_coeff = {
        "u": -0.458,
        "g": -0.208,
        "r": -0.122,
        "i": -0.074,
        "z": -0.057,
        "y": -0.095,
    }
    zp_per_sec_zenith = {
        "u": 26.524,
        "g": 28.508,
        "r": 28.361,
        "i": 28.171,
        "z": 27.782,
        "y": 26.818,
    }

    # Here we just check that magnitude-flux conversion is correct
    airmass = np.array([1, 1.5, 2]).reshape(-1, 1, 1)
    exptime = np.array([30, 38, 45]).reshape(1, -1, 1)
    bands = ["u", "g", "r", "i", "z", "y"]
    mag = magnitude_electron_zeropoint(
        filter=bands,
        airmass=airmass,
        exptime=exptime,
        instr_zp_mag=zp_per_sec_zenith,
        ext_coeff=extinction_coeff,
    )
    flux = flux_electron_zeropoint(
        filter=bands,
        airmass=airmass,
        exptime=exptime,
        instr_zp_mag=zp_per_sec_zenith,
        ext_coeff=extinction_coeff,
    )
    np.testing.assert_allclose(mag, flux2mag(flux), rtol=1e-10)

    # Test that the same result is obtained if we pass in the full arrays.
    # The arrays must be in the same order as the bands.
    extinction_coeff_array = np.array([extinction_coeff[b] for b in bands])
    zp_per_sec_zenith_array = np.array([zp_per_sec_zenith[b] for b in bands])
    flux2 = flux_electron_zeropoint(
        filter=bands,
        airmass=airmass,
        exptime=exptime,
        instr_zp_mag=zp_per_sec_zenith_array,
        ext_coeff=extinction_coeff_array,
    )
    np.testing.assert_allclose(flux, flux2, rtol=1e-10)


def _fluxeq(
    flux, sky=None, gain=None, read_noise=None, nexposure=1, fwhm_px=None, dark_current=None, exptime=None
):
    """Define the equation to solve for flux at 5-sigma limit"""

    npix = 2.266 * fwhm_px**2  # = 4 * pi * sigma**2 = pi/2/ln2 * FWHM**2
    y = (
        flux**2
        - 25 * flux
        - 25
        * (sky * npix * gain + read_noise**2 * nexposure * npix + dark_current * npix * exptime * nexposure)
    )
    return y


def test_zp_from_maglim():
    """Test the calculation with numerical solution to the maglim function."""
    maglim = 19.0
    sky = 100.0
    fwhm = 1.5
    gain = 1.0
    read_noise = 10.0
    dark_current = 0.1
    exptime = 30.0

    root = fsolve(
        lambda x: _fluxeq(
            x,
            sky=sky,
            gain=gain,
            read_noise=read_noise,
            fwhm_px=fwhm,
            dark_current=dark_current,
            exptime=exptime,
        ),
        [100],
    )
    zp_expected = 2.5 * np.log10(root) + maglim

    zp_cal = calculate_zp_from_maglim(
        maglim=maglim,
        sky_bg_electrons=sky_bg_adu_to_electrons(sky, gain),
        fwhm_px=fwhm,
        read_noise=read_noise,
        dark_current=dark_current,
        exptime=exptime,
        nexposure=1,
    )

    assert np.isclose(zp_cal, zp_expected)
