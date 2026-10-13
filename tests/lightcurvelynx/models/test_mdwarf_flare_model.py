"""Tests for the m dwarf flare model"""

import numpy as np
from astropy import units as u

from lightcurvelynx.models.mdwarf_flare_model import MDwarfFlareModel

# test values with known output. first, no sampling

wavelengths = np.linspace(3000, 12000, 4000)
times = np.linspace(0, 20, 201)  # want peak time to actually be in times


def check_if_monotonic(values_array):
    """
    checks whether an array of values is monotonic

    Parameters
    ----------
    values_array : np.nparray
        the array of values to check

    Returns Bool
    """
    return np.all(np.diff(values_array) > 0) | np.all(np.diff(values_array) < 0)


def test_mdwarf_flare_model_with_assigned_inputs():
    """
    Test the functions involved in creating and sampling an MDwarfFlareModel object.

    Using specific inputs and making sure the values are what we expect
    (Not sampling from parameter distributions)
    """
    star_temp = 4000
    star_radius = 0.09 * u.R_sun.to(u.cm)
    flare_fwhm = 0.1
    flare_amplitude = 0.03
    t0 = 10

    model = MDwarfFlareModel(
        star_temp=star_temp,
        star_radius=star_radius,
        flare_fwhm=flare_fwhm,
        flare_amplitude=flare_amplitude,
        t0=t0,
        distance=100 * u.parsec,
    )

    assert np.isclose(
        model._quiescent_flux_no_distance(star_temp, wavelengths, for_tess=False)[0].value,
        9.133526742251799e-08,
    )
    assert np.isclose(
        model._quiescent_flux_no_distance(star_temp, wavelengths=None, for_tess=True)[0].value,
        4.24870522231194e-06,
    )

    assert np.isclose(
        model._build_spectrum_bb_with_balmer(wavelengths, for_tess=False)[0].value, 0.00014342158215047032
    )
    assert np.isclose(
        model._build_spectrum_bb_with_balmer(None, for_tess=True)[0].value, 0.00013805179319227408
    )

    I_lam_star_for_tess = model._quiescent_flux_no_distance(star_temp, wavelengths=None, for_tess=True)
    assert np.isclose(model._tess_band_integrate(I_lam_star_for_tess), 8.070721970058228e-06)

    I_lam_flare_for_tess = model._build_spectrum_bb_with_balmer(None, temp_low=9000, for_tess=True)
    assert np.isclose(model._tess_band_integrate(I_lam_flare_for_tess), 0.0001229577732360932)

    sample = model.sample_parameters()
    values = model.evaluate_sed(times, wavelengths, sample)
    assert check_if_monotonic(values[0])  # should be no flare
    # that may not be a good check if the wavelength range is crazy but should work for normal ranges
    assert not check_if_monotonic(values[100])  # when flare should exist

    # check that the peak amp of the flare is the same as the amplitude that we put in

    norm_shape = model._norm_flare_shape(times, t0, flare_fwhm) * flare_amplitude
    I_flare_full = model._build_spectrum_bb_with_balmer(wavelengths, for_tess=False)
    I_star_full = model._quiescent_flux_no_distance(star_temp, wavelengths, for_tess=False)
    integral_flare = model._tess_band_integrate(I_lam_flare_for_tess)
    integral_star = model._tess_band_integrate(I_lam_star_for_tess)
    constants = np.pi * star_radius**2 * u.sr * u.cm**2

    flux_flare_no_distance = (
        constants * (integral_star / integral_flare) * norm_shape[None, :] * I_flare_full[:, None]
    )
    q_no_distance = I_star_full * constants

    idx_peak = np.argmin(np.abs(times - t0))
    peak_flare_tess = np.trapezoid(flux_flare_no_distance[:, idx_peak])
    star_tess = np.trapezoid(q_no_distance)

    assert np.isclose(peak_flare_tess / star_tess, flare_amplitude, atol=0.005)

    # test that compute_sed runs/has the shape we expect
    state = model.sample_parameters(num_samples=1)
    fluxes = model.evaluate_sed(times, wavelengths, state)

    assert fluxes.shape == (201, 4000)


def test_mdwarf_flare_model_parameter_sampling():
    """Test the sampling from parameter distributions for MDwarfFlareModel objects."""

    model = MDwarfFlareModel(t0=10, node_label="model")
    state = model.sample_parameters(num_samples=1)
    assert state["mw"]["distance_pc"] > 0
    assert state["mw"]["ra"] >= 0
    assert state["mw"]["ra"] <= 360
    assert state["mw"]["dec"] >= -90
    assert state["mw"]["dec"] <= 90
    # star temp
    assert state["model"]["star_temp"] > 0
    # star radius - to make sure it's not in solar radii i assert >100
    assert state["model"]["star_radius"] > 100
    # flare_fwhm
    assert state["model"]["flare_fwhm"] > 0
    # flare_amp
    assert state["model"]["flare_amplitude"] > 0
