from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from astropy import units as u
from sncosmo.models import StretchSource, get_source

from lightcurvelynx import _LIGHTCURVELYNX_TEST_DATA_DIR
from lightcurvelynx.astro_utils.unit_utils import fnu_to_flam
from lightcurvelynx.math_nodes.np_random import NumpyRandomFunc
from lightcurvelynx.models.sncosmo_models import SncosmoWrapperModel
from lightcurvelynx.utils.extrapolate import ConstantPadding, ExponentialDecay


def _fake_nugent_data_path(*args, **kwargs):
    """A function for pointing to the test data directory's version of the nugent model file."""
    return str(
        Path(_LIGHTCURVELYNX_TEST_DATA_DIR) / "fake_sncosmo" / "models" / "nugent" / "sn1a_flux.v1.2.dat"
    )


def test_sncosmo_models_hsiao() -> None:
    """Test that we can create and evalue a 'hsiao' model."""
    model = SncosmoWrapperModel("hsiao", t0=0.0, amplitude=2.0e10)
    state = model.sample_parameters()
    assert model.get_param(state, "amplitude") == 2.0e10
    assert model.get_param(state, "t0") == 0.0
    assert str(model) == "SncosmoWrapperModel_0"
    assert np.array_equal(model.param_names, ["amplitude"])

    # Test against a manual query from sncosmo with no redshift and an
    # unrealistically high amplitude.
    #     model = sncosmo.Model(source='hsiao')
    #     model.set(z=0.0, t0=0.0, amplitude=2.0e10)
    #     model.flux(5., [4000., 4100., 4200.])
    fluxes_fnu = model.evaluate_sed([5.0], [4000.0, 4100.0, 4200.0])
    fluxes_flam = fnu_to_flam(
        fluxes_fnu,
        [4000.0, 4100.0, 4200.0],
        wave_unit=u.AA,
        flam_unit=u.erg / u.second / u.cm**2 / u.AA,
        fnu_unit=u.nJy,
    )
    assert np.allclose(fluxes_flam, [133.98143039, 152.74613574, 134.40916824])


def test_sncosmo_models_hsiao_t0() -> None:
    """Test that we can create and evalue a 'hsiao' model with a t0."""
    model = SncosmoWrapperModel("hsiao", t0=55000.0, amplitude=2.0e10)
    state = model.sample_parameters()
    assert model.get_param(state, "amplitude") == 2.0e10
    assert model.get_param(state, "t0") == 55000.0
    assert np.array_equal(model.param_names, ["amplitude"])

    # Test against a manual query from sncosmo with no redshift and an unrealistically high amplitude.
    #     model = sncosmo.Model(source='hsiao')
    #     model.set(z=0.0, t0=55000., amplitude=2.0e10)
    #     model.flux(54990., [4000., 4100., 4200.])
    fluxes_fnu = model.evaluate_sed([54990.0], [4000.0, 4100.0, 4200.0])
    fluxes_flam = fnu_to_flam(
        fluxes_fnu,
        [4000.0, 4100.0, 4200.0],
        wave_unit=u.AA,
        flam_unit=u.erg / u.second / u.cm**2 / u.AA,
        fnu_unit=u.nJy,
    )
    assert np.allclose(fluxes_flam, [67.83696271, 67.98471119, 47.20395186])

    # We raise a warning if the time is outside the bounds.
    with pytest.warns(UserWarning):
        model.evaluate_sed([0.0], [4000.0, 4100.0])


def test_sncosmo_models_hsiao_extrap_time() -> None:
    """Test that we can extrapolate to times outside the model bounds."""
    time_extrapolation = ExponentialDecay(rate=0.1)
    model = SncosmoWrapperModel(
        "hsiao",
        t0=55000.0,
        amplitude=2.0e10,
        time_extrapolation=time_extrapolation,
    )

    query_times = np.array([54500.0, 54900.0, 54960.0, 54990.0, 55000.0, 55010.0, 55100.0, 55620.0, 99990.0])
    fluxes = model.evaluate_sed(query_times, [4000.0])

    # All the times before are monotonically increasing to the first valid time.
    assert np.all(fluxes[0:3] < fluxes[3])
    assert np.all(np.diff(fluxes[0:3]) >= 0.0)
    # All the times after are monotonically decreasing from the last valid time.

    assert np.all(fluxes[7:9] < fluxes[6])
    assert np.all(np.diff(fluxes[5:9]) <= 0.0)


def test_sncosmo_models_bounds() -> None:
    """Test that we do not crash if we give wavelengths outside the model bounds."""
    # Use a massively subsampled version of the 'nugent-sn1a' model (only 3 time steps)
    # that is cached in the test data directory. This is okay because we are only using
    # the model's wavelength bounds.
    with patch("sncosmo.utils.DataMirror.abspath", side_effect=_fake_nugent_data_path):
        model = SncosmoWrapperModel("nugent-sn1a", amplitude=2.0e10, t0=54990.0)
    min_w = model.source.minwave()
    max_w = model.source.maxwave()

    wavelengths = [
        min_w - 100.0,  # Out of bounds
        min_w,  # edge of bounds (included)
        0.5 * min_w + 0.5 * max_w,  # included
        max_w,  # edge of bounds (included)
        max_w + 0.1,  # Out of bounds
    ]

    # SNCosmo raises an error if we give wavelengths outside the bounds. It should
    # raise a warning first.
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError):
            _ = model.evaluate_sed([54990.0, 54990.5], wavelengths)


def test_sncosmo_models_linear_extrapolate() -> None:
    """Test that we do not crash if we give wavelengths outside the model bounds."""
    # Use a massively subsampled version of the 'nugent-sn1a' model (only 3 time steps)
    # that is cached in the test data directory. This is okay because we are only using
    # the model's wavelength bounds.
    with patch("sncosmo.utils.DataMirror.abspath", side_effect=_fake_nugent_data_path):
        model = SncosmoWrapperModel(
            "nugent-sn1a",
            amplitude=2.0e10,
            t0=54990.0,
            wave_extrapolation=ExponentialDecay(rate=0.1),
        )
    min_w = model.source.minwave()
    max_w = model.source.maxwave()

    wavelengths = [
        min_w - 100.0,  # Out of bounds
        min_w,  # edge of bounds (included)
        0.5 * min_w + 0.5 * max_w,  # included
        max_w,  # edge of bounds (included)
        max_w + 0.1,  # Out of bounds
        max_w + 500.0,  # Out of bounds
        max_w + 5000.0,  # Out of bounds
    ]

    # Check that columns 0, 4, 5, and 6 are correctly extrapolated.
    fluxes_fnu = model.evaluate_sed([54990.0, 54990.5], wavelengths)
    assert np.all(fluxes_fnu[:, 0:6] > 0.0)
    assert np.all(fluxes_fnu[:, 0] < fluxes_fnu[:, 1])
    assert np.all(fluxes_fnu[:, 4] < fluxes_fnu[:, 3])
    assert np.all(fluxes_fnu[:, 5] < fluxes_fnu[:, 4])
    assert np.all(fluxes_fnu[:, 6] < fluxes_fnu[:, 5])


def test_sncosmo_models_chained() -> None:
    """Test that we can create and evalue a 'hsiao' model using randomized parameters."""
    # Generate the amplitude from a uniform distribution, but use a fixed seed so we have
    # reproducible tests.
    model = SncosmoWrapperModel(
        "hsiao",
        amplitude=NumpyRandomFunc("uniform", low=2.0, high=12.0),
        node_label="model",
    )
    rng = np.random.default_rng(100)
    state = model.sample_parameters(rng_info=rng)
    assert np.array_equal(model.param_names, ["amplitude"])
    assert 2.0 <= state["model"]["amplitude"] <= 12.0

    # When we resample, we get different model parameters. Create a histogram
    # of 10,000 samples.
    state = model.sample_parameters(num_samples=10_000, rng_info=rng)
    bins = np.floor(state["model"]["amplitude"] - 2.0).astype(int)
    hist = np.bincount(bins, minlength=10)

    # Each bin should have around 1,000 samples.
    for i in range(10):
        assert 800 < hist[i] < 1200


def test_sncosmo_dynamic_bounds():
    """Test that the bounds on SncosmoWrapperModel work correctly."""
    # Use the source information to create a new StretchSource. Save the phase bounds.
    org_source = get_source("hsiao")
    min_phase = org_source.minphase()
    max_phase = org_source.maxphase()
    phases = np.linspace(min_phase, max_phase, 100)
    waves = np.linspace(org_source.minwave(), org_source.maxwave(), 5)
    fluxes = org_source.flux(phases[:, None], waves[None, :])
    custom_source = StretchSource(phase=phases, wave=waves, flux=fluxes, name="custom")

    # Create a Sncosmo model using the custom StretchSource.
    # Sample the stretch parameter (s) uniformly.
    model = SncosmoWrapperModel(
        custom_source,
        t0=55000.0,
        amplitude=1000.0,
        s=NumpyRandomFunc("uniform", low=0.5, high=2.0),
        time_extrapolation=ConstantPadding(0.0),
        node_label="model",
    )

    # Check that the bounds change with the sampled stretch parameter.
    all_times = np.linspace(min_phase - 100, max_phase + 100, 10)
    states = model.sample_parameters(num_samples=100, rng_info=np.random.default_rng(100))
    for ind_state in states:
        curr_minpahse = model.minphase(graph_state=ind_state)
        curr_maxphase = model.maxphase(graph_state=ind_state)
        if ind_state["model.s"] > 1.0:
            assert curr_minpahse < min_phase
            assert curr_maxphase > max_phase
        elif ind_state["model.s"] < 1.0:
            assert curr_minpahse > min_phase
            assert curr_maxphase < max_phase
        else:
            assert curr_minpahse == pytest.approx(min_phase)
            assert curr_maxphase == pytest.approx(max_phase)

        # Compute the flux over an extend time range. The values inside the phase
        # bounds should be small while the outside should be 10000.
        flux = model.evaluate_sed(all_times, waves, ind_state)
        mask = (all_times < curr_minpahse) | (all_times > curr_maxphase)
        assert np.all(flux[~mask, :] < 100.0)
        assert np.all(flux[mask, :] == 0.0)
