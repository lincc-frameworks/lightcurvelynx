"""Test the PyLIMAWrapperModel. Note this test only runs if pylima is installed
(which is it not by default)."""

from importlib.util import find_spec

import numpy as np
import pytest

from lightcurvelynx.models.pylima_models import PyLIMAWrapperModel

if find_spec("pyLIMA") is None:
    pytest.skip("pyLIMA not installed, skipping tests.", allow_module_level=True)  # type: ignore


def test_pylima_create():
    """Test that we can create and query a PyLIMA model."""
    source_mags = {
        "g": 22.0 + np.random.normal(scale=0.3),
        "r": 21.5 + np.random.normal(scale=0.3),
        "i": 21.2 + np.random.normal(scale=0.3),
    }
    # No i in blend mags. It gets defaulted to very faint.
    blend_mags = {
        "g": 24.5,
        "r": 24.0,
    }
    pylima_params = {
        "u0": 0.01,
        "tE": 25.0,
        "piEN": 0.1,
        "piEE": 0.1,
    }

    model = PyLIMAWrapperModel(
        "PSPL",
        source_mags=source_mags,
        blend_mags=blend_mags,
        ra=1.0,
        dec=-20.0,
        t0=64350.0,
        pylima_params=pylima_params,
        blend_flux_parameter="fblend",
        node_label="source",
    )

    state = model.sample_parameters()
    assert state["source"]["u0"] == pytest.approx(0.01)
    assert state["source"]["tE"] == pytest.approx(25.0)
    assert state["source"]["piEN"] == pytest.approx(0.1)
    assert state["source"]["piEE"] == pytest.approx(0.1)
    for band in source_mags:
        assert isinstance(state["source"][f"fsource_{band}"], float)
        assert isinstance(state["source"][f"fblend_{band}"], float)

    query_times = 64350.0 + np.arange(-50, 50, 1)
    query_filters = np.array(["g"] * len(query_times))
    fluxes = model.evaluate_bandfluxes(None, query_times, query_filters, state)
    assert np.all(np.diff(fluxes[0:49]) > 0.0)  # Rising
    assert np.all(np.diff(fluxes[51:]) < 0.0)  # Falling


def test_pylima_fail_create():
    """Test that we can create and query a PyLIMA model."""
    source_mags = {
        "g": 22.0 + np.random.normal(scale=0.3),
        "r": 21.5 + np.random.normal(scale=0.3),
        "i": 21.2 + np.random.normal(scale=0.3),
    }
    # tE is missing
    pylima_params = {
        "u0": 0.01,
        "piEN": 0.1,
        "piEE": 0.1,
    }

    with pytest.raises(ValueError):
        _ = PyLIMAWrapperModel(
            "PSPL",
            source_mags=source_mags,
            blend_mags={},
            ra=1.0,
            dec=-20.0,
            t0=64350.0,
            pylima_params=pylima_params,
            blend_flux_parameter="fblend",
            node_label="source",
        )


def test_pylima_blends():
    """Test that we can create PyLIMA models with different blend parameters."""
    from lightcurvelynx.astro_utils.mag_flux import mag2flux

    source_mags = {
        "g": 22.0 + np.random.normal(scale=0.3),
        "r": 21.5 + np.random.normal(scale=0.3),
        "i": 21.2 + np.random.normal(scale=0.3),
    }
    # No i in blend mags. It gets defaulted to very faint.
    blend_mags = {
        "g": 24.5,
        "r": 24.0,
    }
    pylima_params = {
        "u0": 0.01,
        "tE": 25.0,
    }

    for blend_flux_parameter in ["ftotal", "gblend", "fblend"]:
        model = PyLIMAWrapperModel(
            "PSPL",
            source_mags=source_mags,
            blend_mags=blend_mags,
            ra=1.0,
            dec=-20.0,
            t0=64350.0,
            pylima_params=pylima_params,
            blend_flux_parameter=blend_flux_parameter,
            node_label="source",
        )

        state = model.sample_parameters()
        for band in source_mags:
            source_flux = mag2flux(source_mags[band])
            blend_flux = mag2flux(blend_mags[band]) if band in blend_mags else 0.0
            assert isinstance(state["source"][f"fsource_{band}"], float)
            assert isinstance(state["source"][f"{blend_flux_parameter}_{band}"], float)
            if blend_flux_parameter == "fblend":
                model_blend_flux = blend_flux
            elif blend_flux_parameter == "ftotal":
                model_blend_flux = source_flux + blend_flux
            elif blend_flux_parameter == "gblend":
                model_blend_flux = blend_flux / source_flux
            received_blend_flux = state["source"][f"{blend_flux_parameter}_{band}"]
            assert pytest.approx(np.abs(received_blend_flux - model_blend_flux), 0.66) == 0.0
