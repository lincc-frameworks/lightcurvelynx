"""Test the RedbackWrapperModel."""

import numpy as np
import pytest
from citation_compass import find_in_citations
from lightcurvelynx.math_nodes.given_sampler import GivenValueList
from lightcurvelynx.models.redback_models import RedbackWrapperModel


class ToySNModel:
    """A toy model that mimics the data structure of a RedbackTimeSeriesSource,
    but we can control.

    Attributes
    ----------
    self.height : float
        The height of the toy model peak.
    self.width : float
        The width of the toy model peak.
    """

    def __init__(self, height, width):
        self.height = height
        self.width = width

    def minwave(self):
        """Get the minimum wavelength of the model."""
        # The value of 5.0 is arbitrary. We just need something that provides
        # a tighter lower bound than the wrapper model.
        return 5.0

    def maxwave(self):
        """Get the maximum wavelength of the model."""
        return np.inf

    def get_flux_density(self, times, wavelengths):
        """A toy flux function that depends on time and wave.
        Peaks at t=0 and decreases with wave.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of rest frame timestamps.
        wavelengths : numpy.ndarray
            A length N array of wavelengths (in angstroms).

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of SED values (in nJy).
        """
        lightcurve = self.height * np.exp(-(times**2) / (self.width**2))
        flux_density = lightcurve[:, np.newaxis] * (1000.0 / wavelengths[np.newaxis, :])
        return flux_density


def _toy_redback_model(times, height, width, **kwargs):
    """Create and return the toy model."""
    return ToySNModel(height, width)


# Fake the appending of a citation to the model function.
_toy_redback_model.citation = "TEST_CITATION_2025"


def test_redback_models_toy() -> None:
    """Test that we can create and evaluate a simple model."""
    # Define static parameters.
    t0 = 64350.0
    parameters = {
        "height": 1000.0,
        "width": 10.0,
    }

    # Create the model.
    model = RedbackWrapperModel(
        _toy_redback_model,
        parameters=parameters,  # Set ALL the redback model parameters
        ra=0.0,  # Set other parameters
        dec=-10.0,
        t0=t0,
        node_label="source",
    )
    assert set(model.source_param_names) == {"height", "width"}

    state = model.sample_parameters()
    assert state["source"]["height"] == 1000.0
    assert state["source"]["width"] == 10.0
    assert state["source"]["t0"] == 64350.0

    times = np.array([-10.0, 0.5, 10.0]) + t0
    waves_ang = np.array([1000.0, 2000.0])
    fluxes = model.evaluate_sed(times, waves_ang, graph_state=state)

    # Check that the fluxes spike around t0.
    assert fluxes.shape == (3, 2)
    assert np.all(fluxes[0, :] < fluxes[1, :])
    assert np.all(fluxes[1, :] > fluxes[2, :])

    # Check that the fluxes are different at different wavelengths.
    assert np.all(fluxes[:, 0] != fluxes[:, 1])

    # Check that we can recover the citations
    rb_citations = find_in_citations("RedbackWrapperModel")
    assert len(rb_citations) >= 1
    for citation in rb_citations:
        assert "https://ui.adsabs.harvard.edu/abs/2024MNRAS.531.1203S/abstract" in citation
    rb_model_citations = find_in_citations("redback model")
    assert len(rb_model_citations) >= 1
    for citation in rb_model_citations:
        assert "TEST_CITATION_2025" in citation


def test_redback_models_fail_toy() -> None:
    """Test that we can create, but fail to evaluate a model if we don't have the parameters we need."""
    # Create a model with a missing parameter.
    t0 = 64350.0
    model = RedbackWrapperModel(
        _toy_redback_model,
        parameters={"height": 1000.0},  # Missing "width"
        t0=t0,
        node_label="source",
    )
    assert set(model.source_param_names) == {"height"}

    state = model.sample_parameters()
    times = np.array([1.0, 5.0, 6.5, 10.0])
    waves_ang = np.array([1000.0, 2000.0])
    with pytest.raises(TypeError):
        _ = model.evaluate_sed(times, waves_ang, graph_state=state)

    # Fail if we give it the same parameter in two different ways (repeat redshift).
    with pytest.raises(ValueError):
        _ = RedbackWrapperModel(
            "one_component_kilonova_model",
            parameters={"height": 1000.0, "width": 10.0, "redshift": 0.05},
            redshift=0.1,
            node_label="toy",
        )


def test_redback_models_chained_toy() -> None:
    """Test that we can create and evaluate a model with chained parameters."""
    t0 = 64350.0
    parameters = {
        "height": GivenValueList([500.0, 1000.0, 1500.0]),
        "width": 10.0,
    }

    # Create the model.
    model = RedbackWrapperModel(
        _toy_redback_model,
        parameters=parameters,  # Set ALL the redback model parameters
        t0=t0,
        node_label="source",
    )
    assert set(model.source_param_names) == {"height", "width"}

    state = model.sample_parameters(num_samples=3)
    times = np.array([t0 + 0.5])
    waves_ang = np.array([1000.0])
    fluxes = model.evaluate_sed(times, waves_ang, graph_state=state)

    # The returned fluxes should all be different since height is changing.
    assert fluxes.shape == (3, 1, 1)
    assert len(np.unique(fluxes)) == 3
