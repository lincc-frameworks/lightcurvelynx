import numpy as np
from lightcurvelynx.astro_utils.mag_flux import mag2flux
from lightcurvelynx.models.bagle_models import BagleWrapperModel


class _DummyBagleModel:
    """A dummy bagle model for testing purposes.

    Attributes
    ----------
    scale : float
        A scaling factor for the fluxes.
    filter_mags : list
        A list of magnitudes for each filter.
    """

    def __init__(self, scale, filter_mags):
        self.scale = scale
        self.filter_mags = filter_mags

    def get_photometry(self, times, filter_idx):
        """Compute dummy fluxes based on parameters and times.

        Parameters
        ----------
        times : numpy.ndarray
            An array of times.
        filter_idx : int
            The index of the filter.
        """
        return np.ones_like(times) * self.scale * self.filter_mags[filter_idx]


def test_bagle_wrapper_model() -> None:
    """Test that we can create and query a BagleWrapperModel object."""
    parameter_dict = {
        "scale": 2.0,
        "filter_mags": [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],  # ugrizy
    }
    model = BagleWrapperModel(_DummyBagleModel, parameter_dict)

    # Check that we have the given parameters.
    all_params = model.list_params()
    assert "scale" in all_params
    assert "filter_mags" in all_params
    assert model.parameter_names == ["scale", "filter_mags"]

    # Check that we have the standard physical model parameters.
    assert "ra" in all_params
    assert "dec" in all_params
    assert "t0" in all_params
    assert "redshift" in all_params
    assert "distance" in all_params

    # Test that we can query the model for fluxes.
    graph_state = model.sample_parameters(num_samples=1)
    query_times = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 20.0, 21.0])
    query_filters = np.array(["u", "r", "u", "r", "u", "r", "u", "r"])

    fluxes = model.evaluate_bandfluxes(None, query_times, query_filters, graph_state)
    expected_mags = 2.0 * np.array([20.0, 22.0, 20.0, 22.0, 20.0, 22.0, 20.0, 22.0])
    expected_fluxes = mag2flux(expected_mags)
    assert np.allclose(fluxes, expected_fluxes)


def test_bagle_wrapper_model_filter_idx() -> None:
    """Test that we can create and query a BagleWrapperModel object with filter indices."""
    parameter_dict = {
        "scale": 2.0,
        "filter_mags": [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],  # ugrizy
    }
    custom_filter_idx = {"u": 5, "g": 4, "r": 3, "i": 2, "z": 1, "y": 0}
    model = BagleWrapperModel(_DummyBagleModel, parameter_dict, filter_idx=custom_filter_idx)

    # Test that we can query the model for fluxes.
    graph_state = model.sample_parameters(num_samples=1)
    query_times = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 20.0, 21.0])
    query_filters = np.array(["u", "r", "u", "r", "u", "r", "u", "r"])

    fluxes = model.evaluate_bandfluxes(None, query_times, query_filters, graph_state)
    expected_mags = 2.0 * np.array([25.0, 23.0, 25.0, 23.0, 25.0, 23.0, 25.0, 23.0])
    expected_fluxes = mag2flux(expected_mags)
    assert np.allclose(fluxes, expected_fluxes)
