import numpy as np
import pytest
from lightcurvelynx.astro_utils.mag_flux import mag2flux
from lightcurvelynx.models.bagle_models import BagleMultiWrapperModel, BagleWrapperModel


class _FlatBagleModel:
    """A flat dummy bagle model for testing purposes.

    Attributes
    ----------
    scale : float
        A scaling factor for the fluxes.
    filter_mags : list
        A list of magnitudes for each filter.
    """

    def __init__(self, scale, filter_mags, t0=None):
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


class _GaussianBagleModel:
    """A Gaussian-shaped dummy bagle model for testing purposes.

    Attributes
    ----------
    scale : float
        A scaling factor for the fluxes.
    filter_mags : list
        A list of magnitudes for each filter.
    shift : float
        A constant magnitude shift to apply.
    """

    def __init__(self, scale, filter_mags, t0=None, shift=None, **kwargs):
        self.scale = scale
        self.filter_mags = filter_mags
        self.shift = shift

        # Simulate handling a t0 that is computed based on arguments.
        if t0 is not None:
            self.t0 = t0
        elif "t0_com" in kwargs:
            self.t0 = kwargs["t0_com"] - 2.0
        else:
            self.t0 = 0.0

    def get_photometry(self, times, filter_idx):
        """Compute dummy fluxes based on parameters and times. Simulates
        a Gaussian light curve shape.

        Parameters
        ----------
        times : numpy.ndarray
            An array of times.
        filter_idx : int
            The index of the filter.
        """
        print(f"Computing photometry with t0 = {self.t0}")

        # Since we are computing magnitudes we subtract out the Gaussian shape.
        base = -np.exp(-0.5 * ((times - self.t0) / 50.0) ** 2)
        print(f"Base values: {base}")
        values = base * self.scale + self.filter_mags[filter_idx]
        print(f"Values before shift: {values}")
        if self.shift is not None:
            values += self.shift
        print(f"Values after shift: {values}")
        return values


def test_bagle_wrapper_model() -> None:
    """Test that we can create and query a BagleWrapperModel object."""
    parameter_dict = {
        "scale": 2.0,
        "filter_mags": [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],  # ugrizy
    }
    model = BagleWrapperModel(_FlatBagleModel, parameter_dict, node_label="model")

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
    assert graph_state["model"]["scale"] == 2.0
    assert np.array_equal(
        graph_state["model"]["filter_mags"],
        [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
    )
    assert graph_state["model"]["ra"] is None
    assert graph_state["model"]["dec"] is None

    query_times = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 20.0, 21.0])
    query_filters = np.array(["u", "r", "u", "r", "u", "r", "u", "r"])

    fluxes = model.evaluate_bandfluxes(None, query_times, query_filters, graph_state)
    expected_mags = 2.0 * np.array([20.0, 22.0, 20.0, 22.0, 20.0, 22.0, 20.0, 22.0])
    expected_fluxes = mag2flux(expected_mags)
    assert np.allclose(fluxes, expected_fluxes)


def test_bagle_wrapper_model_ra_dec() -> None:
    """Test that we can create and query a BagleWrapperModel object
    with different settings for RA, dec."""
    # No RA/dec given.
    parameter_dict = {
        "scale": 2.0,
        "filter_mags": [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],  # ugrizy
    }
    model1 = BagleWrapperModel(
        _FlatBagleModel,
        parameter_dict,
        node_label="model",
    )
    graph_state1 = model1.sample_parameters(num_samples=1)
    assert graph_state1["model"]["ra"] is None
    assert graph_state1["model"]["dec"] is None

    # Specify RA and dec as kwargs to the wrapper model.
    model2 = BagleWrapperModel(
        _FlatBagleModel,
        parameter_dict,
        ra=21.0,
        dec=-11.0,
        node_label="model",
    )
    graph_state2 = model2.sample_parameters(num_samples=1)
    assert graph_state2["model"]["ra"] == 21.0
    assert graph_state2["model"]["dec"] == -11.0

    # Specify RA and dec as part of the parameter_dict (to the bagle model).
    parameter_dict2 = {
        "scale": 2.0,
        "filter_mags": [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],  # ugrizy
        "raL": 150.0,
        "decL": 2.0,
    }
    model3 = BagleWrapperModel(
        _FlatBagleModel,
        parameter_dict2,
        node_label="model",
    )
    graph_state3 = model3.sample_parameters(num_samples=1)
    assert graph_state3["model"]["ra"] == 150.0
    assert graph_state3["model"]["dec"] == 2.0

    # Using both kwargs and parameter_dict should raise an error.
    with pytest.raises(ValueError):
        _ = BagleWrapperModel(
            _FlatBagleModel,
            parameter_dict2,
            ra=21.0,
            dec=-11.0,
            node_label="model",
        )


def test_bagle_wrapper_model_t0() -> None:
    """Test that we can create and query a BagleWrapperModel object with t0 set."""
    parameter_dict = {
        "scale": 5.0,
        "filter_mags": [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],  # ugrizy
        "t0": 10.0,
    }
    model = BagleWrapperModel(
        _GaussianBagleModel,
        parameter_dict,
        node_label="model",
    )

    # Check that we can query the model for fluxes.
    graph_state = model.sample_parameters(num_samples=1)
    assert graph_state["model"]["t0"] == 10.0

    query_times = np.arange(-5.0, 25.0, 0.1)
    query_filters = np.array(["r"] * len(query_times))
    fluxes = model.evaluate_bandfluxes(None, query_times, query_filters, graph_state)

    # The max flux should be at t0 = 10.0
    max_idx = np.int64(150)  # index of time = 10.0
    assert np.argmax(fluxes) == max_idx

    # We do not allow specifying t0 via kwargs to the wrapper model.
    with pytest.raises(ValueError):
        _ = BagleWrapperModel(
            _GaussianBagleModel,
            parameter_dict,
            t0=10.0,
            node_label="model",
        )

    # We can specify t0 as a computed parameter.
    parameter_dict2 = {
        "scale": 2.0,
        "filter_mags": [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],  # ugrizy
        "t0_com": 12.0,  # t0 will be computed as t0_com - 2.0 = 10.0
    }
    model2 = BagleWrapperModel(
        _GaussianBagleModel,
        parameter_dict2,
        node_label="model",
    )

    # The wrapper model will save the base value of t0 (not the computed one).
    # However the correct translation (computed value) will be used for the fluxes.
    graph_state2 = model2.sample_parameters(num_samples=1)
    assert graph_state2["model"]["t0"] == 12.0  # base value
    fluxes2 = model2.evaluate_bandfluxes(None, query_times, query_filters, graph_state2)
    assert graph_state2["model"]["t0"] == 12.0  # base value
    assert np.argmax(fluxes2) == max_idx


def test_bagle_wrapper_model_filter_idx() -> None:
    """Test that we can create and query a BagleWrapperModel object with filter indices."""
    parameter_dict = {
        "scale": 2.0,
        "filter_mags": [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],  # ugrizy
    }
    custom_filter_idx = {"u": 5, "g": 4, "r": 3, "i": 2, "z": 1, "y": 0}
    model = BagleWrapperModel(_FlatBagleModel, parameter_dict, filter_idx=custom_filter_idx)

    # Test that we can query the model for fluxes.
    graph_state = model.sample_parameters(num_samples=1)
    query_times = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 20.0, 21.0])
    query_filters = np.array(["u", "r", "u", "r", "u", "r", "u", "r"])

    fluxes = model.evaluate_bandfluxes(None, query_times, query_filters, graph_state)
    expected_mags = 2.0 * np.array([25.0, 23.0, 25.0, 23.0, 25.0, 23.0, 25.0, 23.0])
    expected_fluxes = mag2flux(expected_mags)
    assert np.allclose(fluxes, expected_fluxes)


def test_bagle_multi_wrapper_model() -> None:
    """Test that we can create and query a BagleMultiWrapperModel object."""
    # Create three different bagle models with different parameters (including
    # a different number of parameters per model).
    parameter_dicts = [
        {
            "scale": 2.0,
            "filter_mags": [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],  # ugrizy
            "t0": 0.0,
        },
        {
            "scale": 3.0,
            "filter_mags": [20.0, 20.0, 20.0, 20.0, 20.0, 20.0],  # ugrizy
            "t0": 3.0,
            "raL": 150.0,
            "decL": 2.0,
        },
        {
            "scale": 4.0,
            "filter_mags": [20.0, 20.0, 20.0, 20.0, 20.0, 20.0],  # ugrizy
            "shift": 1.0,
            "t0": 4.0,
        },
    ]
    model_list = [_FlatBagleModel, _GaussianBagleModel, _GaussianBagleModel]

    model = BagleMultiWrapperModel(
        model_list,
        parameter_dicts,
        in_order=True,
        node_label="model",
    )

    # Check that we have the given parameters.
    all_params = model.list_params()
    assert "scale" in all_params
    assert "filter_mags" in all_params
    assert "shift" in all_params

    # Check that we have the standard physical model parameters.
    assert "ra" in all_params
    assert "dec" in all_params
    assert "t0" in all_params
    assert "redshift" in all_params
    assert "distance" in all_params

    # Test that we can query the model for fluxes.
    graph_state = model.sample_parameters(num_samples=3)
    assert np.allclose(graph_state["model"]["scale"], [2.0, 3.0, 4.0])
    assert np.allclose(graph_state["model"]["t0"], [0.0, 3.0, 4.0])
    assert np.array_equal(
        graph_state["model"]["filter_mags"][0],
        [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
    )
    assert np.allclose(graph_state["model"]["filter_mags"][1], 20.0)
    assert np.allclose(graph_state["model"]["filter_mags"][2], 20.0)
    assert np.array_equal(graph_state["model"]["ra"], [None, 150.0, None])
    assert np.array_equal(graph_state["model"]["dec"], [None, 2.0, None])

    query_times = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 20.0, 21.0])
    query_filters = np.array(["u", "r", "u", "r", "u", "r", "u", "r"])

    fluxes = model.evaluate_bandfluxes(None, query_times, query_filters, graph_state)
    print(fluxes)
    assert fluxes.shape == (3, len(query_times))

    # Check the flat model.
    mag_0 = 2.0 * np.array([20.0, 22.0, 20.0, 22.0, 20.0, 22.0, 20.0, 22.0])
    assert np.allclose(fluxes[0], mag2flux(mag_0))

    # Check the Gaussian models have the correct peak.
    assert np.argmax(fluxes[1]) == 4  # max at t0 = 3.0
    mag_1 = -np.exp(-0.5 * ((query_times - 3.0) / 50.0) ** 2) * 3.0 + 20.0
    assert np.allclose(fluxes[1], mag2flux(mag_1))

    assert np.argmax(fluxes[2]) == 5  # max at t0 = 4.0
    mag_2 = -np.exp(-0.5 * ((query_times - 4.0) / 50.0) ** 2) * 4.0 + 21.0
    assert np.allclose(fluxes[2], mag2flux(mag_2))
