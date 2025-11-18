import numpy as np
import pytest
from lightcurvelynx.models.static_sed_model import StaticBandfluxModel, StaticSEDModel
from lightcurvelynx.utils.extrapolate import LinearDecay
from lightcurvelynx.utils.io_utils import write_numpy_data


def test_single_static_sed() -> None:
    """Test that we can create and sample a StaticSEDModel object with a single SED."""
    sed = np.array(
        [
            [100.0, 200.0, 300.0, 400.0],  # Wavelengths
            [10.0, 20.0, 20.0, 10.0],  # fluxes
        ]
    )
    model = StaticSEDModel(sed, node_label="test")
    assert len(model) == 1

    times = np.array([1, 2, 3, 10, 20])
    wavelengths = np.array([50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0])

    with pytest.warns(UserWarning):
        # Warns for wavelengths outside the SED range.
        values = model.evaluate_sed(times, wavelengths)
    assert values.shape == (5, 9)

    expected = np.array([0.0, 10.0, 15.0, 20.0, 20.0, 20.0, 15.0, 10.0, 0.0])
    for t_idx in range(5):
        assert np.array_equal(values[t_idx, :], expected)


def test_static_sed_fail() -> None:
    """Test that we correctly fail on bad SEDs."""
    # Non-numpy arrays
    with pytest.raises(ValueError):
        _ = StaticSEDModel([None], node_label="test")
    with pytest.raises(ValueError):
        _ = StaticSEDModel([1.0], node_label="test")

    # Not enough points.
    sed = np.array([[100.0], [10.0], [0.0]])
    with pytest.raises(ValueError):
        _ = StaticSEDModel([sed], node_label="test")

    # Incorrectly shaped data.
    sed = np.array(
        [
            [100.0, 200.0, 400.0],  # Wavelengths
            [10.0, 20.0, 20.0],  # fluxes
            [0.0, 0.0, 0.0],  # Other row
        ]
    )
    with pytest.raises(ValueError):
        _ = StaticSEDModel([sed], node_label="test")

    # Wavelengths unsorted.
    sed = np.array(
        [
            [100.0, 200.0, 400.0, 300.0],  # Wavelengths
            [10.0, 20.0, 20.0, 10.0],  # fluxes
        ]
    )
    with pytest.raises(ValueError):
        _ = StaticSEDModel([sed], node_label="test")


def test_static_sed_from_file(tmp_path) -> None:
    """Test that we can create a StaticSEDModel object from a file."""
    test_sed = np.array(
        [
            [100.0, 200.0, 300.0, 400.0],  # Wavelengths
            [10.0, 20.0, 20.0, 10.0],  # fluxes
        ]
    )
    times = np.array([1, 2, 3, 10, 20])
    wavelengths = np.array([100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0])
    expected = np.array([10.0, 15.0, 20.0, 20.0, 20.0, 15.0, 10.0])

    for fmt in ["npy", "npz", "txt", "csv"]:
        file_path = tmp_path / f"test_sed.{fmt}"
        write_numpy_data(file_path, test_sed.T)

        model = StaticSEDModel.from_file(file_path, node_label="test")
        assert len(model) == 1

        values = model.evaluate_sed(times, wavelengths)
        assert values.shape == (5, 7)

        for t_idx in range(5):
            assert np.array_equal(values[t_idx, :], expected)

    # Try an invalid array shape.
    test_sed_invalid = np.array(
        [
            [100.0, 200.0, 300.0, 400.0],  # Wavelengths
            [10.0, 20.0, 20.0, 10.0],  # fluxes
            [0.0, 0.0, 0.0, 0.0],  # Other row
        ]
    )
    file_path_invalid = tmp_path / "test_sed.csv"
    write_numpy_data(file_path_invalid, test_sed_invalid.T)

    with pytest.raises(ValueError):
        _ = StaticSEDModel.from_file(file_path_invalid, node_label="test")


def test_multiple_static_seds() -> None:
    """Test that we can create and sample a StaticSEDModel object with a multiple SEDs."""
    sed0 = np.array(
        [
            [100.0, 200.0, 300.0, 400.0],  # Wavelengths
            [10.0, 20.0, 20.0, 10.0],  # fluxes
        ]
    )
    sed1 = np.array(
        [
            [100.0, 200.0, 300.0, 400.0],  # Wavelengths
            [20.0, 40.0, 40.0, 20.0],  # fluxes
        ]
    )
    model = StaticSEDModel([sed0, sed1], weights=[0.25, 0.75], node_label="test")
    assert len(model) == 2

    # Check that all of the indices are 0 or 1 and the split is approximately 25/75
    params = model.sample_parameters(num_samples=10_000)
    inds_0 = params["test"]["selected_idx"] == 0
    inds_1 = params["test"]["selected_idx"] == 1
    assert np.all(inds_0 | inds_1)
    assert 1_500 < np.count_nonzero(inds_0) < 3_500
    assert 6_500 < np.count_nonzero(inds_1) < 8_500

    times = np.array([1, 2, 3])
    wavelengths = np.array([50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0])
    with pytest.warns(UserWarning):
        # Warns for wavelengths outside the SED range.
        values = model.evaluate_sed(times, wavelengths, params)

    expected_0 = np.tile([0.0, 10.0, 15.0, 20.0, 20.0, 20.0, 15.0, 10.0, 0.0], 3).reshape(3, 9)
    expected_1 = 2.0 * expected_0

    for idx in range(10_000):
        if inds_0[idx]:
            assert np.array_equal(values[idx], expected_0)
        else:
            assert np.array_equal(values[idx], expected_1)


def test_multiple_static_seds_min_max():
    """Test that we can get the min and max wavelengths from a StaticSEDModel object with multiple SEDs."""
    sed0 = np.array(
        [
            [100.0, 200.0, 300.0, 400.0],  # Wavelengths
            [10.0, 20.0, 20.0, 10.0],  # fluxes
        ]
    )
    sed1 = np.array(
        [
            [200.0, 300.0, 400.0, 500.0],  # Wavelengths
            [20.0, 40.0, 40.0, 20.0],  # fluxes
        ]
    )
    model = StaticSEDModel([sed0, sed1], weights=[0.5, 0.5], node_label="test")

    states = model.sample_parameters(num_samples=1)

    # Force the selected_idx to be 0 for the test.
    states.set("test", "selected_idx", 0)
    assert model.minwave(states) == 100.0
    assert model.maxwave(states) == 400.0

    # Force the selected_idx to be 1 for the test.
    states.set("test", "selected_idx", 1)
    assert model.minwave(states) == 200.0
    assert model.maxwave(states) == 500.0

    # We fail if we do not pass in the states.
    with pytest.raises(ValueError):
        _ = model.minwave()
    with pytest.raises(ValueError):
        _ = model.maxwave()


def test_single_static_bandflux() -> None:
    """Test that we can create and sample a StaticBandfluxModel object with a single bandflux."""
    bandflux = {"r": 10.0, "g": 20.0, "b": 30.0}
    model = StaticBandfluxModel(bandflux, node_label="test")
    assert len(model) == 1
    assert len(model.list_effects()) == 0

    times = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    filters = ["r", "r", "g", "b", "r", "g", "b", "g", "b", "r"]
    expected = np.array([10.0, 10.0, 20.0, 30.0, 10.0, 20.0, 30.0, 20.0, 30.0, 10.0])

    state = model.sample_parameters(num_samples=1)
    fluxes = model.evaluate_bandfluxes(None, times, filters, state)
    assert len(fluxes) == 10
    assert np.array_equal(fluxes, expected)


def test_multiple_static_bandflux() -> None:
    """Test that we can create and sample a StaticBandfluxModel object with multiple bandfluxes."""
    bandflux0 = {"r": 10.0, "g": 20.0, "b": 30.0}
    bandflux1 = {"r": 15.0, "g": 25.0, "b": 35.0}
    model = StaticBandfluxModel([bandflux0, bandflux1], weights=[0.25, 0.75], node_label="test")
    assert len(model) == 2

    # Check that all of the indices are 0 or 1 and the split is approximately 25/75
    params = model.sample_parameters(num_samples=10_000)
    inds_0 = params["test"]["selected_idx"] == 0
    inds_1 = params["test"]["selected_idx"] == 1
    assert np.all(inds_0 | inds_1)
    assert 1_500 < np.count_nonzero(inds_0) < 3_500
    assert 6_500 < np.count_nonzero(inds_1) < 8_500

    times = np.array([1, 2, 3, 4, 5])
    filters = ["r", "r", "g", "b", "r"]
    expected0 = np.array([10.0, 10.0, 20.0, 30.0, 10.0])
    expected1 = np.array([15.0, 15.0, 25.0, 35.0, 15.0])

    fluxes = model.evaluate_bandfluxes(None, times, filters, params)
    assert fluxes.shape == (10_000, 5)
    for idx in range(10_000):
        if inds_0[idx]:
            assert np.array_equal(fluxes[idx], expected0)
        else:
            assert np.array_equal(fluxes[idx], expected1)


def test_static_bandflux_set_extrapolation():
    """Test that we can set the time extrapolation for a StaticBandfluxModel."""
    bandflux = {"r": 10.0, "g": 20.0, "b": 30.0}

    before_extrap = LinearDecay(decay_width=10.0)  # 10 days to go to zero
    after_extrap = LinearDecay(decay_width=20.0)  # 20 days to go to zero
    model = StaticBandfluxModel(bandflux, time_extrapolation=(before_extrap, after_extrap), node_label="test")
    assert model._time_extrap_before is not None
    assert model._time_extrap_after is not None
    assert isinstance(model._time_extrap_before, LinearDecay)
    assert isinstance(model._time_extrap_after, LinearDecay)

    # We provide a warning if the user tries to set wave_extrapolation.
    with pytest.warns(UserWarning):
        _ = StaticBandfluxModel(bandflux, wave_extrapolation=LinearDecay(decay_width=5.0))

    # We fail with invalid time_extrapolation settings.
    with pytest.raises(ValueError):
        _ = StaticBandfluxModel(bandflux, time_extrapolation=(before_extrap, before_extrap, before_extrap))
    with pytest.raises(TypeError):
        _ = StaticBandfluxModel(bandflux, time_extrapolation="not a model")


class DummySynphotModel:
    """A fake synphot model used for testing.

    Attributes
    ----------
    waveset : numpy.ndarray
        The wavelengths at which the SED is defined (in angstroms)
    fluxset : numpy.ndarray
        The flux at each given wavelength (in nJy.)
    """

    def __init__(self, waveset, fluxset):
        self.waveset = waveset
        self.fluxset = fluxset
        self.z = 0.0  # Redshift

    def __call__(self, waves, **kwargs):
        """Return the flux for the given wavelengths as interpolated PHOTLAM.

        Parameters
        ----------
        waves : numpy.ndarray
            The wavelengths at which to evaluate the SED (in angstroms).
        **kwargs : dict
            Additional keyword arguments (ignored).

        Returns
        -------
        numpy.ndarray
            The interpolated flux values at the given wavelengths (in PHOTLAM).
        """
        # Return a dummy SED for the given wavelengths
        return np.interp(waves, self.waveset, self.fluxset, left=0.0, right=0.0)


def test_static_sed_from_synphot() -> None:
    """Test that we can create a StaticSEDModel from a synphot model."""
    # Create a dummy model with 4 samples of SEDs [10.0, 20.0, 30.0, 40.0] in nJy.
    # Since synphot uses PHOTLAM, we preconvert and provide in that unit.
    sp_model = DummySynphotModel(
        waveset=np.array([1000.0, 2000.0, 3000.0, 4000.0]),
        fluxset=np.array([1.50919018e-08, 1.50919018e-08, 1.50919018e-08, 1.50919018e-08]),
    )
    model = StaticSEDModel.from_synphot(sp_model)
    assert len(model) == 1

    times = np.array([1, 2, 3, 10, 20])
    wavelengths = np.array([500.0, 1000.0, 1500.0, 2000.0, 3000.0, 5000.0])
    expected = np.array([0.0, 10.0, 15.0, 20.0, 30.0, 0.0])
    with pytest.warns(UserWarning):
        # Warns for wavelengths outside the SED range.
        fluxes = model.evaluate_sed(times, wavelengths)
    assert fluxes.shape == (len(times), len(wavelengths))
    for i in range(len(times)):
        np.testing.assert_allclose(fluxes[i, :], expected, rtol=1e-5)

    # We fail is the synphot model has a redshift defined.
    sp_model.z = 0.5
    with pytest.raises(ValueError):
        _ = StaticSEDModel.from_synphot(sp_model)


def test_static_bandflux_with_extrapolation():
    """Test that we can extrapolate a Bandflux model by adding time bounds
    to a StaticBandfluxModel.
    """

    class _BoundStaticBandfluxModel(StaticBandfluxModel):
        """A StaticBandfluxModel that defines time bounds for extrapolation."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._time_bounds = (0.0, 10.0)

        def minphase(self, **kwargs):
            """Return the minimum phase for the model."""
            return self._time_bounds[0]

        def maxphase(self, **kwargs):
            """Return the maximum phase for the model."""
            return self._time_bounds[1]

    before_extrap = LinearDecay(decay_width=10.0)  # 10 days to go to zero
    after_extrap = LinearDecay(decay_width=20.0)  # 20 days to go to zero
    bandflux = {"r": 10.0, "g": 20.0, "b": 30.0}
    model = _BoundStaticBandfluxModel(
        bandflux,
        time_extrapolation=(before_extrap, after_extrap),
        node_label="test",
    )

    # Data in the form of (time, filter, expected_flux)
    data = np.array(
        [
            [-5.0, "r", 5.0],  # 50% of red
            [-4.0, "r", 6.0],  # 60% of red
            [-2.0, "g", 16.0],  # 80% of green
            [-1.0, "b", 27.0],  # 90% of blue
            [0.0, "r", 10.0],  # No extrapolation
            [1.0, "g", 20.0],  # No extrapolation
            [2.0, "b", 30.0],  # No extrapolation
            [3.0, "r", 10.0],  # No extrapolation
            [4.0, "g", 20.0],  # No extrapolation
            [5.0, "b", 30.0],  # No extrapolation
            [5.0, "b", 30.0],  # No extrapolation
            [11.0, "r", 9.5],  # 95% of red
            [12.0, "g", 18.0],  # 90% of green
            [13.0, "b", 25.5],  # 85% of blue
            [14.0, "r", 8.0],  # 80% of red
            [15.0, "b", 22.5],  # 75% of blue
        ]
    )

    times = data[:, 0].astype(float)
    filters = data[:, 1]
    expected = data[:, 2].astype(float)
    state = model.sample_parameters(num_samples=1)
    fluxes = model.evaluate_bandfluxes(None, times, filters, state)
    assert np.array_equal(fluxes, expected)
