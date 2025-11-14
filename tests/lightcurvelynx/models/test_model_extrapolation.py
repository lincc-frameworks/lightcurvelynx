"""Additional tests for extrapolation with a time and wavelength bounded model."""

import numpy as np
import pytest
from lightcurvelynx.models.physical_model import SEDModel
from lightcurvelynx.utils.extrapolate import LinearDecay


class _LinearLinearTestModel(SEDModel):
    """A model that emits flux as a linear function of wavelength and time:
    f(t, w) = 0.5 * w + 2.0 * t + 100.0

    The model includes bounds on valid times and wavelengths to test extrapolation.

    Parameters
    ----------
    min_phase : float
        The minimum phase of the model (time relative to t0).
    max_phase : float
        The maximum phase of the model (time relative to t0).
    min_wave : float
        The minimum wavelength of the model (in angstroms).
    max_wave : float
        The maximum wavelength of the model (in angstroms).
    **kwargs : dict
        Additional keyword arguments.
    """

    def __init__(self, min_phase=0.0, max_phase=100.0, min_wave=1000.0, max_wave=12000.0, **kwargs):
        super().__init__(**kwargs)
        self.min_phase = min_phase
        self.max_phase = max_phase
        self.min_wave = min_wave
        self.max_wave = max_wave

    def minwave(self, graph_state=None):
        """Get the minimum wavelength of the model."""
        return self.min_wave

    def maxwave(self, graph_state=None):
        """Get the maximum wavelength of the model."""
        return self.max_wave

    def minphase(self, graph_state=None):
        """Get the minimum phase of the model."""
        return self.min_phase

    def maxphase(self, graph_state=None):
        """Get the maximum phase of the model."""
        return self.max_phase

    def compute_sed(self, times, wavelengths, graph_state, **kwargs):
        """Draw effect-free observations for this object.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of rest frame timestamps.
        wavelengths : numpy.ndarray, optional
            A length N array of wavelengths (in angstroms).
        graph_state : GraphState
            An object mapping graph parameters to their values.
        **kwargs : dict, optional
            Any additional keyword arguments.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of SED values (in nJy).
        """
        t0 = self.get_param(graph_state, "t0")
        if t0 is None:
            raise ValueError("t0 parameter is required for this model.")

        phase = times - t0
        if np.any(phase < self.min_phase) or np.any(phase > self.max_phase):
            raise ValueError("Times are out of bounds for this model.")
        if np.any(wavelengths < self.min_wave) or np.any(wavelengths > self.max_wave):
            raise ValueError("Wavelengths are out of bounds for this model.")

        time_component = 2.0 * phase[:, np.newaxis]
        wavelength_component = 0.5 * wavelengths[np.newaxis, :]
        return 100.0 + time_component + wavelength_component


def test_linear_linear_model() -> None:
    """Test the _LinearLinearTestModel with valid times and wavelengths."""
    query_waves = np.array([1500.0, 2000.0])
    query_times = np.array([0.0, 10.0, 100.0])
    expected = np.array(
        [
            [100.0 + 2.0 * 0.0 + 0.5 * 1500.0, 100.0 + 2.0 * 0.0 + 0.5 * 2000.0],
            [100.0 + 2.0 * 10.0 + 0.5 * 1500.0, 100.0 + 2.0 * 10.0 + 0.5 * 2000.0],
            [100.0 + 2.0 * 100.0 + 0.5 * 1500.0, 100.0 + 2.0 * 100.0 + 0.5 * 2000.0],
        ]
    )

    model = _LinearLinearTestModel(t0=0.0)
    values = model.evaluate_sed(query_times, query_waves)
    assert values.shape == (3, 2)
    assert np.allclose(values, expected)


def test_linear_linear_model_bounds() -> None:
    """Test the _LinearLinearTestModel with out-of-bounds times and wavelengths."""
    model = _LinearLinearTestModel(t0=0.0)

    # We fail a call evaluate_sed since the times are out of bounds. But we
    # should get a warning about extrapolation first.
    query_times = np.array([-10.0, 50.0, 110.0])
    query_waves = np.array([1500.0, 2000.0])
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError):
            _ = model.evaluate_sed(query_times, query_waves)

    # We fail a call to evaluate_sed with out-of-bounds wavelengths. But we
    # should get a warning about extrapolation first.
    query_times = np.array([10.0, 20.0])
    query_waves = np.array([0.0, 2000.0, 5000.0, 13000.0])
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError):
            _ = model.evaluate_sed(query_times, query_waves)


def test_linear_linear_model_extrapolators() -> None:
    """Test the _LinearLinearTestModel with given extrapolators on
    out-of-bounds times and wavelengths."""
    # Wavelength falls of faster (per step) than time.
    wave_extrapolator = LinearDecay(decay_width=500.0)  # 500 angstroms to zero
    time_extrapolator = LinearDecay(decay_width=100.0)  # 100 days to zero
    model = _LinearLinearTestModel(
        wave_extrapolation=wave_extrapolator,
        time_extrapolation=time_extrapolator,
        t0=0.0,
    )

    # Three query wavelengths: one before (80% factor) and two inside.
    query_waves = np.array([900.0, 2000.0, 5000.0])
    # Three query times: one before (90% factor), three inside, and one after (80% factor).
    query_times = np.array([-10.0, 0.0, 50.0, 100.0, 120.0])
    values = model.evaluate_sed(query_times, query_waves)

    expected = np.array(
        [
            [432.0, 990.0, 2340.0],
            [480.0, 1100.0, 2600.0],
            [560.0, 1200.0, 2700.0],
            [640.0, 1300.0, 2800.0],
            [512.0, 1040.0, 2240.0],
        ]
    )
    assert np.allclose(values, expected)
