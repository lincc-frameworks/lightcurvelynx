import numpy as np
import pytest

from lightcurvelynx.models.periodic_model import PeriodicModel


class SineSourceModel(PeriodicModel):
    """A simple sine model with power (Rayleigh–Jeans) spectrum."""

    def _evaluate_phases(self, phases, wavelengths, graph_state, **kwargs):
        del kwargs

        amplitude = 2 + np.sin(2 * np.pi * phases[:, None])
        return amplitude * wavelengths[None, :] ** -2


@pytest.mark.parametrize("period, t0", [(1.0, 0.0), (2.0, 0.75), (4.0, 100 / 3)])
def test_periodicity(period, t0):
    """Test that the model is periodic."""
    max_time = 16
    n_periods = int(max_time / period)

    model = SineSourceModel(period=period, t0=t0)
    times = np.linspace(0, max_time, max_time * 100 + 1)
    wavelengths = np.linspace(100, 200, 3)
    fluxes = model.evaluate_sed(times, wavelengths)

    # Test that the flux is periodic on boundaries.
    np.testing.assert_allclose(fluxes[0], fluxes[-1])

    # Test that each period is identical.
    # First :-1 is to drop the last point that belongs to the next period.
    fluxes_in_periods = fluxes[:-1].reshape(n_periods, -1, len(wavelengths))
    for i_period in range(1, n_periods):
        np.testing.assert_allclose(fluxes_in_periods[0], fluxes_in_periods[i_period])
