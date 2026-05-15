import numpy as np
import pytest
from lightcurvelynx.math_nodes.np_random import NumpyRandomFunc
from lightcurvelynx.models.gopreaux_models import GoPreauxModel
from lightcurvelynx.utils.extrapolate import LastValue


# Since we cannot pip install gopreaux, we create a fake model that
# simulates its functionality for testing purposes.
class _FakeGoPreauxModel:
    """A fake model that simulates the interface of a gopreaux SNModel for testing purposes.

    This model simulates a simple Gaussian-shaped light curve in both wavelength and time,
    with a peak at 5500 angstroms and phase=0. The brightness is returned in magnitudes
    relative to the peak, where a delta of 1.0 indicates an increase in brightness by 1.0 magnitude
    (i.e., a decrease in magnitude).
    """

    @property
    def min_wl(self):
        return 2999.0

    @property
    def max_wl(self):
        return 8001.0

    @property
    def min_phase(self):
        return -20.0

    @property
    def max_phase(self):
        return 50.0

    def predict_photometry_points(self, wavelengths, phases, show=False):
        """Simulate some photometry points for testing purposes.

        Parameters
        ----------
        wavelengths : numpy.ndarray
            A length N array of wavelengths (in angstroms).
        phases : numpy.ndarray
            A length N array of phases (in days).
        show : bool
            Whether to show a plot of the predicted photometry points (ignored in this fake model).

        Returns
        -------
        None
            Used data value for API consistency with the real SNModel, but not used in this fake model.
        rel_mag : numpy.ndarray
            A length N array of relative magnitudes (simulated as a Gaussian over wavelength and time).
        None
            Used data value for API consistency with the real SNModel, but not used in this fake model.
        """
        # The flux density is a Gaussian over wavelength centered at 5500 angstroms.
        if np.any(wavelengths < self.min_wl) or np.any(wavelengths > self.max_wl):
            raise ValueError("Wavelengths out of bounds.")
        wave_scale = np.exp(-0.5 * ((wavelengths - 5500.0) / 1000.0) ** 2)

        # The time dependence is a Gaussian centered at phase=0.0 with a width of 10 days.
        if np.any(phases < self.min_phase) or np.any(phases > self.max_phase):
            raise ValueError("Phases out of bounds.")
        time_scale = np.exp(-0.5 * (phases / 10.0) ** 2)

        # Total scale is the product of the wavelength and time dependence, scaled to be between
        # -3 and 2 magnitudes. The sign of this delta corresponds to the change in brightness,
        # so a positive delta indicates an increase in brightness (decrease in magnitude) and a
        # negative delta corresponds to a decrease in brightness (increase in magnitude).
        total_scale = wave_scale * time_scale
        total_scale = (total_scale / np.max(total_scale)) * 5.0 - 3.0

        return None, total_scale, None


def test_gopreaux_model():
    """
    Test creating a simple GoPreauxModel and evaluating it at some
    times and wavelengths.
    """
    # Create the fake model.
    fake_model = _FakeGoPreauxModel()

    # Sample the brightness randomly from 15 to 18 mag.
    brightness_sampler = NumpyRandomFunc("uniform", low=15, high=18)

    # Create the GoPreauxModel using the fake model and brightness sampler.
    t_start = 64350.0
    model = GoPreauxModel(
        fake_model,
        intrinsic_brightness=brightness_sampler,
        # Standard attributes for all physical objects, set all to constants
        # for this example.
        ra=45.0,
        dec=-20.0,
        redshift=1e-10,  # Effectively zero redshift for testing purposes
        t0=t_start + 10.0,
        node_label="source",
    )

    # Evaluate the model at some times (every 2 days starting 10 days before t0) and
    # wavelengths around the peak (3000 to 8000 angstroms).
    times = t_start + np.arange(30) * 2.0
    waves_ang = np.array([3000.0, 4000.0, 5500.0, 6000.0, 7000.0, 8000.0])
    fluxes = model.evaluate_sed(times, waves_ang)

    assert fluxes.shape == (len(times), len(waves_ang))
    assert np.all(fluxes > 0.0)
    assert np.all(fluxes[5] > fluxes[:5])  # Peak at phase=0
    assert np.all(fluxes[5] > fluxes[6:])  # Peak at phase=0
    assert np.all(fluxes[:, 2] > fluxes[:, 0])  # Peak at 5500 A
    assert np.all(fluxes[:, 2] > fluxes[:, 1])  # Peak at 5500 A
    assert np.all(fluxes[:, 2] > fluxes[:, 3])  # Peak at 5500 A
    assert np.all(fluxes[:, 2] > fluxes[:, 4])  # Peak at 5500 A
    assert np.all(fluxes[:, 2] > fluxes[:, 5])  # Peak at 5500 A

    # We fail with times outside the model bounds. The PhysicalModel base class
    # prints a warning first.
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError):
            model.evaluate_sed([t_start + 100.0], [5500.0])
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError):
            model.evaluate_sed([t_start - 100.0], [5500.0])

    # We fail with wavelengths outside the model bounds. The PhysicalModel base class
    # prints a warning first.
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError):
            model.evaluate_sed([t_start + 10.0], [2000.0])
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError):
            model.evaluate_sed([t_start + 10.0], [9000.0])


def test_gopreaux_model_extrapolate():
    """
    Test creating a simple GoPreauxModel and evaluating it at some
    times and wavelengths with extrapolation outside the model bounds.
    """
    # Create the fake model.
    fake_model = _FakeGoPreauxModel()

    # Sample the brightness randomly from 15 to 18 mag.
    brightness_sampler = NumpyRandomFunc("uniform", low=15, high=18)

    # Create the GoPreauxModel using the fake model and brightness sampler.
    t_start = 64350.0
    model = GoPreauxModel(
        fake_model,
        intrinsic_brightness=brightness_sampler,
        # Standard attributes for all physical objects, set all to constants
        # for this example.
        ra=45.0,
        dec=-20.0,
        redshift=1e-10,  # Effectively zero redshift for testing purposes
        t0=t_start + 10.0,
        node_label="source",
        time_extrapolation=LastValue(),
        wave_extrapolation=LastValue(),
    )

    # Evaluate the model at some times (every 10 days starting 50 days before t0) and
    # wavelengths around the peak (2000 to 10000 angstroms).
    times = t_start + np.arange(100) * 10.0 - 40.0
    waves_ang = np.array([2000.0, 3000.0, 4000.0, 5500.0, 6000.0, 8000.0, 10000.0])
    fluxes = model.evaluate_sed(times, waves_ang)
    assert fluxes.shape == (len(times), len(waves_ang))
    assert np.all(fluxes > 0.0)

    # Check that the extrapolated values are flat after the edge of the model.
    # They won't exactly match the edge value due to the small bounds offset.
    assert np.allclose(fluxes[times < t_start - 10.0, :], fluxes[times == t_start - 10.0, :], rtol=0.01)
    assert np.allclose(fluxes[times > t_start + 50.0, :], fluxes[times == t_start + 50.0, :], rtol=0.01)
    assert np.allclose(fluxes[:, 0], fluxes[:, 1], rtol=0.01)
    assert np.allclose(fluxes[:, 6], fluxes[:, 5], rtol=0.01)
