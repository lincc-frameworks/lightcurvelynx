"""SpectrographNoiseModels are used to simulate the noise on a per-observation, per-spectral-bin
basis.  They may use information from the Spectrograph object and/or ObsTable.
"""

from abc import ABC, abstractmethod

import numpy as np


class SpectrographNoiseModel(ABC):
    """An abstract base class noise model for simulating spectrograph measurements.

    Noise is applied by computing `flux_err`, which is the standard deviation of Gaussian noise
    to apply to each measurement (observation and bin). Subclasses must implement the `compute_flux_error`
    method to compute the noise parameters.

    Attributes
    ----------
    spectrograph : Spectrograph
        The spectrograph object containing the instrument parameters.
    """

    # A list of column names that must be present in the ObsTable for this noise model to work.
    _required_values = []

    def __init__(self, *, spectrograph=None):
        """Create a SpectrographNoiseModel.

        Parameters
        ----------
        spectrograph : Spectrograph, optional
            The spectrograph object containing the instrument parameters.
        """
        self.spectrograph = spectrograph

    @property
    def required_values(self):
        """List of column names that must be present in the ObsTable for this noise model to work."""
        return self._required_values

    @abstractmethod
    def compute_flux_error(self, measurements, **kwargs):
        """Compute the flux error for the given measurements and observation parameters.

        Parameters
        ----------
        measurements : matrix of float
            A T x B matrix of flux measurements in energy units (e.g. erg/s/cm²), where
            T is the number of observations and B is the number of spectral bins.
        **kwargs
            Additional parameters for the noise model.

        Returns
        -------
        flux_err : numpy.ndarray
            The standard deviation of the flux measurement error (in erg/s/cm²)
        """
        raise NotImplementedError("Subclasses must implement this method.")  # pragma: no cover

    def apply_noise(
        self,
        measurements,
        *,
        obs_table=None,
        indices=None,
        rng=None,
        **kwargs,
    ):
        """Compute the noise parameters for given observations in an ObsTable and
        apply noise to the input measurements.

        Parameters
        ----------
        measurements : matrix of float
            A T x B matrix of flux measurements in energy units (e.g. erg/s/cm²), where
            T is the number of observations and B is the number of spectral bins.
        obs_table : ObsTable, optional
            Table containing the observation parameters, including all
            parameters needed to compute the noise.
        indices : array_like of int, optional
            Indices of the observations in the ObsTable to which noise should be applied.
            If provided, the length of `indices` must match the number of rows in `measurements`.
        rng : np.random.Generator, optional
            The random number generator to use for applying noise. If None,
            a default generator will be used.
        **kwargs
            Additional parameters for the noise model.

        Returns
        -------
        measurements : numpy.ndarray
            The updated T x B matrix of flux measurements after applying noise, in the same
            units as the input measurements.
        flux_err : numpy.ndarray
            The T x B matrix of flux measurement error used for applying noise, in the
            same units as the input measurements (erg/s/cm²).
        """
        # Define the random number generator if not provided.
        if rng is None:  # pragma: no cover
            rng = np.random.default_rng()

        # Compute the standard deviation of the noise and make sure it is a numpy array.
        flux_err = self.compute_flux_error(
            measurements,
            obs_table=obs_table,
            indices=indices,
            rng=rng,
            **kwargs,
        )
        flux_err = np.asarray(flux_err)

        # Generate the actual noisy bandflux measurements.
        noisy_measurements = rng.normal(loc=measurements, scale=flux_err)
        return noisy_measurements, flux_err

    def check_compatibility(self, obs_table, fail_on_incompatible=False):
        """Check if the noise model is compatible with the given ObsTable.

        Parameters
        ----------
        obs_table : ObsTable
            The observation table to check for compatibility.
        fail_on_incompatible : bool, optional
            If True, raise a ValueError if the noise model is not compatible with the ObsTable.
            If False, simply return False in that case. Default is False.

        Returns
        -------
        bool
            True if the noise model is compatible with the ObsTable, False otherwise.
        """
        missing_columns = [col for col in self._required_values if col not in obs_table]
        if missing_columns:
            if fail_on_incompatible:
                raise ValueError(
                    f"Noise model {self.__class__.__name__} is not compatible with the given ObsTable. "
                    f"Missing required columns: {missing_columns}"
                )
            return False

        # Check if the required columns have valid data for each row.
        for col in self._required_values:
            values = obs_table.get_value_per_row(col)
            if np.issubdtype(values.dtype, np.number) and not np.isfinite(values).all():
                if fail_on_incompatible:
                    raise ValueError(f"Found invalid values in column '{col}'")
                return False

        return True


class ConstantSpectrographNoiseModel(SpectrographNoiseModel):
    """A noise model that simulates photon noise for spectrograph measurements
    sampled from a normal distribution with a constant standard deviation (for all bins).
    This class is primarily meant for testing purposes.

    Attributes
    ----------
    noise_level : float
        The (constant) standard deviation of the noise to apply to the spectrograph
        measurements, in the same units as the input measurements.
    """

    def __init__(self, noise_level, *, spectrograph=None):
        """Create a ConstantSpectrographNoiseModel.

        Parameters
        ----------
        noise_level : float
            The (constant) standard deviation of the noise to apply to the
            spectrograph flux measurements, in the same units as the input measurements.
        spectrograph : Spectrograph, optional
            The spectrograph object containing the instrument parameters.
        """
        super().__init__(spectrograph=spectrograph)
        if noise_level < 0:
            raise ValueError("Noise level must be non-negative.")
        self.noise_level = noise_level

    def compute_flux_error(self, measurements, **kwargs):
        """Compute the flux error for the given measurements and observation parameters.

        Parameters
        ----------
        measurements : matrix of float
            A T x B matrix of flux measurements in energy units (e.g. nJy), where
            T is the number of observations and B is the number of spectral bins.
        **kwargs
            Additional parameters for the noise model.

        Returns
        -------
        flux_err : numpy.ndarray
            The standard deviation of the flux measurement error (in nJy)
        """
        return np.full_like(measurements, self.noise_level, dtype=float)
