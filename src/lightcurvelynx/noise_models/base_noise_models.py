from abc import ABC, abstractmethod

import numpy as np
from lightcurvelynx.noise_models.noise_utils import poisson_bandflux_std


class FluxNoiseModel(ABC):
    """An abstract baseclass noise model for simulating bandflux measurements."""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def apply_noise(
        self,
        bandflux,
        *,
        obs_table=None,
        indices=None,
        rng=None,
        **kwargs,
    ):
        """Compute the noise parameters for given observations in
        an ObsTable and apply noise to the input bandflux.

        Parameters
        ----------
        bandflux : array_like of float
            Source bandflux in energy units, e.g. nJy.
        obs_table : ObsTable, optional
            Table containing the observation parameters, including all
            parameters needed to compute the noise.
        indices : array_like of int, optional
            Indices of the observations in the ObsTable to which noise should
            be applied.
        rng : np.random.Generator, optional
            The random number generator to use for applying noise. If None,
            a default generator will be used.
        **kwargs
            Additional parameters for the noise model.

        Returns
        -------
        flux : array_like
            The updated flux measuements after applying noise, in the same
            units as the input bandflux.
        flux_err : array_like
            The bandflux measurement error used for applying noise, in the
            same units as the input bandflux.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class ConstantFluxNoiseModel(FluxNoiseModel):
    """A noise model that simulates photon noise for bandflux measurements
    with a constant noise level. This class is primarily meant for
    testing purposes.

    Attributes
    ----------
    noise_level : float
        The constant noise level to apply to the bandflux measurements, in the
        same units as the input bandflux.
    """

    def __init__(self, noise_level):
        if noise_level < 0:
            raise ValueError("Noise level must be non-negative.")
        self.noise_level = noise_level

    def apply_noise(
        self,
        bandflux,
        *,
        rng=None,
        **kwargs,
    ):
        """Compute the noise parameters for given observations in
        an ObsTable and apply noise to the input bandflux.

        Parameters
        ----------
        bandflux : array_like of float
            Source bandflux in energy units, e.g. nJy.
        rng : np.random.Generator, optional
            The random number generator to use for applying noise. If None,
            a default generator will be used.
        **kwargs
            Additional parameters for the noise model.

        Returns
        -------
        flux : array_like
            The updated flux measuements after applying noise, in the same
            units as the input bandflux.
        flux_err : array_like
            The bandflux measurement error used for applying noise, in the
            same units as the input bandflux.
        """
        if rng is None:
            rng = np.random.default_rng()

        noisy_bandflux = rng.normal(loc=bandflux, scale=self.noise_level)
        return noisy_bandflux, np.full_like(bandflux, self.noise_level)


class PoissonFluxNoiseModel(FluxNoiseModel):
    """A noise model that simulates photon noise for bandflux measurements
    with a Poisson noise level that are extracted from an ObsTable.

    This class is meant to be subclassed for specific ObsTable implementations
    where the columns vary. Subclass should override the compute_flux_error method.
    """

    def __init__(self):
        pass

    def compute_flux_error(self, bandflux, obs_table, indices):
        """Compute the flux error for the given bandflux and observation parameters.

        Parameters
        ----------
        bandflux : array_like of float
            Source bandflux in nJy.
        obs_table : ObsTable
            Table containing the observation parameters needed to compute the noise.
        indices : array_like of int
            Indices of the observations in the ObsTable for which to compute the noise.

        Returns
        -------
        flux_err : array_like
            The standard deviation of the bandflux measurement error (in nJy)
        """
        # Extract the features needed to compute the noise from the ObsTable.
        total_exposure_time = obs_table.get_value_per_row("exptime", indices=indices)
        exposure_count = obs_table.get_value_per_row("nexposure", indices=indices, default=1)
        sky = obs_table.get_value_per_row("sky_bg_e", indices=indices)
        psf_footprint = obs_table.get_value_per_row("psf_footprint", indices=indices)
        zp = obs_table.get_value_per_row("zp", indices=indices)
        readout_noise = obs_table.get_value_per_row("read_noise", indices=indices)
        dark_current = obs_table.get_value_per_row("dark_current", indices=indices)
        zp_err_mag = obs_table.get_value_per_row("zp_err_mag", indices=indices, default=0.0)

        # Compute the flux error standard deviation.
        return poisson_bandflux_std(
            bandflux,
            total_exposure_time=total_exposure_time,
            exposure_count=exposure_count,
            psf_footprint=psf_footprint,
            sky=sky,
            zp=zp,
            readout_noise=readout_noise,
            dark_current=dark_current,
            zp_err_mag=zp_err_mag,
        )

    def apply_noise(
        self,
        bandflux,
        *,
        obs_table=None,
        indices=None,
        rng=None,
        **kwargs,
    ):
        """Compute the noise parameters for given observations in
        an ObsTable and apply noise to the input bandflux.

        Parameters
        ----------
        bandflux : array_like of float
            Source bandflux in energy units, e.g. nJy.
        obs_table : ObsTable, optional
            Table containing the observation parameters, including all
            parameters needed to compute the noise.
        indices : array_like of int, optional
            Indices of the observations in the ObsTable to which noise should
            be applied.
        rng : np.random.Generator, optional
            The random number generator to use for applying noise. If None,
            a default generator will be used.
        **kwargs
            Additional parameters for the noise model.

        Returns
        -------
        flux : array_like
            The updated flux measuements after applying noise, in the same
            units as the input bandflux.
        flux_err : array_like
            The bandflux measurement error used for applying noise, in the
            same units as the input bandflux.
        """
        if obs_table is None:
            raise ValueError("ObsTable must be provided for PoissonFluxNoiseModel.")
        if indices is None:
            raise ValueError("Indices must be provided for PoissonFluxNoiseModel.")
        if len(indices) != len(bandflux):
            raise ValueError("Length of indices must match length of bandflux.")

        flux_err = self.compute_flux_error(
            bandflux,
            obs_table=obs_table,
            indices=indices,
        )

        # Make sure the array is a numpy array.
        flux_err = np.asarray(flux_err)

        # Generate the actual noisy bandflux measurements.
        if rng is None:
            rng = np.random.default_rng()
        noisy_bandflux = rng.normal(loc=bandflux, scale=flux_err)
        return noisy_bandflux, flux_err
