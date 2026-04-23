from abc import ABC, abstractmethod

import numpy as np


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
