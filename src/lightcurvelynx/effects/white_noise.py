import numpy as np

from lightcurvelynx.effects.effect_model import EffectModel
from lightcurvelynx.math_nodes.np_random import NumpyRandomFunc


class WhiteNoise(EffectModel):
    """A white noise model.

    Attributes
    ----------
    white_noise_sigma : parameter
        The scale of the noise.
    """

    def __init__(self, white_noise_sigma, **kwargs):
        super().__init__(**kwargs)
        self.add_effect_parameter("white_noise_sigma", white_noise_sigma)

        # Instead of generating all the white noise values and storing them in the
        # graph state (which would be huge), we create a per model seed so we can
        # deterministically recreate the same noise values when needed.
        seed_generator = NumpyRandomFunc("integers", low=0, high=2**32 - 1)
        self.add_effect_parameter("white_noise_seed", seed_generator)

    def apply(
        self,
        flux_density,
        times=None,
        wavelengths=None,
        white_noise_sigma=None,
        white_noise_seed=None,
        **kwargs,
    ):
        """Apply the effect to observations (flux_density values).

        Parameters
        ----------
        flux_density : numpy.ndarray
            A length T X N matrix of flux density values (in nJy).
        times : numpy.ndarray, optional
            A length T array of times (in MJD). Not used for this effect.
        wavelengths : numpy.ndarray, optional
            A length N array of wavelengths (in angstroms). Not used for this effect.
        white_noise_sigma : float, optional
            The scale of the noise. Raises an error if None is provided.
        white_noise_seed : int, optional
            The seed for the random number generator. If None, a random seed is used.
        **kwargs : `dict`, optional
           Any additional keyword arguments, including any additional
           parameters needed to apply the effect.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of flux densities after the effect is applied (in nJy).
        """
        if white_noise_sigma is None:
            raise ValueError("white_noise_sigma must be provided")

        rng_info = np.random.default_rng(white_noise_seed)
        return rng_info.normal(loc=flux_density, scale=white_noise_sigma)

    def apply_bandflux(
        self,
        bandfluxes,
        *,
        times=None,
        filters=None,
        white_noise_sigma=None,
        white_noise_seed=None,
        **kwargs,
    ):
        """Apply the effect to band fluxes.

        Parameters
        ----------
        bandfluxes : numpy.ndarray
            A length T array of band fluxes (in nJy).
        times : numpy.ndarray, optional
            A length T array of times (in MJD).
        filters : numpy.ndarray, optional
            A length N array of filters. If not provided, the effect is applied to all
            band fluxes.
        white_noise_sigma : float, optional
            The scale of the noise. Raises an error if None is provided.
        white_noise_seed : int, optional
            The seed for the random number generator. If None, a random seed is used.
        **kwargs : `dict`, optional
           Any additional keyword arguments, including any additional
           parameters needed to apply the effect.

        Returns
        -------
        bandfluxes : numpy.ndarray
            A length T array of band fluxes after the effect is applied (in nJy).
        """
        if white_noise_sigma is None:
            raise ValueError("white_noise_sigma must be provided")

        rng_info = np.random.default_rng(white_noise_seed)
        return rng_info.normal(loc=bandfluxes, scale=white_noise_sigma)
