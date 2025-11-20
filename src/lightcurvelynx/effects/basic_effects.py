"""A collection of toy effect models that are primarily used for testing."""

from lightcurvelynx.effects.effect_model import EffectModel


class ScaleFluxEffect(EffectModel):
    """An effect that scales the flux by a constant multiplicative amount.

    Attributes
    ----------
    flux_scale : parameter
        The multiplicative factor by which to scale the flux.
    """

    def __init__(self, flux_scale, **kwargs):
        super().__init__(**kwargs)
        self.add_effect_parameter("flux_scale", flux_scale)

    def apply(
        self,
        flux_density,
        times=None,
        wavelengths=None,
        flux_scale=None,
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
        flux_scale : float, optional
            The multiplicative factor by which to scale the flux. Raises an error if None is provided.
        **kwargs : `dict`, optional
           Any additional keyword arguments, including any additional
           parameters needed to apply the effect.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of flux densities after the effect is applied (in nJy).
        """
        if flux_scale is None:
            raise ValueError("flux_scale must be provided")
        return flux_density * flux_scale

    def apply_bandflux(
        self,
        bandfluxes,
        *,
        times=None,
        filters=None,
        flux_scale=None,
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
        flux_scale : float, optional
            The multiplicative factor by which to scale the flux. Raises an error if None is provided.
        **kwargs : `dict`, optional
           Any additional keyword arguments, including any additional
           parameters needed to apply the effect.

        Returns
        -------
        bandfluxes : numpy.ndarray
            A length T array of band fluxes after the effect is applied (in nJy).
        """
        if flux_scale is None:
            raise ValueError("flux_scale must be provided")
        if flux_scale < 0.0:
            raise ValueError("flux_scale cannot be negative.")

        return bandfluxes * flux_scale
