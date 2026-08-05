"""Noise model modifiers are noise models that wrap and transform other noise models. They can be
used to apply additional transformations to the noise parameters computed by a base noise model, such
as scaling or adding additional noise components. You can stack multiple noise model modifiers to
create complex noise models that combine different noise sources and transformations.
"""

import numpy as np

from lightcurvelynx.noise_models.base_noise_models import FluxNoiseModel


class LinearTransformFluxNoiseModel(FluxNoiseModel):
    """A noise model that transforms the flux error computed by a base noise model using a
    linear function: flux_err_scaled = scale_factor * flux_err_base + flux_err_offset, where
    flux_err_base is the flux error computed by the base noise model.

    Attributes
    ----------
    base_noise_model : FluxNoiseModel
        The base noise model to which the scaling will be applied.
    scale_factor : float, optional
        The factor by which to scale the flux error computed by the base noise model. Default is 1.0.
    flux_err_offset : float, optional
        The additive offset to the flux error computed by the base noise model. Default is 0.0.
    """

    def __init__(self, base_noise_model, scale_factor=1.0, flux_err_offset=0.0):
        if not isinstance(base_noise_model, FluxNoiseModel):  # pragma: no cover
            raise TypeError("base_noise_model must be an instance of FluxNoiseModel.")
        self.base_noise_model = base_noise_model
        self.scale_factor = scale_factor
        self.flux_err_offset = flux_err_offset

        # Copy the required values from the base noise model to ensure compatibility with ObsTable.
        self._required_values = base_noise_model.required_values

    def compute_flux_error(self, bandflux, **kwargs):
        """Compute a base flux error for the given bandflux and observation parameters, and then
        scale it by the scale_factor and add the flux_err_offset.

        Parameters
        ----------
        bandflux : array_like of float
            Source bandflux in nJy.
        **kwargs
            Additional parameters for the noise model.

        Returns
        -------
        flux_err : array_like
            The standard deviation of the bandflux measurement error (in nJy)
        """
        flux_err_base = np.asarray(self.base_noise_model.compute_flux_error(bandflux, **kwargs))
        return self.scale_factor * flux_err_base + self.flux_err_offset
