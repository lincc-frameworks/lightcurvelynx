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
        self._required_values = base_noise_model.required_values.copy()

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


class GivenLinearTransformFluxNoiseModel(FluxNoiseModel):
    """A noise model that transforms the flux error computed by a base noise model using a per-observation
    linear function: flux_err_scaled = scale_factor * flux_err_base + flux_err_offset, where
    flux_err_base is the flux error computed by the base noise model and scale_factor and flux_err_offset
    are extracted from the observation table for each observation.

    The scale_factor and flux_err_offset are expected to be provided in the observation table as columns
    with names specified by the scale_factor_col and flux_err_offset_col parameters, respectively. If
    these columns are not present in the observation table, default values of 1.0 for scale_factor and 0.0
    for flux_err_offset will be used.

    Attributes
    ----------
    base_noise_model : FluxNoiseModel
        The base noise model to which the scaling will be applied.
    scale_factor_col : str, optional
        The column name in the observation table that provides the
        factor by which to scale the flux error computed by the base noise model. Default is "scale_factor".
    flux_err_offset_col : str, optional
        The column name in the observation table that provides the
        additive offset to the flux error computed by the base noise model. Default is "flux_err_offset".
    """

    def __init__(
        self, base_noise_model, scale_factor_col="scale_factor", flux_err_offset_col="flux_err_offset"
    ):
        if not isinstance(base_noise_model, FluxNoiseModel):  # pragma: no cover
            raise TypeError("base_noise_model must be an instance of FluxNoiseModel.")
        self.base_noise_model = base_noise_model
        self.scale_factor_col = scale_factor_col
        self.flux_err_offset_col = flux_err_offset_col

        # Copy the required values from the base noise model to ensure compatibility with ObsTable.
        self._required_values = base_noise_model.required_values.copy()

    def compute_flux_error(self, bandflux, *, obs_table=None, indices=None, **kwargs):
        """Compute a base flux error for the given bandflux and observation parameters, and then
        scale it by the scale_factor and add the flux_err_offset extracted from the observation table for
        each observation.

        Parameters
        ----------
        bandflux : array_like of float
            Source bandflux in nJy.
        obs_table : ObsTable
            Table containing the observation parameters needed to compute the noise.
        indices : array_like of int
            Indices of the observations in the ObsTable for which to compute the noise.
        **kwargs
            Additional parameters for the noise model.

        Returns
        -------
        flux_err : array_like
            The standard deviation of the bandflux measurement error (in nJy)
        """
        if obs_table is None:  # pragma: no cover
            raise ValueError("ObsTable must be provided for GivenLinearTransformFluxNoiseModel.")
        if indices is None:  # pragma: no cover
            raise ValueError("Indices must be provided for GivenLinearTransformFluxNoiseModel.")
        if len(indices) != len(bandflux):  # pragma: no cover
            raise ValueError("Length of indices must match length of bandflux.")

        # Start by computing the base flux error using the base noise model.
        flux_err_base = self.base_noise_model.compute_flux_error(
            bandflux,
            obs_table=obs_table,
            indices=indices,
            **kwargs,
        )
        flux_err_base = np.asarray(flux_err_base)

        # Extract the scale_factor and flux_err_offset from the observation table for each observation.
        scale_factor = obs_table.get_value_per_row(self.scale_factor_col, indices=indices, default=1.0)
        flux_err_offset = obs_table.get_value_per_row(self.flux_err_offset_col, indices=indices, default=0.0)
        return scale_factor * flux_err_base + flux_err_offset
