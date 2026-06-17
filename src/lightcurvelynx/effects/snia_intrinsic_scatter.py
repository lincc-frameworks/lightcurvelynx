"""Intrinsic scatter for SN Ia"""

import numpy as np
import sncosmo
from scipy.interpolate import RectBivariateSpline

from lightcurvelynx.effects.effect_model import EffectModel

# C11 model constants from Chotard et al. (2011) via SNANA sntools_genSmear.c.
# 6 bands: v(2500), U(3560), B(4390), V(5490), R(6545), I(8045) Angstroms.
# The v band (first row/column) is uncorrelated with others (OPT_farUV=0 default).
_C11_COVARIANCE = np.array(
    [
        [+1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
        [0.000000, +1.000000, -0.118516, -0.768635, -0.908202, -0.219447],
        [0.000000, -0.118516, +1.000000, +0.570333, -0.238470, -0.888611],
        [0.000000, -0.768635, +0.570333, +1.000000, +0.530320, -0.399538],
        [0.000000, -0.908202, -0.238470, +0.530320, +1.000000, +0.490134],
        [0.000000, -0.219447, -0.888611, -0.399538, +0.490134, +1.000000],
    ]
)
_C11_DIAG = np.array([0.5900, 0.06001, 0.040034, 0.050014, 0.040017, 0.080007])
_C11_KNOT_WAVELENGTHS = np.array([2500.0, 3560.0, 4390.0, 5490.0, 6545.0, 8045.0])
_C11_COV_SCALE = 1.3


class SNIaIntrinsicScatter(EffectModel):
    """An effect model for intrinsic scatter in SN Ia.

    Attributes
    ----------
    modelname : parameter
        The name of the intrinsic scatter model.
    """

    def __init__(self, modelpars, **kwargs):
        super().__init__(**kwargs)
        self.modelpars = modelpars

    def _get_g10_color_dispersion(self, sourcename="salt2"):
        # Returns a callable spline sigma(wavelength) loaded from the sncosmo source.
        source = sncosmo.get_source(sourcename)
        return source._colordisp

    def _interpolate_c11_covariance(self, wavelengths):
        """Interpolate the C11 covariance matrix onto an arbitrary wavelength grid.

        Parameters
        ----------
        wavelengths : np.ndarray
            Target wavelength grid (Angstroms), shape (N,)

        Returns
        -------
        np.ndarray
            Interpolated covariance matrix, shape (N, N)
        """
        cov_knots = _C11_COVARIANCE * np.outer(_C11_DIAG, _C11_DIAG) * _C11_COV_SCALE

        spline = RectBivariateSpline(
            _C11_KNOT_WAVELENGTHS,
            _C11_KNOT_WAVELENGTHS,
            cov_knots,
            kx=3,
            ky=3,
        )

        return spline(wavelengths, wavelengths)

    def get_intrinsic_scatter_covariance(self, wavelengths, modelpars=None):
        """Return the wavelength-dependent scatter covariance matrix for G10 or C11 models.

        Parameters
        ----------
        wavelengths : np.ndarray
            Target wavelength grid (Angstroms), shape (N,)
        modelpars : dict, optional
            Model parameters dict; defaults to self.modelpars if not provided.

        Returns
        -------
        np.ndarray
            Covariance matrix of shape (N, N).
        """
        if modelpars is None:
            modelpars = self.modelpars

        if modelpars["modelname"] == "G10":
            g10_colordisp = self._get_g10_color_dispersion(modelpars.get("sourcename", "salt2"))
            g10_sigma = g10_colordisp(wavelengths)
            coh_sigma = modelpars.get("coh_sigma", 0.09)
            return coh_sigma**2 * np.eye(len(wavelengths)) + np.diag(g10_sigma**2)

        elif modelpars["modelname"] == "C11":
            return self._interpolate_c11_covariance(wavelengths)

        else:
            raise ValueError(f"Unknown intrinsic scatter model: {modelpars['modelname']}")

    def apply(
        self,
        flux_density,
        times=None,
        wavelengths=None,
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
            A length N array of wavelengths (in angstroms).
        **kwargs : `dict`, optional
           Any additional keyword arguments. Pass ``modelpars`` to override
           the instance-level modelpars for this call.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of flux densities after the effect is applied (in nJy).
        """
        modelpars = {**self.modelpars, **kwargs.get("modelpars", {})}

        if modelpars["modelname"] == "COH":
            # Coherent scatter: one magnitude shift per epoch, same at all wavelengths.
            sigma = modelpars.get("sigma", 0.1)
            scatter = np.random.normal(0, sigma, size=flux_density.shape[0])
            return flux_density * np.power(10, -0.4 * scatter[:, np.newaxis])

        covariance = self.get_intrinsic_scatter_covariance(wavelengths, modelpars=modelpars)
        flux_scatter = np.random.multivariate_normal(
            mean=np.zeros(flux_density.shape[1]),
            cov=covariance,
            size=flux_density.shape[0],
        )
        return flux_density * np.power(10, -0.4 * flux_scatter)
