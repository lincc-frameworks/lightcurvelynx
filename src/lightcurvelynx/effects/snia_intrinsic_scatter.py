"""Intrinsic scatter for SN Ia"""

import os
from pathlib import Path

import numpy as np
import sncosmo
from lightcurvelynx.effects.effect_model import EffectModel

_C11_COVARIANCE = np.array([
    [+1.00, -0.12, -0.77, -0.91, -0.22],
    [-0.12, +1.00, +0.57, -0.24, -0.89],
    [-0.77, +0.57, +1.00, +0.53, -0.40],
    [-0.91, -0.24, +0.53, +1.00, +0.49],
    [-0.22, -0.89, -0.40, +0.49, +1.00],
])

_C11_DIAG = np.array([0.06, 0.04, 0.05, 0.04, 0.08])

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

    def _read_g10_color_dispersion_grid(self, sourcename="salt2"):
        # This function reads in the color dispersion for the G10 model from Guy et al. (2010). 
        modeldir = "/Users/mi/.astropy/cache/sncosmo/models/salt2/salt2-4"
        color_dispersion_file = os.path.join(modeldir, "salt2_color_dispersion.dat")
        g10_color_dispersion_grid = np.loadtxt(color_dispersion_file, unpack=True)
        return g10_color_dispersion_grid

    def get_intrinsic_scatter_covariance(self, wavelengths):
    
        if self.modelpars["modelname"] == "COH":
            # This should act as gray dust, with the same scatter at all wavelengths.
            sigma = self.modelpars.get("sigma", 0.1)
            return sigma ** 2 * np.eye(len(wavelengths))
        
        elif self.modelpars["modelname"] == "G10":
            # This applies a wavelength-dependent scatter that follows the G10 model from Guy et al. (2010). 
            g10_color_dispersion_grid = self._read_g10_color_dispersion_grid(self.modelpars.get("sourcename", "salt2"))
            g10_color_dispersion = np.interp(wavelengths, g10_color_dispersion_grid[0], g10_color_dispersion_grid[1])   
            coh_sigma = self.modelpars.get("coh_sigma", 0.09)
            return coh_sigma ** 2 * np.eye(len(wavelengths)) + np.diag(g10_color_dispersion ** 2)
            
        elif self.modelpars["modelname"] == "C11":
            # This applies a wavelength-dependent scatter that follows the C11 model from Chotard et al. (2011). 
            pass
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
        modelpars : dict, optional
            A dictionary of parameters needed to apply the effect. Should include the key "modelname" to specify the intrinsic scatter model to use, and any additional parameters needed for that model (e.g., "sigma" for the COH model).
        **kwargs : `dict`, optional
           Any additional keyword arguments, including any additional
           parameters needed to apply the effect.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of flux densities after the effect is applied (in nJy).
        """
        intrinsic_scatter_covariance = self.get_intrinsic_scatter_covariance(wavelengths)

        # Add Gaussian noise with mean 0 and standard deviation intrinsic_scatter_sigma to each flux density value
        flux_scatter = np.random.multivariate_normal(mean=np.zeros(flux_density.shape[1]), cov=intrinsic_scatter_covariance, size=flux_density.shape[0])
        return flux_density * np.power(10, -0.4 * flux_scatter)