"""Intrinsic scatter for SN Ia"""

from pathlib import Path

import numpy as np
import sncosmo
from lightcurvelynx.effects.effect_model import EffectModel

class SNIaIntrinsicScatter(EffectModel):
    """An effect model for intrinsic scatter in SN Ia.

    Attributes
    ----------
    modelname : parameter
        The name of the intrinsic scatter model.
    """

    def __init__(self, modelpars, **kwargs):
        super().__init__(**kwargs)
        self.add_effect_parameter("modelpars", modelpars)

    def _read_g10_coefficients(self):
        # This function reads in the polynomial coefficients for the G10 model from Guy et al. (2010). 
        salt2_source = sncosmo.SALT2Source()
        color_dispersion_file = Path(salt2_source.modeldir) / Path(salt2_source.cdfile)
        g10_coefficients = np.loadtxt(color_dispersion_file, skiprows=1)
        return g10_coefficients  

    def get_intrinsic_scatter_covariance(self, wavelengths, modelpars=None):
        if modelpars is None:
            modelpars = {"modelname": "COH", "sigma": 0.1}

        if modelpars["modelname"] == "COH":
            # This should act as gray dust, with the same scatter at all wavelengths.
            return modelpars["sigma"] ** 2 * np.eye(len(wavelengths))
        elif modelpars["modelname"] == "G10":
            # This applies a wavelength-dependent scatter that follows the G10 model from Guy et al. (2010). 
            # First we read in the polynomial coefficients for the G10 model from Guy et al. (2010). The G10 model is defined as a 4th order polynomial in wavelength.
            g10_coefficients = self._read_g10_coefficients()
            # Then we evaluate the polynomial at the given wavelengths to get the scatter at each wavelength.
            scatter_at_wavelengths = np.polyval(g10_coefficients, wavelengths)
            # Finally, we construct the covariance matrix as a diagonal matrix with the scatter at each wavelength on the diagonal.
            return np.diag(scatter_at_wavelengths ** 2)
            
        elif modelpars["modelname"] == "C11":
            # This applies a wavelength-dependent scatter that follows the C11 model from Chotard et al. (2011). 
            pass
        else:
            raise ValueError(f"Unknown intrinsic scatter model: {modelpars['modelname']}")
        
    def apply(
        self,
        flux_density,
        times=None,
        wavelengths=None,
        modelpars=None,
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
        intrinsic_scatter_covariance = self.get_intrinsic_scatter_covariance(wavelengths, modelpars=modelpars)

        # Add Gaussian noise with mean 0 and standard deviation intrinsic_scatter_sigma to each flux density value
        flux_scatter = np.random.multivariate_normal(mean=np.zeros(flux_density.shape[1]), cov=intrinsic_scatter_covariance, size=flux_density.shape[0])
        return flux_density + flux_scatter