"""
Wrapper classes for models constructed using the gopreaux (Gaussian process Optimized Photometric
Regression of Extragalactic Archival Ultraviolet-infrared eXplosions) package.

Adapted from: https://github.com/crpellegrino/gopreaux/blob/main/src/caat/SNModel.py
"""
import logging
import pickle
from pathlib import Path

import numpy as np
from astropy.io import fits
from citation_compass import CiteClass

from lightcurvelynx.models.physical_model import SEDModel
from lightcurvelynx.models.sed_template_model import SEDTemplate


class GoPreauxModel(SEDModel, CiteClass):
    """A class that can load and query models constructed using the gopreaux
    (Gaussian process Optimized Photometric Regression of Extragalactic Archival Ultraviolet-infrared
    eXplosions) package.

    Parameterized values include:
      * dec - The object's declination in degrees. [from BasePhysicalModel]
      * distance - The object's luminosity distance in pc. [from BasePhysicalModel]
      * ra - The object's right ascension in degrees. [from BasePhysicalModel]
      * redshift - The object's redshift. [from BasePhysicalModel]
      * t0 - The t0 of the zero phase, date. [from BasePhysicalModel]
    Additional parameterized values are used for specific gopreaux models.

    References
    ----------
    * Gaussian process Optimized Photometric Regression of Extragalactic Archival Ultraviolet-infrared
      eXplosions (GoPreaux):

    Attributes
    ----------
    model : SEDTemplate or GaussianProcessRegressor
        The model that will be used to generate the SED surface.
    template_mags: SEDTemplate or None
        The magnitudes used to normalize the input photometry prior to fitting.
    phases: np.ndarray, optional
        The phases used to produce the final SED surface model.
    wavelengths: np.ndarray, optional
        The wavelengths used to produce the final SED surface model.
    log_transform: bool or int or float
        The value of log_transform used in the fitting, if one was used.

    Parameters
    ----------
    model : gopreaux.SurfaceArray, SEDTemplate, or GaussianProcessRegressor
        The model that will be used to generate the SED surface.
    template_mags: np.ndarray, optional
        The magnitudes used to normalize the input photometry prior to fitting.
        Default: None.
    phases: np.ndarray, optional
        The phases used to produce the final SED surface model.
        Default: None.
    wavelengths: np.ndarray, optional
        The wavelengths used to produce the final SED surface model.
        Default: None.
    log_transform: bool or int or float
        The value of log_transform used in the fitting, if one was used.
        Default: False
    **kwargs : dict, optional
        Any additional keyword arguments.
    """

    def __init__(
        self,
        model,
        *,
        template_mags=None,
        phases=None,
        wavelengths=None,
        log_transform=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(model, SEDTemplate):
            self.model = model
            self._using_sed_template = True

            # Check phase and wavelength if given.
            if phases is not None and not np.allclose(model.times, phases):
                raise ValueError("Given phases provided do not match those in the SEDTemplate model.")
            if wavelengths is not None and not np.allclose(model.wavelengths, wavelengths):
                raise ValueError("Given wavelengths provided do not match those in the SEDTemplate model.")
            self.phases = model.times
            self.wavelengths = model.wavelengths

        elif np.all([hasattr(model, attr) for attr in ["phase_grid", "wl_grid", "surface"]]):
            # If we have a SurfaceArray object, we transform that into an SEDTemplate object,
            # so we do not need to import gopreaux.
            self.model = SEDTemplate.from_components(
                phase_grid=model.phase_grid,
                wl_grid=model.wl_grid,
                surface=model.surface,
            )

            # Check phase and wavelength if given.
            if phases is not None and not np.allclose(model.phase_grid, phases):
                raise ValueError("Given phases provided do not match those in the SurfaceArray model.")
            if wavelengths is not None and not np.allclose(model.wl_grid, wavelengths):
                raise ValueError("Given wavelengths provided do not match those in the SurfaceArray model.")
            self.phases = model.phase_grid
            self.wavelengths = model.wl_grid

        else:
            # This is another model type (probably a GaussianProcessRegressor), so we just use it directly.
            # The user is responsible for unsuring the correct packages have been installed.
            self.model = model
            self._using_sed_template = False

            # Use the given phases and wavelengths.
            self.phases = phases
            self.wavelengths = wavelengths

        # If we have template mags, we transform that into a SEDTemplate object to support interpolation.
        if template_mags is not None:
            if self.phases is None or self.wavelengths is None:
                raise ValueError("Need to specify phases and wavelengths if using template mags.")
            self.template_mags = SEDTemplate.from_components(
                phase_grid=self.phases,
                wl_grid=self.wavelengths,
                surface=template_mags,
            )
        else:
            self.template_mags = None

        self.log_transform = log_transform

    @classmethod
    def load_from_fits(cls, filename):
        """
        Load the data for gopreaux.SNModel model from a .fits file and use it
        to create the GoPreauxModel object.

        Parameters
        ----------
        filename: str, Path
            The complete path to the .fits file.
        """
        logging.getLogger(__name__).info(f"Loading gopreaux model from {filename}...")
        filename = Path(filename)
        if not filename.exists():
            raise FileNotFoundError(f"File {filename} does not exist.")

        with fits.open(filename) as hdul:
            # The first (0th) HDU contains the pickled surface model which could be
            # a GaussianProcessRegressor or a gopreaux.SurfaceArray.
            model = pickle.loads(hdul[0].data)
            log_transform = hdul[0].header["LOG_TRANSFORM"]

            # The second layer should be the magntiude template, and the third and fourth layers
            # should be the phase and wavelength grids, respectively. The later three layers are optional.
            template = hdul[1].data if len(hdul) > 1 else None
            phase_grid = hdul[2].data if len(hdul) > 2 else None
            wl_grid = hdul[3].data if len(hdul) > 3 else None

        return cls(
            model=model,
            template_mags=template,
            phases=phase_grid,
            wavelengths=wl_grid,
            log_transform=log_transform,
        )

    def compute_sed(self, times, wavelengths, graph_state):
        """Draw effect-free observer frame flux densities.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of observer frame timestamps in MJD.
        wavelengths : numpy.ndarray, optional
            A length N array of observer frame wavelengths (in angstroms).
        graph_state : GraphState
            An object mapping graph parameters to their values.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of observer frame SED values (in nJy).
        """
        phases = times - self.get_param(graph_state, "t0")

        if self._using_sed_template:
            result_mag = self.template.evaluate_sed(phases, wavelengths)
        else:
            # Evaluate the given model (GaussianProcessRegressor)
            result_mag, _ = self.model.predict(np.vstack((phases, wavelengths)).T, return_std=False)

        # Add on the template magnitudes if we have them.
        if self.template_mags is not None:
            template_addition = self.template_mags.evaluate_sed(phases, wavelengths)
            result_mag += template_addition

        return result_mag
