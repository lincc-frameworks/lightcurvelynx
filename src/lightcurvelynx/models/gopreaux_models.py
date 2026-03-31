"""
Wrapper classes for models constructed using the gopreaux (Gaussian process Optimized Photometric
Regression of Extragalactic Archival Ultraviolet-infrared eXplosions) package.

This model requires that the GoPreaux (caat) package is installed. Currently GoPreaux is not available
on PyPI, so users will need to install it from source: https://github.com/crpellegrino/gopreaux

There is a version conflict with numpy between GoPreaux and LightCurveLynx, but this does not impact
the functions we need. Users can install GoPreax first and then install LightCurveLynx (upgrading all
dependencies). You will still get errors about the version requirements for caat, but they can be ignored.
"""
import logging
from pathlib import Path

import numpy as np
from citation_compass import CiteClass

from lightcurvelynx.astro_utils.mag_flux import mag2flux
from lightcurvelynx.models.physical_model import SEDModel


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
    model : caat.SNModel
        The wrapped model that will be used to generate the SED surface.

    Parameters
    ----------
    model : caat.SNModel
        The gopreaux SNModel object that defines the surface to be evaluated.
    **kwargs : dict, optional
        Any additional keyword arguments.
    """

    def __init__(self, model, **kwargs):
        self.model = model
        super().__init__(**kwargs)

    @classmethod
    def load_from_fits(cls, filename, **kwargs):
        """
        Load the data for gopreaux.SNModel model from a .fits file and use it
        to create the GoPreauxModel object.

        Parameters
        ----------
        filename: str, Path
            The complete path to the .fits file.
        **kwargs: dict, optional
            Any additional keyword arguments to be passed to the GoPreauxModel constructor.
        """
        assert isinstance(filename, str | Path), "filename must be a string or Path object."

        logging.getLogger(__name__).info(f"Loading gopreaux model from {filename}...")
        filename = Path(filename)
        if not filename.exists():
            raise FileNotFoundError(f"File {filename} does not exist.")

        try:
            from caat import SNModel
        except ImportError as err:
            raise ImportError(
                "The gopreaux package is required to load GoPreauxModel objects. "
                "Please install it from source: https://github.com/your-repo/gopreaux"
            ) from err

        # Load the SNModel from the .fits file using gopreaux's constructor, which
        # can take a path to the fits file.
        model = SNModel(str(filename))
        return cls(model, **kwargs)

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
        num_times = len(times)
        num_wavelengths = len(wavelengths)

        # Take the single list of times and wavelengths and create a grid of all combinations of them,
        # into a grid of points on which we will query the model.
        phases, wavelengths = np.meshgrid(
            np.asarray(times) - self.get_param(graph_state, "t0"),
            np.asarray(wavelengths),
            indexing="ij",
        )
        phases = phases.ravel()
        wavelengths = wavelengths.ravel()

        # Use the model's built-in predict_photometry_points function to get predictions at the
        # given phases and wavelengths.
        _, results_mag, _ = self.model.predict_photometry_points(phases, wavelengths, show=False)
        results_mag = results_mag.reshape(num_times, num_wavelengths)

        # The results are returned in magnitudes relative to the peak. We need to convert them to
        # flux densities in nJy.
        return mag2flux(results_mag)
