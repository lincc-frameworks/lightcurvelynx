"""
Wrapper classes for models constructed using the gopreaux (Gaussian process Optimized Photometric
Regression of Extragalactic Archival Ultraviolet-infrared eXplosions) package.

This model requires that the GoPreaux (caat) package is installed. Currently GoPreaux is not available
on PyPI, so users will need to install it from source: https://github.com/crpellegrino/gopreaux

There is a version conflict with numpy between GoPreaux and LightCurveLynx, but this does not impact
the functions we need. Users can install GoPreaux first and then install LightCurveLynx (upgrading all
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
      * brightness - The intrinsic brightness of the supernova at its peak and the wavelength closest
        to the V-band (in magnitudes). [specific to GoPreauxModel]
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
    intrinsic_brightness : Parameter
        The intrinsic brightness of the supernova at its peak and the wavelength closest
        to the V-band (in magnitudes).
    **kwargs : dict, optional
        Any additional keyword arguments.
    """

    def __init__(self, model, intrinsic_brightness, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.add_parameter(
            "brightness",
            intrinsic_brightness,
            description=(
                "The intrinsic brightness of the supernova at its peak and the wavelength closest"
                " to the V-band (in magnitudes).",
            ),
            **kwargs,
        )

    def minwave(self, **kwargs):
        """Get the minimum supported wavelength of the model.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments, not used in this method.

        Returns
        -------
        minwave : float or None
            The minimum wavelength of the model (in angstroms) or None
            if the model does not have a defined minimum wavelength.
        """
        # Add a small epsilon to the minimum wavelength to avoid issues with querying
        # the model at exactly the minimum wavelength.
        return self.model.min_wl + 1e-10

    def maxwave(self, **kwargs):
        """Get the maximum supported wavelength of the model.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments, not used in this method.

        Returns
        -------
        maximum : float or None
            The maximum wavelength of the model (in angstroms) or None
            if the model does not have a defined maximum wavelength.
        """
        # Subtract a small epsilon to the maximum wavelength to avoid issues with querying
        # the model at exactly the maximum wavelength.
        return self.model.max_wl - 1e-10

    def minphase(self, **kwargs):
        """Get the minimum supported phase of the model in days.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments, not used in this method.

        Returns
        -------
        minphase : float or None
            The minimum phase of the model (in days) or None
            if the model does not have a defined minimum phase.
        """
        # Add a small epsilon to the minimum phase to avoid issues with querying
        # the model at exactly the minimum phase.
        return self.model.min_phase + 1e-10

    def maxphase(self, **kwargs):
        """Get the maximum supported phase of the model in days.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments, not used in this method.

        Returns
        -------
        maximum : float or None
            The maximum phase of the model (in days) or None
            if the model does not have a defined maximum phase.
        """
        # Subtract a small epsilon to the maximum phase to avoid issues with querying
        # the model at exactly the maximum phase.
        return self.model.max_phase - 1e-10

    @classmethod
    def load_from_fits(cls, filename, intrinsic_brightness, **kwargs):
        """
        Load the data for gopreaux.SNModel model from a .fits file and use it
        to create the GoPreauxModel object.

        Parameters
        ----------
        filename: str, Path
            The complete path to the .fits file.
        intrinsic_brightness: Parameter or callable or float
            The intrinsic brightness of the supernova at its peak and the wavelength closest
            to the V-band (in magnitudes). This may be provided as a fixed scalar value
            or as a parameterized/sampled node accepted by the constructor.
        **kwargs: dict, optional
            Any additional keyword arguments to be passed to the GoPreauxModel constructor.
        """
        if not isinstance(filename, str | Path):
            raise TypeError("filename must be a string or Path object.")

        logging.getLogger(__name__).info(f"Loading gopreaux model from {filename}...")
        filename = Path(filename)
        if not filename.exists():
            raise FileNotFoundError(f"File {filename} does not exist.")

        try:
            from caat import SNModel
        except ImportError as err:
            raise ImportError(
                "The gopreaux package is required to load GoPreauxModel objects. "
                "Please install it from source: https://github.com/crpellegrino/gopreaux"
            ) from err

        # Load the SNModel from the .fits file using gopreaux's constructor, which
        # can take a path to the fits file.
        model = SNModel(str(filename))
        return cls(model, intrinsic_brightness, **kwargs)

    def compute_sed(self, times, wavelengths, graph_state):
        """Draw effect-free observer frame flux densities.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of observer frame timestamps in MJD.
        wavelengths : numpy.ndarray
            A length N array of observer frame wavelengths (in angstroms).
        graph_state : GraphState
            An object mapping graph parameters to their values.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of observer frame SED values (in nJy).
        """
        # We check against the model's internal bounds for phase, which may be slightly different
        # than this object's minphase and maxphase.
        times = np.asarray(times)
        t0 = self.get_param(graph_state, "t0")

        if (np.min(times - t0) < self.model.min_phase) or (np.max(times - t0) > self.model.max_phase):
            raise ValueError(
                f"Times need to be within the bounds of the model: [{self.minphase() + t0}, "
                f"{self.maxphase() + t0}] MJD or a time extrapolation method must be provided "
                "during model creation using the 'time_extrapolation' parameter."
            )
        num_times = len(times)

        # We check against the model's internal bounds for wavelength, which may be slightly different
        # than this object's minwave and maxwave.
        wavelengths = np.asarray(wavelengths)
        if (np.min(wavelengths) < self.model.min_wl) or (np.max(wavelengths) > self.model.max_wl):
            raise ValueError(
                f"Wavelengths need to be within the bounds of the model: [{self.minwave()}, "
                f"{self.maxwave()}] angstroms or a wavelength extrapolation method must be "
                "provided during model creation using the 'wave_extrapolation' parameter."
            )
        num_wavelengths = len(wavelengths)

        # Take the single list of times and wavelengths and create a grid of all combinations of them,
        # into a grid of points on which we will query the model.
        phases, wavelengths = np.meshgrid(times - t0, wavelengths, indexing="ij")
        phases = phases.ravel()
        wavelengths = wavelengths.ravel()

        # Use the model's built-in predict_photometry_points function to get predictions at the
        # given phases and wavelengths.
        _, rel_mag, _ = self.model.predict_photometry_points(
            wavelengths=wavelengths,
            phases=phases,
            show=False,
        )
        rel_mag = rel_mag.reshape(num_times, num_wavelengths)

        # The results are returned in the delta magnitude (relative to the peak and the
        # wavelength closest to the V-band) where a delta of 1.0 indicates an increase
        # in brightness (and thus a decrease in magnitude) by 1.0. So we need to *subtract*
        # these changes from the intrinsic brightness.
        total_mag = self.get_param(graph_state, "brightness") - rel_mag

        # Convert from magnitudes to fluxes in nJy and return the result.
        return mag2flux(total_mag)
