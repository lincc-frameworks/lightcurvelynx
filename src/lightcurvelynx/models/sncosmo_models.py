"""Wrappers for the models defined in sncosmo.

https://github.com/sncosmo/sncosmo/blob/v2.10.1/sncosmo/models.py
https://sncosmo.readthedocs.io/en/stable/models.html
"""

import copy

from astropy import units as u
from citation_compass import CiteClass

from lightcurvelynx.astro_utils.unit_utils import flam_to_fnu
from lightcurvelynx.models.physical_model import SEDModel


class SncosmoWrapperModel(SEDModel, CiteClass):
    """A wrapper for sncosmo models.

    Parameterized values include:

    * dec - The object's declination in degrees. [from BasePhysicalModel]
    * distance - The object's luminosity distance in pc. [from BasePhysicalModel]
    * ra - The object's right ascension in degrees. [from BasePhysicalModel]
    * redshift - The object's redshift. [from BasePhysicalModel]
    * t0 - The t0 of the zero phase, date. [from BasePhysicalModel]

    Additional parameterized values are used for specific sncosmo models.

    References
    ----------
    * sncosmo - https://zenodo.org/records/14714968
    * Individual models might require citation. See references in the sncosmo documentation.

    Attributes
    ----------
    source : sncosmo.Source
        The underlying source model.
    source_name : str
        The name used to set the source.
    source_param_names : list
        A list of the source model's parameters that we need to set.

    Parameters
    ----------
    source_name : str
        The name used to set the source.
    node_label : str, optional
        An identifier (or name) for the current node.
    wave_extrapolation : FluxExtrapolationModel or tuple, optional
        The extrapolation model(s) to use for wavelengths that fall outside the model's defined
        bounds. If a tuple is provided, then it is expected to be of the form (before_model, after_model)
        where before_model is the model for before the first valid wavelength and after_model is
        the model for after the last valid wavelength. If None is provided the model will not try to
        extrapolate, but rather call compute_sed() for all wavelengths.
    time_extrapolation : FluxExtrapolationModel or tuple, optional
        The extrapolation model(s) to use for times that fall outside the model's defined
        bounds. If a tuple is provided, then it is expected to be of the form (before_model, after_model)
        where before_model is the model for before the first valid time and after_model is
        the model for after the last valid time. If None is provided the model will not try to
        extrapolate, but rather call compute_sed() for all times.
    **kwargs : dict, optional
        Any additional keyword arguments.
    """

    # A class variable for the units so we are not computing them each time.
    _FLAM_UNIT = u.erg / u.second / u.cm**2 / u.AA

    def __init__(
        self,
        source_name,
        node_label=None,
        wave_extrapolation=None,
        time_extrapolation=None,
        **kwargs,
    ):
        try:
            from sncosmo.models import get_source
        except ImportError as err:  # pragma: no cover
            raise ImportError(
                "sncosmo package is not installed by default. To use the SncosmoWrapperModel, "
                "please install sncosmo. For example, you can install it with "
                "`pip install sncosmo` or `conda install conda-forge::sncosmo`."
            ) from err

        # We explicitly ask for and pass along the PhysicalModel parameters such
        # as node_label and wave_extrapolation so they do not go into kwargs
        # and get added to the sncosmo model below.
        super().__init__(
            node_label=node_label,
            wave_extrapolation=wave_extrapolation,
            time_extrapolation=time_extrapolation,
            **kwargs,
        )
        self.source_name = source_name
        self.source = get_source(source_name)

        # Use the kwargs to initialize the sncosmo model's parameters.
        self.source_param_names = []
        for key, value in kwargs.items():
            if key not in self.setters:
                self.add_parameter(key, value, description="Parameter for sncosmo model.")
            if key in self.source.param_names:
                self.source_param_names.append(key)

    @property
    def param_names(self):
        """Return a list of the model's parameter names."""
        return self.source.param_names

    def minphase(self, **kwargs):
        """Get the minimum phase of the model (in days relative to t0).

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments, not used in this method.

        Returns
        -------
        minphase : float or None
            The minimum phase of the model (in days relative to t0) or None
            if the model does not have a defined minimum phase.
        """
        return self.source.minphase()

    def maxphase(self, **kwargs):
        """Get the maximum phase of the model (in days relative to t0).

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments, not used in this method.

        Returns
        -------
        maxphase : float or None
            The maximum phase of the model (in days relative to t0) or None
            if the model does not have a defined maximum phase.
        """
        return self.source.maxphase()

    def minwave(self, **kwargs):
        """Get the minimum wavelength of the model.

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
        return self.source.minwave()

    def maxwave(self, **kwargs):
        """Get the maximum wavelength of the model.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments, not used in this method.

        Returns
        -------
        maxwave : float or None
            The maximum wavelength of the model (in angstroms) or None
            if the model does not have a defined maximum wavelength.
        """
        return self.source.maxwave()

    def compute_sed(self, times, wavelengths, graph_state=None, **kwargs):
        """Draw effect-free observations for this object.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of rest frame timestamps.
        wavelengths : numpy.ndarray, optional
            A length N array of wavelengths (in angstroms).
        graph_state : GraphState
            An object mapping graph parameters to their values.
        **kwargs : dict, optional
           Any additional keyword arguments.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of SED values (in nJy).
        """
        params = self.get_local_params(graph_state)

        # Create a local copy of the source to avoid modifying the original one
        # and set the sampled parameters for this local copy.
        local_source = copy.deepcopy(self.source)
        sn_params = {name: params[name] for name in self.source_param_names}
        local_source.set(**sn_params)

        # Provide debugging information about the wavelength range if the wavelengths are out of bounds.
        if wavelengths.min() < self.minwave() or wavelengths.max() > self.maxwave():
            raise ValueError(
                "Wavelengths are out of the sncosmo model's valid range. "
                f"Model wavelength range: [{self.minwave()}, {self.maxwave()}], "
                f"Query wavelength range: [{wavelengths.min()}, {wavelengths.max()}]. Use the "
                "'wave_extrapolation' parameter specify how to handle out-of-bounds wavelengths."
            )

        # Query the model and convert the output to nJy.
        phase = times - params["t0"]
        model_flam = local_source.flux(phase, wavelengths)
        model_fnu = flam_to_fnu(
            model_flam,
            wavelengths,
            wave_unit=u.AA,
            flam_unit=self._FLAM_UNIT,
            fnu_unit=u.nJy,
        )
        return model_fnu
