import logging

from lightcurvelynx.obstable.obs_table import ObsTable

logger = logging.getLogger(__name__)


class FakeObsTable(ObsTable):
    """A subclass for a (simplified) fake survey. The user must provide a constant
    flux error (bandflux_error) to use or enough information to compute the provided noise model.

    The class uses a flexible deriver to try to compute any missing parameters needed from
    what is provided.

    Defaults are set for other parameters (e.g. exptime, nexposure, read_noise, dark_current), which
    the user can override with keyword arguments to the constructor.

    Parameters
    ----------
    table : dict or pandas.core.frame.DataFrame
        The table with all the ObsTable information.  Must have columns
        "time", "ra", "dec", and "filter".
    colmap : dict, optional
        A mapping of standard column names to a list of possible names in the input table.
        Each value in the dictionary can be a string or a list of strings.
    bandflux_error : float or dict, optional
        If provided, use this constant flux error (in nJy) for all observations (overriding
        the normal noise compuation). A value of 0.0 will produce a noise-free simulation.
        If a dictionary is provided, it should map filter names to constant flux errors per-band.
        This setting should primarily be used for testing purposes.
    radius : float, optional
        The angular radius of the field of view of the observations in degrees (default=None).
    saturation_mags : dict, optional
        A dictionary mapping filter names to their saturation thresholds in magnitudes. The filters
        provided must match those in the table. If not provided, saturation effects will not be applied.
    **kwargs : dict
        Additional keyword arguments to pass to the ObsTable constructor. This includes overrides
        for survey parameters, such as "survey_name" or "read_noise".
    """

    # Default survey values.
    _default_survey_values = {
        "dark_current": 0,
        "exptime": 30,  # seconds
        "fwhm_px": None,  # pixels
        "nexposure": 1,  # exposures
        "radius": None,  # degrees
        "read_noise": 0,  # electrons
        "sky_bg_electrons": None,  # electrons / pixel^2
        "survey_name": "FAKE_SURVEY",
    }

    def __init__(
        self,
        table,
        *,
        colmap=None,
        bandflux_error=None,
        radius=None,
        saturation_mags=None,
        **kwargs,
    ):
        # Pass along all the survey parameters to the parent class.
        super().__init__(
            table,
            colmap=colmap,
            bandflux_error=bandflux_error,
            radius=radius,
            saturation_mags=saturation_mags,
            **kwargs,
        )
