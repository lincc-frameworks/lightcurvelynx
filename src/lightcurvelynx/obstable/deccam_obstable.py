from lightcurvelynx.noise_models.base_noise_models import PoissonFluxNoiseModel
from lightcurvelynx.obstable.obs_table import ObsTable


class DECamPoissonFluxNoiseModel(PoissonFluxNoiseModel):
    """A subclass of PoissonFluxNoiseModel for DECam survey data."""

    def __init__(self):
        super().__init__()

    def compute_flux_error(self, bandflux, obs_table, indices):
        """Compute the flux error for the given bandflux and observation parameters.

        Parameters
        ----------
        bandflux : array_like of float
            Source bandflux in nJy.
        obs_table : ObsTable
            Table containing the observation parameters needed to compute the noise.
        indices : array_like of int
            Indices of the observations in the ObsTable for which to compute the noise.

        Returns
        -------
        flux_err : array_like
            The standard deviation of the bandflux measurement error (in nJy)
        """
        raise NotImplementedError(
            "The DECamPoissonFluxNoiseModel is not yet implemented. This method needs to "
            "be implemented to compute the flux error for DECam observations."
        )


class DECamObsTable(ObsTable):
    """A subclass for DECam exposure table.

    Parameters
    ----------
    table : dict or pandas.core.frame.DataFrame
        The table with all the observation information.
    colmap : dict
        A mapping of standard column names to a list of possible names in the input table.
        Each value in the dictionary can be a string or a list of strings.
        Defaults to the DECam column names, stored in _default_colnames.
    saturation_mags : dict, optional
        A dictionary mapping filter names to their saturation thresholds in magnitudes. The filters
        provided must match those in the table. If not provided, DECam-specific defaults will be used.
    noise_model : NoiseModel, optional
        The noise model to use for this ObsTable. If not provided, defaults to
        DECamPoissonFluxNoiseModel.
    **kwargs : dict
        Additional keyword arguments to pass to the ObsTable constructor. This includes overrides
        for survey parameters such as:

        - dark_current : The dark current for the camera in electrons per second per pixel.
        - gain: The CCD gain (in e-/ADU).
        - pixel_scale: The pixel scale for the camera in arcseconds per pixel.
        - radius: The angular radius of the observations (in degrees).
        - read_noise: The standard deviation of the count of readout electrons per pixel.
    """

    # Default column names for the DECam survey data with data extracted from qcinv files.
    # See helper_tools/Build_DECAT_Dataset.ipynb for details on how these were derived.
    _default_colnames = {
        "maglim": "maglim",
        "sky": "sky",
        "dec": "dec",
        "exptime": "exptime",
        "filter": "filter",
        "ra": "ra",
        "time": "time",
        "zp": "zp",  # We add this column to the table
    }

    # Constant from the DECam camera specifications. See
    # https://noirlab.edu/science/programs/ctio/instruments/Dark-Energy-Camera/characteristics
    _default_survey_values = {
        "dark_current": 0.0,  # electrons per second per pixel
        "gain": 4.0,  # e-/ADU
        "pixel_scale": 0.263,  # arcseconds per pixel
        "radius": 1.1,  # degrees
        "read_noise": 7,  # electrons per pixel
        "zp_err_mag": 0,  # Place holder
        "survey_name": "DECam",
    }

    def __init__(
        self,
        table,
        colmap=None,
        saturation_mags=None,
        noise_model=None,
        **kwargs,
    ):
        colmap = self._default_colnames if colmap is None else colmap

        # If noise model is not provided, then set to the DECam default.
        if noise_model is None:
            noise_model = DECamPoissonFluxNoiseModel()

        super().__init__(
            table,
            colmap=colmap,
            saturation_mags=saturation_mags,
            noise_model=noise_model,
            **kwargs,
        )

    def _assign_zero_points(self):
        """Assign instrumental zero points in ADU to the ObsTable."""
        raise NotImplementedError(
            "The DECamObsTable is not yet implemented. This method needs to be implemented "
            "to assign zero points for DECam observations."
        )
