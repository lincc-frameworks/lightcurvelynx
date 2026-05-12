"""The SurveyInfo module collects the information about the survey characteristics, including
where the survey was pointing, instrument characteristics, and bandpass information. It is primarily
a data class that is used to keep the pieces of information together.

Key components of the SurveyInfo class include:
- `ObsTable`: A table that contains the observation information, such as pointing coordinates,
  time of observation, and other relevant metadata. This also contains instrument information,
  such as the detector footprint and pixel scale.
- `Bandpass`: A class that contains the information about the bandpass of the instrument,
  including the wavelength range and the transmission curve.
- `NoiseModel`: A computation class for calculating the noise from characteristics of the survey.
"""

import logging

from lightcurvelynx.obstable.obs_table import ObsTable


class SurveyInfo:
    """The SurveyInfo class collects the information about the survey characteristics, including
    where the survey was pointing, instrument characteristics, and bandpass information.

    Attributes
    ----------
    obstable : ObsTable
        A table that contains the observation information, such as pointing coordinates,
        time of observation, and other relevant metadata. This also contains instrument information,
        such as the detector footprint and pixel scale.
    passbands : PassbandGroup, optional
        A class that contains the information about the bandpass of the instrument for each filter,
        including the wavelength range and the transmission curve. This is unused for spectroscopic surveys.
    noise_model : NoiseModel
        A computation class for calculating the noise from characteristics of the survey.
    name : str
        The name of the survey.
    """

    def __init__(self, obstable, *, passbands=None, noise_model=None, name=""):
        logger = logging.getLogger(__name__)

        if obstable is None or not isinstance(obstable, ObsTable):
            raise ValueError("obstable must be an instance of ObsTable.")
        self.obstable = obstable
        obstable_type = obstable.__class__.__name__
        logger.info(f"Initialized SurveyInfo with ObsTable type {obstable_type}.")

        # Set the passband group, using the default if not provided
        if passbands is not None:
            self.passbands = passbands
        else:
            logger.info(f"Using default passband group for ObsTable type {obstable_type}")
            self.passbands = obstable.default_passband_group

        # Set the noise model, using the default if not provided
        if noise_model is not None:
            self.noise_model = noise_model
        else:
            logger.info(f"Using default noise model for ObsTable type {obstable_type}")
            self.noise_model = obstable.default_noise_model
        self.name = name
