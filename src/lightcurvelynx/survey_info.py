"""The SurveyInfo module collects the information about the survey characteristics, including
where the survey was pointing, instrument characteristics, and bandpass information. It is primarily
a data class that is used to keep the pieces of information together.

Key components of the SurveyInfo class include:
- `obstable`: An `ObsTable` object that contains the observation information, such as pointing coordinates,
  time of observation, and other relevant metadata. This also contains instrument information,
  such as the detector footprint and pixel scale.
- `passbands`: A `PassbandGroup` object that contains the information about the bandpass of the instrument,
  including the wavelength range and the transmission curve, or a `Spectrograph` object for spectroscopic
  surveys. This is optional and will use a default if not provided.
- `noise_model`: A computation class for calculating the noise from characteristics of the survey.
"""

import logging

import numpy as np

from lightcurvelynx.astro_utils.passbands import PassbandGroup
from lightcurvelynx.astro_utils.spectrograph import Spectrograph
from lightcurvelynx.noise_models.base_noise_models import PoissonFluxNoiseModel
from lightcurvelynx.obstable.obs_table import ObsTable
from lightcurvelynx.obstable.roman_obstable import RomanPoissonFluxNoiseModel

logger = logging.getLogger(__name__)


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
    noise_model : NoiseModel, optional
        A computation class for calculating the noise from characteristics of the survey.
    survey_name : str, optional
        The name of the survey, which is used to determine the default passbands and noise model
        if they are not provided. You can use "None" for no defaults (for testing).
        If not provided, it will be inferred from the `obstable`.
    validate : bool, optional
        Whether to validate the SurveyInfo instance after initialization. This should be True
        for most runs, but can be set to False for testing.
    """

    def __init__(
        self,
        obstable,
        *,
        passbands=None,
        noise_model=None,
        survey_name=None,
        validate=True,
    ):
        if obstable is None or not isinstance(obstable, ObsTable):
            raise ValueError("obstable must be an instance of ObsTable.")
        self.obstable = obstable
        obstable_type = obstable.__class__.__name__
        logger.info(f"Initialized SurveyInfo with ObsTable type {obstable_type}.")

        # Save the survey name, passband group, and noise model, using defaults for
        # anything that is not provided.
        if survey_name is not None:
            survey_name = survey_name.lower()
        elif "survey_name" in obstable.survey_values:
            survey_name = obstable.survey_values["survey_name"].lower()
        else:
            survey_name = "unknown"
        self.survey_name = survey_name
        self.noise_model = noise_model if noise_model is not None else self._load_default_noise_model()
        self.passbands = passbands if passbands is not None else self._load_default_passbands()

        # Validate the SurveyInfo instance to ensure that the provided and default
        # values are mutually consistent and valid.
        if validate and self.survey_name != "none":
            self._validate()

    def _load_default_noise_model(self):
        """Load the default noise model for the given survey."""
        logger.info(f"Loading default noise model for survey {self.survey_name}")
        if self.survey_name in ["argus", "lsst", "skymapper", "ztf"]:
            return PoissonFluxNoiseModel()
        elif self.survey_name == "roman":
            return RomanPoissonFluxNoiseModel()
        elif self.survey_name == "spectrograph":
            return None  # We don't currently have a default noise model for spectrographs.
        elif self.survey_name == "none":
            return None
        else:
            raise ValueError(
                f"Survey '{self.survey_name}' does not have a default noise model defined. "
                "Please provide a noise model when initializing the SurveyInfo instance."
            )

    def _load_default_passbands(self):
        """Load the default passbands for the given survey."""
        logger.info(f"Loading default passbands for survey {self.survey_name}")
        if self.survey_name == "lsst":
            return PassbandGroup.from_preset("LSST")
        elif self.survey_name == "roman":
            return PassbandGroup.from_preset("roman")
        elif self.survey_name == "skymapper":
            return PassbandGroup.from_svo("SkyMapper/SkyMapper")
        elif self.survey_name == "spectrograph":
            return None  # No passband group for spectrographs.
        elif self.survey_name == "ztf":
            return PassbandGroup.from_preset("ztf")
        elif self.survey_name == "none":
            return None
        else:
            raise ValueError(
                f"Survey '{self.survey_name}' does not have a default passband group defined. "
                "Please provide a passband group when initializing the SurveyInfo instance."
            )

    def _validate(self):
        """Check that the attributes of the SurveyInfo instance are mutually consistent and valid."""
        if self.noise_model is not None:
            self.noise_model.check_compatibility(self.obstable, fail_on_incompatible=True)

        if self.passbands is None:
            raise ValueError("passbands cannot be None.")
        elif not isinstance(self.passbands, Spectrograph):
            for filter_name in np.unique(self.obstable["filter"]):
                if filter_name not in self.passbands:
                    raise ValueError(f"Filter '{filter_name}' in ObsTable is not present in PassbandGroup.")
