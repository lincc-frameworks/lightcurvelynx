"""Models to simulate M-Dwarf Flares."""

import numpy as np

from lightcurvelynx.astro_utils.black_body import black_body_luminosity_density_per_solid
from lightcurvelynx.models.physical_model import SEDModel


class MDwarfFlareModel(SEDModel):
    """An M-Dwarf Flare Model. The M-dwarf is modeled as the sum of several blackbody
    functions (each of which has a temperature and a peak amplitude):

    * The underlying star (amplitude is constant).
    * The hot flare peak
    * The cold flare peak

    Parameterized values include:

    * dec - The object's declination in degrees. [from BasePhysicalModel]
    * distance - The object's luminosity distance in pc. [from BasePhysicalModel]
    * ra - The object's right ascension in degrees. [from BasePhysicalModel]
    * redshift - The object's redshift. [from BasePhysicalModel]
    * t0 - The t0 of the zero phase, date. [from BasePhysicalModel. Not used.]

    Parameters
    ----------
    star_temp : parameter, optional
        The temperature of the star in Kelvins.
    star_radius : parameter, optional
        The radius of the star in cm.
    hot_flare_temp : parameter, optional
        The temperature of the hot flare in Kelvins.
    cold_flare_temp : parameter, optional
        The temperature of the cold flare in Kelvins.
    flare_radius : parameter, optional
        The radius of the combined hot and cold flare???
    """

    def __init__(
        self,
        *,
        star_temp=None,
        star_radius=None,
        hot_flare_temp=None,
        cold_flare_temp=None,
        flare_radius=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Add each model parameter, so it can be supported by the sampler.
        self.add_parameter(
            "star_temp", value=star_temp, description="The temperature of the star in Kelvins."
        )
        self.add_parameter("star_radius", value=star_radius, description="The radius of the star in cm.")
        self.add_parameter(
            "hot_flare_temp", value=hot_flare_temp, description="The temperature of the hot flare in Kelvins."
        )
        self.add_parameter(
            "cold_flare_temp",
            value=cold_flare_temp,
            description="The temperature of the cold flare in Kelvins.",
        )
        self.add_parameter(
            "flare_radius",
            value=flare_radius,
            description="The radius of the combined hot and cold flare in cm.",
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
        params = self.get_local_params(graph_state)
        num_times = len(times)
        num_waves = len(wavelengths)

        flux_density = np.zeros((num_times, num_waves))

        # If the star's temperature is defined, add it to the result flux.
        if params["star_temp"] is not None:
            flux_density += black_body_luminosity_density_per_solid(
                wavelengths,
                params["star_temp"],
                params["star_radius"],
            )

        # Compute the size of the flare for each phase (time relative to t0).
        # Compute and add in the contribution of the hot and cold flare
        # using those blackbody curves.
        # phase = times - params["t0"]

        # Return the result.
        return flux_density
