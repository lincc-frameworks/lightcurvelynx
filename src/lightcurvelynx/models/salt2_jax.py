from pathlib import Path

from astropy import units as u
from citation_compass import CiteClass

from lightcurvelynx.astro_utils.salt2_color_law import SALT2ColorLaw
from lightcurvelynx.astro_utils.unit_utils import flam_to_fnu
from lightcurvelynx.models.physical_model import SEDModel


class SALT2JaxModel(SEDModel, CiteClass):
    """A SALT2 model implemented with JAX for it can use auto-differentiation.

    The model is defined in (Guy J., 2007) as:

    flux(time, wave) = x0 * [M0(time, wave) + x1 * M1(time, wave)] * exp(c * CL(wave))

    where x0, x1, and c are given parameters, M0 is the average spectral sequence,
    M1 is the first compoment to describe variability, and CL is the average color
    correction law.

    We use the formulation in sncosmo where CL is defined such that:

    flux(time, wave) = x0 * [M0(time, wave) + x1 * M1(time, wave)] * 10 ** (-0.4 * c * CL(wave))

    This class is based on the sncosmo implementation at:
    https://github.com/sncosmo/sncosmo/blob/v2.10.1/sncosmo/models.py
    The wrapped sncosmo version in sncosmo_models.py is faster and should be used
    when auto-differentiation is not needed.

    Parameterized values include:
      * c - The SALT2 c parameter.
      * dec - The object's declination in degrees. [from BasePhysicalModel]
      * distance - The object's luminosity distance in pc. [from BasePhysicalModel]
      * period - The period of the source, in days.
      * ra - The object's right ascension in degrees. [from BasePhysicalModel]
      * redshift - The object's redshift. [from BasePhysicalModel]
      * t0 - The t0 of the zero phase, date. [from BasePhysicalModel]
      * x0 - The SALT2 x0 parameter.
      * x1 - The SALT2 x1 parameter.

    References
    ----------
    * SALT2: Guy J., 2007 - https://doi.org/10.48550/arXiv.astro-ph/0701828
    * sncosmo - https://zenodo.org/records/14714968

    Attributes
    ----------
    _m0_model : BicubicInterpolator
        The interpolator for the m0 parameter.
    _m1_model : BicubicInterpolator
        The interpolator for the m1 parameter.
    _colorlaw : SALT2ColorLaw
        The data to apply the color law.

    Parameters
    ----------
    x0 : parameter
        The SALT2 x0 parameter.
    x1 : parameter
        The SALT2 x1 parameter.
    c : parameter
        The SALT2 c parameter.
    model_dir : str
        The path for the model file directory.
        Default: ""
    m0_filename : str
        The file name for the m0 model component.
        Default: "salt2_template_0.dat"
    m1_filename : str
        The file name for the m1 model component.
        Default: "salt2_template_1.dat"
    cl_filename : str
        The file name of the color law correction coefficients.
        Default: "salt2_color_correction.dat",
    **kwargs : dict, optional
        Any additional keyword arguments.
    """

    # A class variable for the units so we are not computing them each time.
    _FLAM_UNIT = u.erg / u.second / u.cm**2 / u.AA

    def __init__(
        self,
        x0=None,
        x1=None,
        c=None,
        model_dir="",
        m0_filename="salt2_template_0.dat",
        m1_filename="salt2_template_1.dat",
        cl_filename="salt2_color_correction.dat",
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Add the model specific parameters.
        self.add_parameter("x0", x0, description="The SALT2 x0 parameter.", **kwargs)
        self.add_parameter("x1", x1, description="The SALT2 x1 parameter.", **kwargs)
        self.add_parameter("c", c, description="The SALT2 c parameter.", **kwargs)

        # Load the data files.
        from lightcurvelynx.utils.bicubic_interp import BicubicInterpolator

        model_path = Path(model_dir)
        self._m0_model = BicubicInterpolator.from_grid_file(
            model_path / m0_filename,
            scale_factor=1e-12,
        )
        self._m1_model = BicubicInterpolator.from_grid_file(
            model_path / m1_filename,
            scale_factor=1e-12,
        )

        # Use the default color correction values.
        self._colorlaw = SALT2ColorLaw.from_file(model_path / cl_filename)

    def minphase(self, **kwargs):
        """Get the minimum supported rest-frame phase of the model in days.

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
        return -20.0

    def maxphase(self, **kwargs):
        """Get the minimum supported rest-frame phase of the model in days.

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
        return 50.0

    def compute_sed(self, times, wavelengths, graph_state, **kwargs):
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
        phase = times - params["t0"]

        m0_vals = self._m0_model(phase, wavelengths)
        m1_vals = self._m1_model(phase, wavelengths)

        flux_density = (
            params["x0"]
            * (m0_vals + params["x1"] * m1_vals)
            * 10.0 ** (-0.4 * self._colorlaw.apply(wavelengths) * params["c"])
        )

        # Convert to the correct units.
        flux_density = flam_to_fnu(
            flux_density,
            wavelengths,
            wave_unit=u.AA,
            flam_unit=self._FLAM_UNIT,
            fnu_unit=u.nJy,
        )
        return flux_density
