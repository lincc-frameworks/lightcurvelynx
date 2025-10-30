"""A collection of toy models that are primarily used for testing."""

import numpy as np

from lightcurvelynx.models.physical_model import SEDModel


class ConstantSEDModel(SEDModel):
    """A model with a constant SED over both wavelength and time.

    Parameterized values include:
      * brightness - The inherent brightness
      * dec - The object's declination in degrees. [from BasePhysicalModel]
      * distance - The object's luminosity distance in pc. [from BasePhysicalModel]
      * ra - The object's right ascension in degrees. [from BasePhysicalModel]
      * redshift - The object's redshift. [from BasePhysicalModel]
      * t0 - No effect for static model. [from BasePhysicalModel]

    Parameters
    ----------
    brightness : float
        The inherent brightness
    **kwargs : dict, optional
        Any additional keyword arguments.
    """

    def __init__(self, brightness, **kwargs):
        super().__init__(**kwargs)
        self.add_parameter(
            "brightness",
            brightness,
            description="The inherent brightness of the object.",
            **kwargs,
        )

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
        return np.full((len(times), len(wavelengths)), params["brightness"])


class StepModel(ConstantSEDModel):
    """A static model that is on for a fixed amount of time.

    Parameterized values include:
      * brightness - The inherent brightness
      * dec - The object's declination in degrees. [from BasePhysicalModel]
      * distance - The object's luminosity distance in pc. [from BasePhysicalModel]
      * ra - The object's right ascension in degrees. [from BasePhysicalModel]
      * redshift - The object's redshift. [from BasePhysicalModel]
      * t0 - The time the step function starts, in MJD.
      * t1- The time the step function ends, in MJD.

    Parameters
    ----------
    brightness : float
        The inherent brightness
    t0 : float
        The time the step function starts, in MJD.
    t1 : float
        The time the step function ends, in MJD.
    **kwargs : dict, optional
        Any additional keyword arguments.
    """

    def __init__(self, brightness, t1, **kwargs):
        super().__init__(brightness, **kwargs)

        # t0 is added in the PhysicalModel constructor.
        self.add_parameter("t1", t1, description="The time the step function ends, in MJD.", **kwargs)

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
        flux_density = np.zeros((len(times), len(wavelengths)))
        params = self.get_local_params(graph_state)

        time_mask = (times >= params["t0"]) & (times <= params["t1"])
        flux_density[time_mask] = params["brightness"]
        return flux_density


class SinWaveModel(SEDModel):
    """A model that emits a sine wave.

    flux = brightness + amplitude * sin(2 * pi * frequency * (time - t0))

    Parameterized values include:
      * brightness - The inherent brightness
      * frequency - The frequence of the sine wave.
      * dec - The object's declination in degrees. [from BasePhysicalModel]
      * distance - The object's luminosity distance in pc. [from BasePhysicalModel]
      * ra - The object's right ascension in degrees. [from BasePhysicalModel]
      * redshift - The object's redshift. [from BasePhysicalModel]
      * t0 - The start of the sine wave's period. [from BasePhysicalModel]

    Parameters
    ----------
    brightness : float
        The inherent brightness.
        Default: 0.0
    amplitude : float
        The amplitude of the sine wave.
        Default: 0.0
    frequency : float
        The frequency of the sine wave.
        Default: 1.0
    **kwargs : dict, optional
        Any additional keyword arguments.
    """

    def __init__(self, *, brightness=0.0, amplitude=0.0, frequency=1.0, **kwargs):
        super().__init__(**kwargs)
        self.add_parameter(
            "brightness",
            brightness,
            description="The inherent brightness of the object.",
            **kwargs,
        )
        self.add_parameter("amplitude", amplitude, description="The amplitude of the sine wave.", **kwargs)
        self.add_parameter("frequency", frequency, description="The frequency of the sine wave.", **kwargs)

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
        phases = 2.0 * np.pi * params["frequency"] * (times - params["t0"])
        single_wave = params["brightness"] + params["amplitude"] * np.sin(phases)
        return np.tile(single_wave[:, np.newaxis], (1, len(wavelengths)))


class LinearWavelengthModel(SEDModel):
    """A model that emits flux as a linear function of wavelength
    (that is constant over time): f(t, w) = scale * w + base.

    Includes optional minimum and maximum wavelength bounds to test
    extrapolation.

    Parameterized values include:
      * linear_base - The base brightness in nJy.
      * linear_scale - The slope of the linear function in nJy/Angstrom.
      * dec - The object's declination in degrees. [from BasePhysicalModel]
      * distance - The object's luminosity distance in pc. [from BasePhysicalModel]
      * ra - The object's right ascension in degrees. [from BasePhysicalModel]
      * redshift - The object's redshift. [from BasePhysicalModel]
      * t0 - No effect for static model. [from BasePhysicalModel]

    Attributes
    ----------
    min_wave : float or None
        The minimum wavelength of the model (in angstroms). Or None if there
        is no minimum wavelength.
    max_wave : float or None
        The maximum wavelength of the model (in angstroms). Or None if there
        is no maximum wavelength.

    Parameters
    ----------
    linear_base : parameter
        The base brightness in nJy.
    linear_scale : parameter
        The slope of the linear function in nJy/Angstrom.
    min_wave : float or None
        The minimum wavelength of the model (in angstroms). Or None if there
        is no minimum wavelength.
    max_wave : float or None
        The maximum wavelength of the model (in angstroms). Or None if there
        is no maximum wavelength.
    **kwargs : dict, optional
        Any additional keyword arguments.
    """

    def __init__(self, linear_base, linear_scale, min_wave=None, max_wave=None, **kwargs):
        super().__init__(**kwargs)
        self.add_parameter("linear_base", linear_base, **kwargs)
        self.add_parameter("linear_scale", linear_scale, **kwargs)
        self.min_wave = min_wave
        self.max_wave = max_wave

    def minwave(self, graph_state=None):
        """Get the minimum wavelength of the model.

        Parameters
        ----------
        graph_state : GraphState, optional
            An object mapping graph parameters to their values. Not used
            for this model.

        Returns
        -------
        minwave : float or None
            The minimum wavelength of the model (in angstroms) or None
            if the model does not have a defined minimum wavelength.
        """
        return self.min_wave

    def maxwave(self, graph_state=None):
        """Get the maximum wavelength of the model.

        Parameters
        ----------
        graph_state : GraphState, optional
            An object mapping graph parameters to their values. Not used
            for this model.

        Returns
        -------
        maxwave : float or None
            The maximum wavelength of the model (in angstroms) or None
            if the model does not have a defined maximum wavelength.
        """
        return self.max_wave

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
        single_wave = params["linear_base"] + params["linear_scale"] * wavelengths
        return np.tile(single_wave[np.newaxis, :], (len(times), 1))
