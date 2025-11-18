from abc import ABC, abstractmethod

from lightcurvelynx.models.physical_model import SEDModel


class PeriodicModel(SEDModel, ABC):
    """The base model for periodic sources.

    Parameterized values include:
      * dec - The object's declination in degrees. [from BasePhysicalModel]
      * distance - The object's luminosity distance in pc. [from BasePhysicalModel]
      * period - The period of the source, in days.
      * ra - The object's right ascension in degrees. [from BasePhysicalModel]
      * redshift - The object's redshift. [from BasePhysicalModel]
      * t0 - The t0 of the zero phase, date. [from BasePhysicalModel]

    Parameters
    ----------
    period : float
        The period of the source, in days.
    **kwargs : dict, optional
        Any additional keyword arguments.
    """

    def __init__(self, period, **kwargs):
        super().__init__(**kwargs)

        # t0 is added in the BasePhysicalModel constructor.
        self.add_parameter("period", period, description="The period of the source, in days.", **kwargs)

    @abstractmethod
    def _evaluate_phases(self, phases, wavelengths, graph_state, **kwargs):
        """Draw effect-free observations for this object, as a function of phase.

        Parameters
        ----------
        phases : numpy.ndarray
            A length T array of phases, in the range [0, 1].
        wavelengths : numpy.ndarray, optional
            A length N array of wavelengths.
        graph_state : GraphState
            An object mapping graph parameters to their values.
        **kwargs : dict, optional
              Any additional keyword arguments.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of SED values.
        """
        raise NotImplementedError()  # pragma: no cover

    def compute_sed(self, times, wavelengths, graph_state, **kwargs):
        """Draw effect-free observations for this object.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of rest frame timestamps.
        wavelengths : numpy.ndarray, optional
            A length N array of wavelengths.
        graph_state : GraphState
            An object mapping graph parameters to their values.
        **kwargs : dict, optional
           Any additional keyword arguments.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of SED values.
        """
        params = self.get_local_params(graph_state)
        period = params["period"]
        phases = (times - params["t0"]) % period / period
        flux_density = self._evaluate_phases(phases, wavelengths, graph_state, **kwargs)

        return flux_density
