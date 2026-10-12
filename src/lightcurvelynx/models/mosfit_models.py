"""A wrapper for the semi-analytic transient models defined in MOSFiT.

https://github.com/guillochon/MOSFiT
https://mosfit.readthedocs.io/en/latest/

MOSFiT's own pipeline takes a model all the way to observed photometry: it redshifts
the SED, applies line-of-sight extinction, integrates through bandpasses and compares
against data. LightCurveLynx wants the opposite, so this wrapper builds on MOSFiT's
``mosfit.lynx.LynxSource``, which reports a model as a rest frame SED in nJy at 10 pc
with redshift, time dilation and extinction all pinned off. LightCurveLynx therefore
owns the redshift, the distance, the dust and the cadence, and MOSFiT supplies only
the intrinsic SED.

Requires MOSFiT >= 2.1, which is the first release to provide ``mosfit.lynx``.
"""

import numpy as np
from citation_compass import CiteClass
from scipy.interpolate import RegularGridInterpolator

from lightcurvelynx.models.physical_model import SEDModel
from lightcurvelynx.utils.extrapolate import ZeroPadding

# The absolute flux convention used by MOSFiT's rest frame SED output, in pc.
MOSFIT_REFERENCE_DISTANCE_PC = 10.0


class MOSFiTWrapperModel(SEDModel, CiteClass):
    """A wrapper for the transient models defined in MOSFiT.

    The user provides the name of a MOSFiT model (such as ``"slsn"``) and setters for
    whichever of that model's free parameters they want to vary. Any free parameter that
    is not given a setter takes the midpoint of its MOSFiT prior, so a simulation can vary
    a handful of physical parameters without having to name the rest.

    Parameterized values include:

    * dec - The object's declination in degrees. [from BasePhysicalModel]
    * distance - The object's luminosity distance in pc. [from BasePhysicalModel]
    * ra - The object's right ascension in degrees. [from BasePhysicalModel]
    * redshift - The object's redshift. [from BasePhysicalModel]
    * t0 - The t0 of the zero phase, date. [from BasePhysicalModel]

    Additional parameterized values are used for the specific MOSFiT model, such as
    ``mejecta``, ``vejecta`` and ``Pspin`` for ``"slsn"``. The names are MOSFiT's own and
    are case sensitive; an unrecognized name raises an error listing the model's free
    parameters.

    Unlike ``RedbackWrapperModel``, the redshift is *not* applied by the wrapped package.
    MOSFiT returns a rest frame SED at 10 pc, so LightCurveLynx applies both the redshift
    and the distance dimming, and dust can be added with the usual ``EffectModel`` objects.

    References
    ----------
    * MOSFiT - https://ui.adsabs.harvard.edu/abs/2018ApJS..236....6G/abstract
    * Individual models require their own citations. See the model references in the
      MOSFiT documentation.

    Attributes
    ----------
    model_name : str
        The name of the MOSFiT model being wrapped.
    source_param_names : list of str
        The names of the MOSFiT parameters that this node sets.
    phases : numpy.ndarray
        The phase grid (in days since explosion) on which MOSFiT is evaluated.
    wavelengths : numpy.ndarray
        The rest frame wavelength grid (in angstroms) on which MOSFiT is evaluated.

    Parameters
    ----------
    model_name : str
        The name of the MOSFiT model to wrap, such as ``"slsn"`` or ``"magnetar"``.
    parameters : dict, optional
        A dictionary of parameter setters, keyed by MOSFiT parameter name, to pass
        through to the MOSFiT model. Values may be constants or other nodes.
    phases : array_like, optional
        The phase grid (in days since explosion) on which to evaluate MOSFiT. MOSFiT
        rebuilds its entire model stack when the grid changes, so the wrapper evaluates
        on this fixed grid and interpolates onto the queried times. Pass a grid that
        covers the phases you intend to simulate.
        Default: 100 points spanning [0, 200] days.
    wavelengths : array_like, optional
        The rest frame wavelength grid (in angstroms) on which to evaluate MOSFiT.
        Default: 100 points spanning [1000, 25000] angstroms, which covers the Rubin
        u through y range with room on either side for the caller's redshift.
    time_extrapolation : FluxExtrapolationModel or tuple, optional
        The extrapolation method to use for times outside the phase grid.
        If nothing is provided, then the code adds zero padding.
    source : object, optional
        An already-constructed ``mosfit.lynx.LynxSource`` (or any object providing a
        ``compute_sed(times=..., wavelengths=..., parameters=...)`` interface) to use
        instead of building one. The source must honor the requested phase and wavelength
        grids and return an array with shape ``(len(phases), len(wavelengths))``.
        Mostly useful for testing and for sharing a single expensive MOSFiT model
        between several LightCurveLynx nodes.
    **kwargs : dict, optional
        Any additional keyword arguments, including those passed through to
        ``LynxSource`` such as ``parameter_path``.

    Note
    ----
    Building the underlying MOSFiT model is expensive, so it is deferred until the first
    call to ``compute_sed`` and then reused. Because MOSFiT would rebuild that model for
    every new grid, this wrapper never re-grids: it evaluates on ``phases`` and
    ``wavelengths`` and linearly interpolates the result, returning zero outside them.
    """

    # Keyword arguments that are forwarded to LynxSource rather than to the superclass.
    _SOURCE_KWARGS = ("parameter_path", "quiet")

    def __init__(
        self,
        model_name,
        *,
        parameters=None,
        phases=None,
        wavelengths=None,
        time_extrapolation=None,
        source=None,
        **kwargs,
    ):
        # Check that the parameters passed in the dictionary and keyword arguments do not
        # overlap, so we only have one source of truth. This is needed for parameters like
        # `redshift` that overlap core parameters.
        if parameters is None:
            parameters = {}
        for key in parameters:
            if key in kwargs:
                raise ValueError(
                    f"Parameter '{key}' specified in both the parameters dictionary "
                    "and as a parameter itself. Please include it only in the dictionary."
                )

        # Pull out the arguments that belong to the MOSFiT source instead of to this node.
        self._source_kwargs = {key: kwargs.pop(key) for key in self._SOURCE_KWARGS if key in kwargs}

        # If no time extrapolation method is provided, we default to zero padding.
        if time_extrapolation is None:
            time_extrapolation = ZeroPadding()
        super().__init__(time_extrapolation=time_extrapolation, **kwargs)

        self.model_name = model_name

        # Use the parameter dictionary to create settable parameters for the model. Some of
        # these might have already been added by the superclass's constructor, so we just
        # change those instead of adding them a second time.
        self.source_param_names = []
        for key, value in parameters.items():
            if key in self.setters:
                self.set_parameter(key, value)
            else:
                self.add_parameter(key, value, description="Parameter for MOSFiT model.")
            self.source_param_names.append(key)

        # MOSFiT reports the SED at 10 pc, so we need a distance to scale it to.
        if not self.has_valid_param("distance"):
            raise ValueError(
                "MOSFiTWrapperModel requires a valid 'distance' parameter representing luminosity "
                "distance in pc. This can be specified as 'distance' directly or derived by a "
                "combination of the 'redshift' and 'cosmology' parameters."
            )

        # Save the grids on which MOSFiT will be evaluated.
        self.phases = np.linspace(0.0, 200.0, 100) if phases is None else np.asarray(phases, dtype=float)
        self.wavelengths = (
            np.linspace(1000.0, 25000.0, 100) if wavelengths is None else np.asarray(wavelengths, dtype=float)
        )
        for name, grid in (("phases", self.phases), ("wavelengths", self.wavelengths)):
            if grid.ndim != 1 or len(grid) < 2:
                raise ValueError(f"MOSFiTWrapperModel's {name} must be a 1-d grid with at least 2 points.")
            if np.any(np.diff(grid) <= 0.0):
                raise ValueError(f"MOSFiTWrapperModel's {name} must be strictly increasing.")

        # The underlying MOSFiT source, built lazily on first use.
        self._source = source

        # A single entry cache of the interpolator for the last SED computed on the grid,
        # so that the repeated compute_sed() calls made during extrapolation neither re-run
        # MOSFiT nor rebuild the interpolator.
        self._cached_params = None
        self._cached_interpolator = None

    @property
    def param_names(self):
        """Return a list of the model's parameter names."""
        return self.source_param_names

    def minwave(self, **kwargs):
        """Get the minimum wavelength of the model.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments, not used in this method.

        Returns
        -------
        minwave : float
            The minimum wavelength of the model (in angstroms).
        """
        return float(self.wavelengths[0])

    def maxwave(self, **kwargs):
        """Get the maximum wavelength of the model.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments, not used in this method.

        Returns
        -------
        maxwave : float
            The maximum wavelength of the model (in angstroms).
        """
        return float(self.wavelengths[-1])

    def minphase(self, **kwargs):
        """Get the minimum supported phase of the model in days.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments, not used in this method.

        Returns
        -------
        minphase : float
            The minimum phase of the model (in days relative to t0).
        """
        return float(self.phases[0])

    def maxphase(self, **kwargs):
        """Get the maximum supported phase of the model in days.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments, not used in this method.

        Returns
        -------
        maxphase : float
            The maximum phase of the model (in days relative to t0).
        """
        return float(self.phases[-1])

    @property
    def source(self):
        """The underlying MOSFiT source object, building it on first access."""
        if self._source is None:
            try:
                from mosfit.lynx import LynxSource
            except ImportError as err:
                raise ImportError(
                    "The mosfit package is not installed by default. To use the MOSFiTWrapperModel, "
                    "please install MOSFiT >= 2.1, which is the first version to provide the "
                    "`mosfit.lynx` module. For example, you can install it with `pip install mosfit`."
                ) from err

            self._source = LynxSource(
                model=self.model_name,
                phases=self.phases,
                wavelengths=self.wavelengths,
                **self._source_kwargs,
            )
        return self._source

    def _grid_interpolator(self, graph_state):
        """Evaluate MOSFiT on the internal grid and return an interpolator over the result.

        Parameters
        ----------
        graph_state : GraphState
            An object mapping graph parameters to their values.

        Returns
        -------
        interpolator : scipy.interpolate.RegularGridInterpolator
            An interpolator over the len(phases) x len(wavelengths) matrix of rest frame
            SED values (in nJy) at the reference distance of 10 pc. Points outside the
            grid are zero rather than extrapolated; the model's bounds are reported by
            minphase(), etc., so the caller can set up an extrapolation model if they
            want something else there.
        """
        params = self.get_local_params(graph_state)
        fn_args = {name: params[name] for name in self.source_param_names}

        # Reuse the last result if the parameters have not changed. compute_sed() is called
        # more than once per sample when extrapolation is in play, and a MOSFiT evaluation
        # is far more expensive than the comparison.
        cache_key = tuple(float(fn_args[name]) for name in self.source_param_names)
        if self._cached_params is not None and cache_key == self._cached_params:
            return self._cached_interpolator

        # Ask for the grids explicitly rather than relying on the ones the source was
        # built with, so that the requested grids and the shape check below are always
        # about the same thing. LynxSource rebuilds itself if it is handed a grid it was
        # not set up on, which also makes an externally supplied `source` safe to use.
        try:
            grid_sed = np.asarray(
                self.source.compute_sed(
                    times=self.phases,
                    wavelengths=self.wavelengths,
                    parameters=fn_args,
                ),
                dtype=float,
            )
        except Exception as err:
            raise RuntimeError(
                f"Error evaluating the MOSFiT model '{self.model_name}'. This is often due to a "
                "parameter value outside its MOSFiT prior range, or a parameter name that is not "
                "free in this model. Original error message: " + str(err)
            ) from err

        expected = (len(self.phases), len(self.wavelengths))
        if grid_sed.shape != expected:
            raise ValueError(
                f"MOSFiT model '{self.model_name}' returned an SED of shape {grid_sed.shape}, "
                f"but the wrapper is set up on a grid of shape {expected}."
            )

        self._cached_interpolator = RegularGridInterpolator(
            (self.phases, self.wavelengths),
            grid_sed,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        self._cached_params = cache_key
        return self._cached_interpolator

    def compute_sed(self, times, wavelengths, graph_state, **kwargs):
        """Draw effect-free observations for this object.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of rest frame timestamps (MJD).
        wavelengths : numpy.ndarray, optional
            A length N array of rest frame wavelengths (in angstroms).
        graph_state : GraphState
            An object mapping graph parameters to their values.
        **kwargs : dict, optional
           Any additional keyword arguments.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of rest frame SED values (in nJy).
        """
        interpolator = self._grid_interpolator(graph_state)

        # MOSFiT works in phases (days since explosion), so shift the query times by t0.
        t0 = self.get_param(graph_state, "t0")
        if t0 is None:
            t0 = 0.0
        phases = np.asarray(times, dtype=float) - t0
        wavelengths = np.asarray(wavelengths, dtype=float)

        # Interpolate the grid onto the queried points.
        query = np.stack(
            np.meshgrid(phases, wavelengths, indexing="ij"),
            axis=-1,
        )
        flux_density = interpolator(query)

        # MOSFiT reports the SED at 10 pc, so rescale it to the object's actual distance.
        distance_pc = self.get_param(graph_state, "distance")
        if distance_pc is None or distance_pc <= 0:
            raise ValueError(f"Received invalid luminosity distance (pc) in MOSFiT model {distance_pc}.")
        return flux_density * (MOSFIT_REFERENCE_DISTANCE_PC / distance_pc) ** 2
