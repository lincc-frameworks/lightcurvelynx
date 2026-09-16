"""The MAF API includes helper functions for calling LightCurveLynx from within
Rubin's metric analysis framework. By default, it is assumed that the MAF query
uses the Rubin passbands, noise models, and OpSim (although all of these
can be overridden).
"""

import numpy as np

from lightcurvelynx.astro_utils.coordinate_utils import validate_ra_dec_degrees
from lightcurvelynx.astro_utils.passbands import PassbandGroup
from lightcurvelynx.noise_models.base_noise_models import PoissonFluxNoiseModel
from lightcurvelynx.obstable.opsim import OpSim
from lightcurvelynx.simulate import simulate_lightcurves
from lightcurvelynx.survey_info import SurveyInfo

# "Global" caches that are only initialized for the MAF query path.
_CACHED_MAF_QUERY_PASSBANDS = None
_CACHED_MAF_NOISE_MODEL = None


class MAFQueryTable(OpSim):
    """A subset of the OpSim style ObsTable that is optimized for a predefined
    query - a single spatial location with multiple times.

    Parameters
    ----------
    table : dict or pandas.core.frame.DataFrame
        The table with all the minimal OpSim information to compute the noise model.
    **kwargs : dict
        Additional keyword arguments to pass to the OpSim constructor. This includes
        overrides for survey parameters, the saturation parameters, and a custom column map.
    """

    def __init__(self, table, **kwargs):
        # We do not support spatial search in MAFQueryTable.
        if "detector_footprint" in kwargs:
            raise ValueError("MAFQueryTable doesn't support detector_footprint.")  # pragma: no cover

        # Call the constructor.
        super().__init__(table, **kwargs)
        self._all_inds = np.arange(len(self))

    def _build_spatial_data(self):
        # The MAFQueryTable does not do spatial filtering.
        self._spatial_data = None

    def range_search(self, query_ra, query_dec, **kwargs):
        """Return all the indices in the table because a MAF query is
        meant to represent a single batch of observations that should
        always be simulated.

        Parameters
        ----------
        query_ra : float or numpy.ndarray
            The query right ascension (in degrees).
        query_dec : float or numpy.ndarray
            The query declination (in degrees).
        **kwargs : dict
            Additional keywords for API compatibility. These are not used in the
            range search.

        Returns
        -------
        inds : numpy.ndarray or list[numpy.ndarray]
            Depending on the input, this is either an array of indices for a single query
            point or a list of arrays (of indices) for an array of query points.
        """
        is_scalar = np.isscalar(query_ra) and np.isscalar(query_dec)
        query_ra, query_dec = validate_ra_dec_degrees(query_ra, query_dec)

        if is_scalar:
            return self._all_inds
        else:
            return [self._all_inds] * len(query_ra)


def execute_maf_query(
    model,
    table,
    *,
    graph_state=None,
    passbands=None,
    noise_model=None,
    rng_info=None,
):
    """Run a single query from Rubin's metric analysis framework.

    Parameters
    ----------
    model : BasePhysicalModel
        The model to draw from. This may have its own parameters which will be randomly
        sampled with each draw. This object's parameters (e.g., ra, dec) will be saved
        to the result columns.
    table : MAFQueryTable, dict, or pandas.core.frame.DataFrame
        The table with all the minimal OpSim information to compute the noise model.
    graph_state : GraphState, optional
        The predefined graph state to use to replay a previous simulation. If None,
        a new graph state is sampled.  Default: None
    passbands : PassbandGroup, optional
        A class that contains the information about the bandpass of the instrument for each filter,
        including the wavelength range and the transmission curve. If not provided, defaults
        to Rubin's passbands. Default: None
    noise_model : NoiseModel, optional
        A computation class for calculating the noise from characteristics of the survey.
        If not provided, defaults to a PoissonFluxNoiseModel. Default: None
    rng_info : dict, optional
        Information about the random number generator to use for sampling.
         Default: None

    Returns
    -------
    lightcurve : Pandas DataFrame
        Returns the Pandas DataFrame containing the light curve for this object.
    params : dict
        The parameters of the model used for this simulation.
    """
    # We use the global cached versions of these bassbands and noise model.
    global _CACHED_MAF_QUERY_PASSBANDS, _CACHED_MAF_NOISE_MODEL

    # If the model has not been sampled, do that first.
    if graph_state is None:
        graph_state = model.sample_parameters(num_samples=1, rng_info=rng_info)

    # If the noise model is not explicitly given, use the (cached) default.
    if noise_model is None:
        global _CACHED_MAF_NOISE_MODEL
        if _CACHED_MAF_NOISE_MODEL is None:
            _CACHED_MAF_NOISE_MODEL = PoissonFluxNoiseModel()
        noise_model = _CACHED_MAF_NOISE_MODEL

    # If the passbands are not not explicitly given, use the (cached) default.
    if passbands is None:
        if _CACHED_MAF_QUERY_PASSBANDS is None:
            _CACHED_MAF_QUERY_PASSBANDS = PassbandGroup.from_preset("LSST")
        passbands = _CACHED_MAF_QUERY_PASSBANDS

    # Construct the MAF Query ObsTable and the SurveyInfo for the simulation.
    if not isinstance(table, MAFQueryTable):
        table = MAFQueryTable(table)
    if len(table) == 0:
        raise ValueError("The MAF query table is empty.")  # pragma: no cover

    survey_info = SurveyInfo(
        table,
        passbands=passbands,
        noise_model=noise_model,
        survey_name="MAF Query",
        validate=False,
    )

    # Run the simulation.
    results = simulate_lightcurves(
        model,
        num_samples=1,
        survey_info=survey_info,
        apply_saturation=True,
        rng=rng_info,
        progress_bar=False,
        graph_state=graph_state,
    )
    return results["lightcurve"].iloc[0], results["params"].iloc[0]
