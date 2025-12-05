"""The core functions for running the LightCurveLynx simulation."""

import concurrent.futures
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from nested_pandas import NestedFrame
from tqdm import tqdm

from lightcurvelynx.astro_utils.noise_model import apply_noise
from lightcurvelynx.models.physical_model import BandfluxModel
from lightcurvelynx.utils.post_process_results import concat_results

logger = logging.getLogger(__name__)


class SimulationInfo:
    """A class to hold all the information (data and configuration) for a simulation
    run. This includes the model, the surveys, the passbands, etc. This object
    is used so we have have a single object to pass around to multiprocessing functions.

    Attributes
    ----------
    model : BasePhysicalModel
        The model to draw from. This may have its own parameters which
        will be randomly sampled with each draw.
    num_samples : int
        The number of samples.
    obstable : ObsTable or List of ObsTable
        The ObsTable(s) from which to extract information for the samples.
    passbands : PassbandGroup or List of PassbandGroup
        The passbands to use for generating the bandfluxes.
    time_window_offset : tuple(float, float), optional
        A tuple specifying the observer-frame time window offset (start, end) relative
        to t0 in days. This is used to filter the observations to only those within the
        specified observer-frame time window (t0 + start, t0 + end). If None or the model
        does not have a t0 specified, no time window is applied.
    obstable_save_cols : list of str, optional
        A list of ObsTable columns to be saved as part of the results. This is used
        to save context information about how the light curves were generated. If the column
        is missing from one of the ObsTables, a null value such as None or NaN is used.
        If None, no additional columns are saved.
    param_cols : list of str, optional
        A list of the model's parameter columns to be saved as separate columns in
        the results (instead of just the full dictionary of parameters). These
        must be specified as strings in the node_name.param_name format.
        If None, no additional columns are saved.
    rng : numpy.random._generator.Generator, optional
        A given numpy random number generator to use for this computation. If not
        provided, the function uses the node's random number generator.
    sample_offset : int
        An offset to apply to the sample indices. This is used when splitting
        the simulation into multiple batches for multiprocessing.
    output_file_path : str or Path, optional
        The file path and name of where to save the results. If provided the results
        are saved to this file instead of being returned directly.
    kwargs : dict
        Additional keyword arguments to pass to the simulation function.
    """

    def __init__(
        self,
        model,
        num_samples,
        obstable,
        passbands,
        *,
        obstable_save_cols=None,
        param_cols=None,
        time_window_offset=None,
        sample_offset=0,
        rng=None,
        output_file_path=None,
        **kwargs,
    ):
        self.model = model
        self.num_samples = num_samples
        self.obstable = obstable
        self.passbands = passbands
        self.time_window_offset = time_window_offset
        self.obstable_save_cols = obstable_save_cols
        self.param_cols = param_cols
        self.sample_offset = sample_offset
        self.rng = rng
        self.kwargs = kwargs
        self.output_file_path = None

        if self.num_samples <= 0:
            raise ValueError("Number of samples must be a positive integer.")

        if output_file_path is not None:
            self.output_file_path = Path(output_file_path)
            if not self.output_file_path.parent.exists():
                raise ValueError(f"Output file directory {self.output_file_path.parent} does not exist.")

    def split(self, num_batches=None, batch_size=None):
        """Split the simulation info into multiple batches for parallel processing.

        Parameters
        ----------
        num_batches : int, optional
            The number of batches to split the simulation into. If None, batch_size
            must be provided.
        batch_size : int, optional
            The size of each batch. If None, num_batches must be provided.

        Returns
        -------
        batches : list of SimulationInfo
            A list of SimulationInfo objects, one for each batch.
        """
        if num_batches is None and batch_size is None:
            raise ValueError("Either num_batches or batch_size must be provided.")
        if num_batches is not None and batch_size is not None:
            raise ValueError("Either num_batches or batch_size must be provided, but not both.")

        if num_batches is not None:
            if num_batches <= 0:
                raise ValueError("num_batches must be a positive integer.")
            if num_batches > self.num_samples:
                num_batches = self.num_samples
            batch_size = np.ceil(self.num_samples / num_batches).astype(int)
        else:
            if batch_size <= 0:
                raise ValueError("batch_size must be a positive integer.")
            num_batches = np.ceil(self.num_samples / batch_size).astype(int)

        # Get overall RNG for this simulation. If we do not have one, use the default.
        global_rng = self.rng if self.rng is not None else np.random.default_rng()

        batches = []
        end_idx = 0
        for batch_idx in range(num_batches):
            # Compute the bounds of the batch.
            start_idx = end_idx
            end_idx = min(start_idx + batch_size, self.num_samples)
            batch_num_samples = end_idx - start_idx
            if batch_num_samples <= 0:
                break

            # Make sure we create a unique RNG for each batch. Even if we do not have a global
            # RNG, we do not want to use the ones created with the nodes because they will be
            # correlated across batches.
            seed = global_rng.integers(0, 2**32 - 1)
            batch_rng = np.random.default_rng(seed)

            # If we are saving to a file, modify the output file path for this batch.
            if self.output_file_path is not None:
                batch_output_file_path = (
                    self.output_file_path.parent
                    / f"{self.output_file_path.stem}_part{batch_idx}{self.output_file_path.suffix}"
                )
            else:
                batch_output_file_path = None

            # Create a subset of the batch. Most information is the same (references to the
            # same objects), except for the number of samples and the RNG.
            batch_info = SimulationInfo(
                model=self.model,
                num_samples=batch_num_samples,
                obstable=self.obstable,
                passbands=self.passbands,
                obstable_save_cols=self.obstable_save_cols,
                param_cols=self.param_cols,
                time_window_offset=self.time_window_offset,
                sample_offset=self.sample_offset + start_idx,
                rng=batch_rng,
                output_file_path=batch_output_file_path,
                **self.kwargs,
            )
            batches.append(batch_info)
        return batches


def get_time_windows(t0, time_window_offset):
    """Get the time windows for each sample state based on the time window offset.

    Parameters
    ----------
    t0 : float or np.ndarray, optional
        The reference time (t0) for the time windows.
    time_window_offset : tuple(float, float), optional
        A tuple specifying the observer-frame time window offset (start, end) relative
        to t0 in days. If None, no time window is applied.

    Returns
    -------
    start_times : np.ndarray or None
        The start times for each sample t0 + time_window_offset[0] in the observer frame.
        If a before time is given, this is always returned as an array (even if t0 is a scalar).
        None returned if there is no start time.
    end_times : np.ndarray or None
        The end times for each sample t0 + time_window_offset[1] in the observer frame.
        If an after time is given, this is always returned as an array (even if t0 is a scalar).
        None returned if there is no end time.
    """
    # If the model did not have a t0 or we do not have a time_window_offset,
    # we cannot apply a time window.
    if t0 is None or time_window_offset is None:
        return None, None
    if len(time_window_offset) != 2:
        raise ValueError("time_window_offset must be a tuple of (before, after) in days.")
    before, after = time_window_offset

    # If t0 is a scalar apply the offset directly.
    if np.isscalar(t0):
        t0 = np.array([t0])
    start_times = t0 + before if before is not None else None
    end_times = t0 + after if after is not None else None

    return start_times, end_times


def _simulate_lightcurves_batch(simulation_info):
    """Generate a number of simulations of the given model and information
    from one or more surveys.

    Parameters
    ----------
    simulation_info : SimulationInfo
        The information needed to perform simulate a single batch of results.

    Returns
    -------
    lightcurves : nested_pandas.NestedFrame or Path
        If no output_file_path is specified in simulation_info, a NestedFrame with a row
        for each object. Otherwise the NestedFrame is saved to a file and the function
        returns that file's path.
    """
    sample_offset = simulation_info.sample_offset
    logger.info(f"Starting batch at {sample_offset} with {simulation_info.num_samples} samples.")

    # Extract the parameters from the SimulationInfo that are used repeated
    # (so we have shorter names).
    model = simulation_info.model
    num_samples = simulation_info.num_samples
    obstable = simulation_info.obstable
    passbands = simulation_info.passbands
    obstable_save_cols = simulation_info.obstable_save_cols
    rng = simulation_info.rng

    # Sample the parameter space of this model. We do this once for all surveys, so the
    # object use the same parameters across all observations.
    if num_samples <= 0:
        raise ValueError("Invalid number of samples.")
    logger.info(f"Sampling {num_samples} parameter sets from the model.")
    sample_states = model.sample_parameters(
        num_samples=num_samples,
        rng_info=rng,
        sample_offset=sample_offset,
    )

    # If we are given information for a single survey, make it into a list.
    if not isinstance(obstable, list):
        obstable = [obstable]
    if not isinstance(passbands, list):
        passbands = [passbands]
    num_surveys = len(obstable)
    if num_surveys != len(passbands):
        raise ValueError("Number of surveys must match number of passbands.")

    # We do not currently support bandflux models with multiple surveys because
    # a bandflux model is defined relative to a single survey.
    if num_surveys > 1 and isinstance(model, BandfluxModel):
        raise ValueError(
            "Simulating a BandfluxModel with multiple surveys is currently not supported, "
            "because the bandflux model is defined relative to the filters of a single survey."
        )

    # Create a dictionary for the object level information, including any saved parameters.
    # Some of these are placeholders (e.g. nobs) until they can be filled in during the simulation.
    # These values are always pulled from the outer-most object (model), which most often
    # corresponds to the source (as opposed to a host).
    logger.info("Setting up result data structures.")
    ra = np.atleast_1d(model.get_param(sample_states, "ra"))
    dec = np.atleast_1d(model.get_param(sample_states, "dec"))
    results_dict = {
        "id": [i for i in range(num_samples)],
        "ra": ra.tolist(),
        "dec": dec.tolist(),
        "nobs": [0] * num_samples,
        "t0": np.atleast_1d(model.get_param(sample_states, "t0")).tolist(),
        "z": np.atleast_1d(model.get_param(sample_states, "redshift")).tolist(),
        "params": [state.to_dict() for state in sample_states],
    }
    if simulation_info.param_cols is not None:
        for col in simulation_info.param_cols:
            if col not in sample_states:
                raise KeyError(
                    f"Parameter column {col} not found in model parameters. "
                    f"Available parameters are: {sample_states.get_all_params_names()}."
                )
            results_dict[col.replace(".", "_")] = np.atleast_1d(sample_states[col]).tolist()

    # Set up the nested array for the per-observation data, including ObsTable information.
    nested_index = []
    nested_dict = {
        "mjd": [],
        "filter": [],
        "flux": [],
        "fluxerr": [],
        "flux_perfect": [],
        "survey_idx": [],  # The index of the survey
        "obs_idx": [],  # The index of the observation in the survey
        "is_saturated": [],
    }
    if obstable_save_cols is None:
        obstable_save_cols = []
    for col in obstable_save_cols:
        nested_dict[col] = []

    # Determine which of the of the simulated positions match the pointings from each ObsTable.
    logger.info("Performing range searches to find matching observations.")
    start_times, end_times = get_time_windows(
        model.get_param(sample_states, "t0"),
        simulation_info.time_window_offset,
    )
    all_obs_matches = [
        obstable[i].range_search(ra, dec, t_min=start_times, t_max=end_times) for i in range(num_surveys)
    ]

    # Get all times and all filters as numpy arrays so we can do easy subsets.
    all_times = [np.asarray(obstable[i]["time"].values, dtype=float) for i in range(num_surveys)]
    all_filters = [np.asarray(obstable[i]["filter"].values, dtype=str) for i in range(num_surveys)]

    # We loop over objects first, then surveys. This allows us to generate a single block
    # of data for the object over all surveys.
    logger.info("Simulating light curves for each object.")
    for idx, state in tqdm(enumerate(sample_states), total=num_samples, desc="Simulating", unit="obj"):
        total_num_obs = 0

        for survey_idx in range(num_surveys):
            # Find the indices and times where the current model is seen.
            obs_index = np.asarray(all_obs_matches[survey_idx][idx])
            if len(obs_index) == 0:
                obs_times = []
                obs_filters = []
            else:
                obs_times = all_times[survey_idx][obs_index]

                # Extract the filters for this observation.
                obs_filters = all_filters[survey_idx][obs_index]

            # Compute the bandfluxes and errors over just the given filters.
            bandfluxes_perfect = model.evaluate_bandfluxes(
                passbands[survey_idx],
                obs_times,
                obs_filters,
                state,
                rng_info=rng,
            )
            bandfluxes_error = obstable[survey_idx].bandflux_error_point_source(bandfluxes_perfect, obs_index)
            bandfluxes = apply_noise(bandfluxes_perfect, bandfluxes_error, rng=rng)

            # Apply saturation thresholds from the ObsTable.
            bandfluxes, bandfluxes_error, saturation_flags = obstable[survey_idx].compute_saturation(
                bandfluxes, bandfluxes_error, obs_index
            )

            # Append the per-observation data to the nested dictionary, including
            # any needed ObsTable columns.
            nobs = len(obs_times)
            nested_dict["mjd"].extend(list(obs_times))
            nested_dict["filter"].extend(list(obs_filters))
            nested_dict["flux_perfect"].extend(list(bandfluxes_perfect))
            nested_dict["flux"].extend(list(bandfluxes))
            nested_dict["fluxerr"].extend(list(bandfluxes_error))
            nested_dict["survey_idx"].extend([survey_idx] * nobs)
            nested_dict["is_saturated"].extend(list(saturation_flags))
            nested_dict["obs_idx"].extend(list(obs_index))
            for col in obstable_save_cols:
                col_data = (
                    list(obstable[survey_idx][col].values[obs_index])
                    if col in obstable[survey_idx]
                    else [None] * nobs
                )
                nested_dict[col].extend(col_data)

            total_num_obs += nobs
            nested_index.extend([idx] * nobs)

        # The number of observations is the total across all surveys.
        results_dict["nobs"][idx] = total_num_obs

    # Create the nested frame and either save it to a file or return it directly.
    logger.info("Compiling results.")
    results = NestedFrame(data=results_dict, index=[i for i in range(num_samples)])
    nested_frame = pd.DataFrame(data=nested_dict, index=nested_index)
    results = results.join_nested(nested_frame, "lightcurve")
    if simulation_info.output_file_path is not None:
        results.to_parquet(simulation_info.output_file_path)
        return simulation_info.output_file_path
    return results


def _simulate_lightcurves_parallel(simulation_info, executor, batch_size=1_000):
    """Generate a number of simulations of the given model and information
    from one or more surveys.

    Parameters
    ----------
    simulation_info : SimulationInfo
        The information needed to perform simulate a single batch of results.
    executor : concurrent.futures.Executor, optional
        The executor object to use for parallel processing.
    batch_size : int, optional
        The number of samples to process in each batch when using multiprocessing.
        Default is 1000.

    Returns
    -------
    lightcurves : list of nested_pandas.NestedFrame or Path
        If no output_file_path is specified in simulation_info, a NestedFrame with a row
        for each object. Otherwise the NestedFrame is saved to a file and the function
        returns that file's path.
    """
    # Perform the simulation in parallel batches and combine the results. Since different
    # frameworks may return either Future objects or direct results from the map function,
    # we check for both.
    batches = simulation_info.split(batch_size=batch_size)
    futures_or_results = executor.map(_simulate_lightcurves_batch, batches)
    result_list = []
    for res in futures_or_results:
        if hasattr(res, "result") and callable(res.result):  # A Future
            result_list.append(res.result())
        else:  # A direct result
            result_list.append(res)
    return result_list


def simulate_lightcurves(
    model,
    num_samples,
    obstable,
    passbands,
    *,
    obstable_save_cols=None,
    param_cols=None,
    time_window_offset=None,
    output_file_path=None,
    rng=None,
    executor=None,
    num_jobs=None,
    batch_size=100_000,
):
    """Generate a number of simulations of the given model and information
    from one or more surveys. The result data can either be returned directly
    (as a single nested data frame) or saved to file(s).

    The columns in the return NestedFrame can include a mix of default information
    from the source object (e.g., ra, dec, t0, redshift), a nested lightcurve table,
    a saved parameters data block (with all the model's parameters), and any additional
    user-specified values.

    Parameters
    ----------
    model : BasePhysicalModel
        The model to draw from. This may have its own parameters which will be randomly
        sampled with each draw. This object's parameters (e.g., ra, dec) will be saved
        to the result columns.
    num_samples : int
        The number of samples.
    obstable : ObsTable or List of ObsTable
        The ObsTable(s) from which to extract information for the samples.
    passbands : PassbandGroup or List of PassbandGroup
        The passbands to use for generating the bandfluxes.
    time_window_offset : tuple(float, float), optional
        A tuple specifying the observer-frame time window offset (start, end) relative
        to t0 in days. This is used to filter the observations to only those within the
        specified observer-frame time window (t0 + start, t0 + end). If None or the model
        does not have a t0 specified, no time window is applied.
    obstable_save_cols : list of str, optional
        A list of ObsTable columns to be saved as part of the results. This is used
        to save context information about how the light curves were generated. If the column
        is missing from one of the ObsTables, a null value such as None or NaN is used.
        If None, no additional columns are saved.
    param_cols : list of str, optional
        A list of the model's parameter columns to be saved as separate columns in
        the results (instead of just the full dictionary of parameters). These
        must be specified as strings in the node_name.param_name format.
        If None, no additional columns are saved.
    output_file_path : str or Path, optional
        The file path and name of where to save the results. If provided the results
        are saved to this file instead of being returned directly. If the simulation
        is run in parallel, multiple files are created with a _partN suffix.
    rng : numpy.random._generator.Generator, optional
        A given numpy random number generator to use for this computation. If not
        provided, the function uses the node's random number generator.
    executor : concurrent.futures.Executor, optional
        The executor object that to use for parallel processing. This can be any object that
        supports concurrent.futures's map() function, returning either Future objects or
        direct results. If None, the function runs in serial.
    num_jobs : int, optional
        If provided (and no executor is provided) creates a process pool (ProcessPoolExecutor)
        with the given number of workers to process the results.
    batch_size : int, optional
        The number of samples to process in each batch when using multiprocessing.
        Default is 100_000.

    Returns
    -------
    lightcurves : nested_pandas.NestedFrame or list of Path
        If output_file_path is None, a NestedFrame with a row for each object. Otherwise
        the file(s) are saved and the function returns a list of Paths to the saved files.
    """
    if output_file_path is not None:
        output_file_path = Path(output_file_path)
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Create the simulation info wrapper.
    simulation_info = SimulationInfo(
        model=model,
        num_samples=num_samples,
        obstable=obstable,
        passbands=passbands,
        rng=rng,
        time_window_offset=time_window_offset,
        obstable_save_cols=obstable_save_cols,
        param_cols=param_cols,
        output_file_path=output_file_path,
    )

    # If we do not have any parallelization information, perform in serial.
    if executor is None and num_jobs is None:
        return _simulate_lightcurves_batch(simulation_info)

    # If we have an executor, use that.
    if executor is not None:
        result_list = _simulate_lightcurves_parallel(simulation_info, executor, batch_size=batch_size)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_jobs) as executor:
            result_list = _simulate_lightcurves_parallel(simulation_info, executor, batch_size=batch_size)

    # If we returned the results directly, concatenate them into a single NestedFrame.
    # If we saved them to file, return the list of file paths.
    if output_file_path is None:
        results = concat_results(result_list)
    else:
        results = result_list
    return results


def compute_single_noise_free_lightcurve(
    model,
    graph_state,
    passbands,
    *,
    rest_frame_phase_min=-50.0,
    rest_frame_phase_max=50.0,
    rest_frame_phase_step=2.0,
):
    """Compute the noise-free light curve for a single object.

    This is a helper function for compute_noise_free_lightcurves.

    Parameters
    ----------
    model : BasePhysicalModel
        The model object to use for generating the light curves.
    graph_state : GraphState
        The state of the graph for the simulation. Must be a single state
        (num_samples=1).
    passbands : PassbandGroup
        The passbands to use for generating the bandfluxes.
    rest_frame_phase_min : float or np.ndarray
        The minimum rest-frame phase (in days) at which to evaluate the light curve.
        If an array is given, it must match the number of samples in graph_state.
        Default is -50.0 days.
    rest_frame_phase_max : float or np.ndarray
        The maximum rest-frame phase (in days) at which to evaluate the light curve.
        If an array is given, it must match the number of samples in graph_state.
        Default is 50.0 days.
    rest_frame_phase_step : float or np.ndarray
        The step size (in days) between rest-frame phases at which to evaluate the light curve.
        If an array is given, it must match the number of samples in graph_state.
        Default is 2.0 days.

    Returns
    -------
    lightcurves : dict
        A dictionary mapping each filter name to the corresponding array of bandfluxes (in nJy),
        with an additional keys "times" (in MJD) and "rest_phase" (in days relative to t0).
    """
    if graph_state.num_samples != 1:
        raise ValueError("graph_state must have num_samples=1.")

    # Generate the rest-frame times at which to evaluate the light curve.
    rest_phase = np.arange(rest_frame_phase_min, rest_frame_phase_max, step=rest_frame_phase_step)
    times = rest_phase.copy()

    # Compute the observed-frame times (MJD) if needed.
    redshift = model.get_param(graph_state, "redshift")
    if redshift is not None and redshift > 0:
        times = times * (1 + redshift)

    t0 = model.get_param(graph_state, "t0")
    if t0 is not None:
        times += t0

    # Compute the light curve without noise for each time in each filter.
    lightcurves = {"times": times, "rest_phase": rest_phase}
    for filter in passbands.filters:
        filters_array = np.full(len(times), filter)
        bandfluxes = model.evaluate_bandfluxes(passbands, times, filters_array, graph_state)
        lightcurves[filter] = bandfluxes
    return lightcurves


def compute_noise_free_lightcurves(
    model,
    graph_state,
    passbands,
    *,
    rest_frame_phase_min=-50.0,
    rest_frame_phase_max=50.0,
    rest_frame_phase_step=2.0,
):
    """Compute the noise-free light curves for a given model and one more more states
    at given times (in either MJD or rest-frame phase).

    This function simulates the light curves without adding any noise, allowing
    for the analysis of the underlying model behavior.

    Parameters
    ----------
    model : BasePhysicalModel
        The model object to use for generating the light curves.
    graph_state : GraphState
        The state of the graph for the simulation.
    passbands : PassbandGroup
        The passbands to use for generating the bandfluxes.
    rest_frame_phase_min : float or np.ndarray
        The minimum rest-frame phase (in days) at which to evaluate the light curve.
        If an array is given, it must match the number of samples in graph_state.
        Default is -50.0 days.
    rest_frame_phase_max : float or np.ndarray
        The maximum rest-frame phase (in days) at which to evaluate the light curve.
        If an array is given, it must match the number of samples in graph_state.
        Default is 50.0 days.
    rest_frame_phase_step : float or np.ndarray
        The step size (in days) between rest-frame phases at which to evaluate the light curve.
        If an array is given, it must match the number of samples in graph_state.
        Default is 2.0 days.

    Returns
    -------
    lightcurves : nested_pandas.NestedFrame
        A NestedFrame with a row for each object that contains a "lightcurve" column
        with a dictionary mapping each filter name to the corresponding array of
        bandfluxes (in nJy), with an additional nested columns "times" (in MJD) and
        "rest_phase" (in days relative to t0).
    """
    # Expand any scalar inputs to arrays.
    num_samples = graph_state.num_samples
    if np.isscalar(rest_frame_phase_min):
        rest_frame_phase_min = np.full(num_samples, rest_frame_phase_min)
    if np.isscalar(rest_frame_phase_max):
        rest_frame_phase_max = np.full(num_samples, rest_frame_phase_max)
    if np.isscalar(rest_frame_phase_step):
        rest_frame_phase_step = np.full(num_samples, rest_frame_phase_step)

    # Set up the per-model dictionary.
    results_dict = {
        "id": np.arange(num_samples).tolist(),
        "ra": np.atleast_1d(model.get_param(graph_state, "ra")).tolist(),
        "dec": np.atleast_1d(model.get_param(graph_state, "dec")).tolist(),
        "z": np.atleast_1d(model.get_param(graph_state, "redshift")).tolist(),
    }

    # Set up the nested dictionary with the light curve information.
    all_filters = passbands.filters
    nested_index = []
    nested_dict = {"times": [], "rest_phase": []}
    for filter in all_filters:
        nested_dict[filter] = []

    # Compute each light curve.
    for idx, state_i in enumerate(graph_state):
        lc = compute_single_noise_free_lightcurve(
            model,
            state_i,
            passbands,
            rest_frame_phase_min=rest_frame_phase_min[idx],
            rest_frame_phase_max=rest_frame_phase_max[idx],
            rest_frame_phase_step=rest_frame_phase_step[idx],
        )

        # Append the light curve data onto the nested dictionary.
        for key in nested_dict:
            nested_dict[key].extend(list(lc[key]))
        nested_index.extend([idx] * len(lc["times"]))

    # Create the nested results frame.
    results = NestedFrame(data=results_dict, index=[i for i in range(num_samples)])
    nested_frame = pd.DataFrame(data=nested_dict, index=nested_index)
    results = results.join_nested(nested_frame, "lightcurve")
    return results
