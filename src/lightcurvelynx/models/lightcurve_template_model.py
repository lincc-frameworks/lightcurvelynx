"""Model that generate the SED or bandflux of a source based on given observer frame
light curves of fluxes in each band.

If we are generating the bandfluxes directly, the models interpolate the given light curves
at the requested times and filters. If we are generating an SED for a given set of
wavelengths, the model computes a box-shaped SED basis function for each filter that
will produce the same bandflux after being passed through the passband filter.

Note: If you are interested in generating SED-level data, use the SEDTemplateModel in
src/lightcurvelynx/models/sed_template_model.py instead.
"""

import logging
from abc import ABC

import matplotlib.pyplot as plt
import numpy as np
from citation_compass import cite_inline
from tqdm import tqdm

from lightcurvelynx.astro_utils.mag_flux import mag2flux
from lightcurvelynx.astro_utils.passbands import Passband, PassbandGroup
from lightcurvelynx.astro_utils.sed_basis_models import SEDBasisModel
from lightcurvelynx.consts import lsst_filter_plot_colors
from lightcurvelynx.math_nodes.given_sampler import GivenValueSampler, GivenValueSelector
from lightcurvelynx.models.physical_model import BandfluxModel
from lightcurvelynx.utils.io_utils import read_lclib_data

logger = logging.getLogger(__name__)


class LightcurveBandData:
    """A class to hold data for a single model light curve defined at the band level (a set of
    fluxes over time for each filter).

    Data can be passed in as fluxes (in nJy) or AB magnitudes (if magnitudes_in=True), but
    is always stored internally as fluxes.

    Attributes
    ----------
    lightcurves : dict
        A dictionary mapping filter names to a 2D array of the bandfluxes in that filter,
        where the first column is time (in days from the reference time of the light curve),
        the second column is the bandflux (in nJy), and an optional third column is fluxerror.
    lc_data_t0 : float
        The reference epoch of the input light curve. This is the time stamp of the input
        array that will correspond to t0 in the model. For periodic light curves, this either
        must be set to the first time of the light curve or set as 0.0 to automatically
        derive the lc_data_t0 from the light curve.
    period : float or None
        The period of the light curve in days. If the light curve is not periodic,
        then this value is set to None.
    min_times : dict
        A dictionary mapping filter names to the minimum time of the light curve
        in that filter (shifted to be relative to the reference epoch of the light curve).
    max_times : dict
        A dictionary mapping filter names to the maximum time of the light curve
        in that filter (shifted to be relative to the reference epoch of the light curve).
    baseline : dict
        A dictionary of baseline bandfluxes for each filter (in nJy). This is only used
        for non-periodic light curves when they are not active.
    has_time_bounds : bool
        Whether the model has time bounds where it is valid. This is True for non-periodic
        light curves without a baseline, and False otherwise.

    Parameters
    ----------
    lightcurves : dict or numpy.ndarray
        The light curves can be passed as either:
        1) a dictionary mapping filter names to a (T, 2) array of the bandfluxes in that filter
        where the first column is time and the second column is the light curve values, or
        2) a numpy array of shape (T, 3) array where the first column is time (in days), the
        second column is the light curve values, and the third column is the filter.
        The light curve values can be either fluxes (in nJy) or AB magnitudes (if magnitudes_in=True).
    lc_data_t0 : float
        The reference epoch of the input light curve. The model will be shifted
        to the model's lc_data_t0 when computing fluxes.  For periodic light curves, this either
        must be set to the first time of the light curve or as 0.0 to automatically
        derive the lc_data_t0 from the light curve.
    periodic : bool
        Whether the light curve is periodic. If True, the model will assume that
        the light curve repeats every period.
        Default: False
    magnitudes_in : bool
        Whether the input light curves are in AB magnitudes (True) or fluxes (False).
    baseline : dict or None
        A dictionary of baseline bandfluxes or AB magnitudes for each filter. This is only used
        for non-periodic light curves when they are not active.
        Default: None
    """

    def __init__(
        self,
        lightcurves,
        lc_data_t0,
        *,
        periodic=False,
        magnitudes_in=False,
        baseline=None,
    ):
        if lc_data_t0 is None:
            raise ValueError("lc_data_t0 must be provided and cannot be None.")
        self.lc_data_t0 = lc_data_t0
        self.period = None

        if isinstance(lightcurves, dict):
            # Make a copy of the light curves to avoid modifying the original data.
            self.lightcurves = {filter: lc.copy() for filter, lc in lightcurves.items()}
        elif isinstance(lightcurves, np.ndarray):
            if lightcurves.shape[1] != 3:
                raise ValueError("Light curves array must have 3 columns: time, flux, and filter.")

            # Break up the light curves by filter.
            self.lightcurves = {}
            filters = np.unique(lightcurves[:, 2])
            for filter in filters:
                filter_mask = lightcurves[:, 2] == filter
                filter_times = lightcurves[filter_mask, 0].astype(float)
                filter_bandflux = lightcurves[filter_mask, 1].astype(float)
                self.lightcurves[str(filter)] = np.column_stack((filter_times, filter_bandflux))
        else:
            raise TypeError(
                "Unknown type for light curve input. Must be dict, numpy array, or astropy Table."
            )

        # Do basic validation of the light curves and shift them so that the time
        # at lc_data_t0 is mapped to 0.0. Convert from AB magnitudes to fluxes if needed.
        for filter, lc in self.lightcurves.items():
            if len(lc.shape) != 2 or (lc.shape[1] != 2 and lc.shape[1] != 3):
                raise ValueError(f"Lightcurve {filter} must have either 2 or 3 columns.")
            if lc.shape[1] == 3:
                lc = lc[:, :2]  # Drop the error column if present.
            if not np.all(np.diff(lc[:, 0]) > 0):
                raise ValueError(f"Lightcurve {filter}'s times are not in sorted order.")

            # Shift the light curve times to be relative to lc_data_t0.
            lc[:, 0] -= self.lc_data_t0

            # Convert from magnitudes to fluxes if needed.
            if magnitudes_in:
                lc[:, 1] = mag2flux(lc[:, 1])

        # Store the minimum and maximum times for each light curve. This is done after
        # validating periodicity in case we needed to adjust the light curve start times.
        if periodic:
            self._validate_periodicity()
        self.min_times = {filter: lc[0, 0] for filter, lc in self.lightcurves.items()}
        self.max_times = {filter: lc[-1, 0] for filter, lc in self.lightcurves.items()}

        # If the model is periodic or has a given baseline, it is considered valid
        # outside the minimum and maximum times of each light curve.
        self.has_time_bounds = (not periodic) and baseline is None

        # Store the baseline values for each filter. If the baseline is provided,
        # make sure it contains all of the filters. If no baseline is provided,
        # set the baseline to 0.0 for each filter.
        if baseline is None:
            self.baseline = {filter: 0.0 for filter in self.lightcurves}
        else:
            for filter in self.lightcurves:
                if filter not in baseline:
                    raise ValueError(f"Baseline value for filter {filter} is missing.")
            self.baseline = baseline

        # Convert the baseline from magnitudes to fluxes if needed.
        if magnitudes_in:
            for filter in self.baseline:
                self.baseline[filter] = mag2flux(self.baseline[filter])

    def __len__(self):
        """Get the number of light curves."""
        return len(self.lightcurves)

    @property
    def filters(self):
        """Get the list of filters in the lightcurves."""
        return list(self.lightcurves.keys())

    def _validate_periodicity(self):
        """Check that the light curves meet the requirements for periodic models:
        - All light curves must be sampled at the same times.
        - The light curves must have a non-zero period.
        - The value at the start and end of each light curve must be the same.
        """
        all_lcs = list(self.lightcurves.values())
        if len(all_lcs) == 0:
            raise ValueError("Periodic light curve models must have at least one light curve.")
        if len(all_lcs[0]) < 2:
            raise ValueError("All periodic light curves must have at least two time points.")

        # Check that all light curves are sampled at the same times and the first value
        # matches the last value.
        num_curves = len(all_lcs)
        for i in range(num_curves):
            if not np.allclose(all_lcs[i][:, 0], all_lcs[0][:, 0]):
                raise ValueError("All light curves in a periodic model must be sampled at the same times.")
            if not np.allclose(all_lcs[i][0, 1], all_lcs[i][-1, 1]):
                raise ValueError("All periodic light curves must have the same value at the start and end.")

        # Check that all light curves have a non-zero period.
        self.period = all_lcs[0][-1, 0] - all_lcs[0][0, 0]
        if self.period <= 0.0:
            raise ValueError("The period of the light curve must be positive.")

        # Shift all the lightcurves so they start at 0 (to make the math easier)
        # and record the offset as lc_data_t0.
        if not np.isclose(all_lcs[0][0, 0], 0.0):
            if self.lc_data_t0 != 0.0:
                raise ValueError(
                    "For periodic models, lc_data_t0 must either be set to the first time "
                    f"or automatically derived. Found lc_data_t0={self.lc_data_t0}."
                )

            self.lc_data_t0 = all_lcs[0][0, 0]
            for lc in self.lightcurves.values():
                lc[:, 0] -= self.lc_data_t0

    @classmethod
    def from_lclib_table(cls, lightcurves_table, *, forced_lc_t0=None, filters=None):
        """Break up a light curves table in LCLIB format into a LightcurveBandData instance.
        This function expects the table to have a "time" column, an optional "type" column,
        and a column for each filter. The "type" column should use "S" for source observation
        and "T" for template (background) observation.

        Parameters
        ----------
        lightcurves_table : astropy.table.Table
            A table with a "time" column, optional "type" column, and a column for each filter.
            If the type column is present it should use "S" for source observation and "T"
            for template (background) observation.
        forced_lc_t0 : float
            By default we use the LCLIB convention of storing the light curves so the first time
            corresponds to the reference epoch (lc_data_t0) of the light curve. This can be overridden
            by providing a value for forced_lc_t0.
            Default: None
        filters : list of str or None
            A list of filters to use for the light curves. If None, all filters will be used.
            Used to select a subset of filters.
            Default: None
        """
        if "time" not in lightcurves_table.colnames:
            raise ValueError("Light curves table must have a 'time' column.")

        # Extract the name of the filters from the table column names.
        filter_cols = [col for col in lightcurves_table.colnames if col != "time" and col != "type"]
        if filters is None:
            filters = filter_cols
        else:
            to_keep = set(filter_cols) & set(filters)
            filters = list(to_keep)
        if len(filters) == 0:
            raise ValueError("Light curves table must have at least one filter column.")

        # Check if there are baseline curves to extract and filter them out of the
        # light curves table. Use a default to 0.0 for each filter if no baselines are found.
        baseline = {filter: 0.0 for filter in filters}
        if "type" in lightcurves_table.colnames:
            obs_mask = lightcurves_table["type"] == "S"
            if np.any(~obs_mask):
                tmp_table = lightcurves_table[~obs_mask]
                if len(tmp_table) > 1:
                    logger.warning(
                        "Multiple template (background) observations found in light curves table. "
                        "The light curve will only use the first one for baseline values."
                    )
                baseline = {filter: mag2flux(tmp_table[filter][0]) for filter in filters}
            lightcurves_table = lightcurves_table[obs_mask]

        # Determine the reference epoch of the light curve (lc_data_t0).
        lc_data_t0 = np.min(lightcurves_table["time"]) if forced_lc_t0 is None else forced_lc_t0

        # Convert the Table to a dictionary of lightcurves.
        lightcurves = {}
        for filter in filters:
            filter_times = lightcurves_table["time"].astype(float)
            filter_bandflux = mag2flux(lightcurves_table[filter].astype(float))
            lightcurves[str(filter)] = np.column_stack((filter_times, filter_bandflux))

        # Check the metadata for periodicity information.
        recur_class = lightcurves_table.meta.get("RECUR_CLASS", "")
        if recur_class == "PERIODIC" or recur_class == "RECUR-PERIODIC":
            periodic = True
            baseline = None  # Baseline is not used for periodic light curves.
        elif recur_class == "RECUR-NONPERIODIC":
            periodic = False
            logger.warning(
                "Recurring non-periodic light curves are treated as non-recurring within LightCurveLynx."
            )
        elif recur_class == "NON-RECUR":
            periodic = False
        elif recur_class == "":
            periodic = False
            logger.warning(
                "No RECUR_CLASS metadata found in light curves table. Using non-periodic light curves."
            )
        else:
            raise ValueError(
                f"Unknown RECUR_CLASS value in light curves table metadata: {recur_class}. "
                "Expected 'PERIODIC', 'RECUR-PERIODIC', 'RECUR-NONPERIODIC', or 'NON-RECUR'."
            )

        # If the light curves are periodic, make sure they start and end at the same value.
        if periodic:
            all_match = True
            for lc in lightcurves.values():
                all_match &= np.isclose(lc[0, 1], lc[-1, 1])

            # Insert a value to wrap. This should be a bit after the last time
            # and have the same value as the first time.
            if not all_match:
                dt = lightcurves_table["time"][-1] - lightcurves_table["time"][0]
                ave_dt = dt / (len(lightcurves_table) - 1)
                new_end = lightcurves_table["time"][-1] + ave_dt

                for filter, lc in lightcurves.items():
                    lc = np.vstack((lc, [new_end, lc[0, 1]]))
                    lightcurves[filter] = lc

        return cls(lightcurves, lc_data_t0, periodic=periodic, baseline=baseline)

    def evaluate_bandfluxes(self, times, filter):
        """Get the bandflux values for a given filter at the specified times. These can
        be multiplied by a basis SED function to produce estimated SED values
        for the given filter at the specified times or can be used directly as bandfluxes.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of times (in days) at which to compute the SED values. These should
            be shift to be relative to the light curve's lc_data_t0.
        filter : str
            The name of the filter for which to compute the SED values.

        Returns
        -------
        values : numpy.ndarray
            A length T array of bandpass fluxes for the specified filter at the given times.
        """
        if filter not in self.lightcurves:
            raise ValueError(f"Filter {filter} not found in light curves.")
        lightcurve = self.lightcurves[filter]

        # If the light curve is periodic, wrap the times around the period.
        if self.period is not None:
            times = times % self.period

        # Start with an array of all baseline values.
        values = np.full(len(times), self.baseline.get(filter, 0.0))

        # For the times that overlap with the light curve, interpolate the light curve values.
        overlap = (times >= self.min_times[filter]) & (times <= self.max_times[filter])
        values[overlap] = np.interp(
            times[overlap],  # The query times
            lightcurve[:, 0],  # The light curve times for this passband filter
            lightcurve[:, 1],  # The light curve flux densities for this passband filter
            left=0.0,  # Do not extrapolate in time
            right=0.0,  # Do not extrapolate in time
        )

        return values

    def plot_lightcurves(self, times=None, ax=None, figure=None):
        """Plot the underlying light curves. This is a debugging
        function to help the user understand the SEDs produced by this
        model.

        Parameters
        ----------
        times : numpy.ndarray or None, optional
            An array of timestamps at which to plot the light curves.
            If None, the function uses the timestamps from each light curve.
        ax : matplotlib.pyplot.Axes or None, optional
            Axes, None by default.
        figure : matplotlib.pyplot.Figure or None
            Figure, None by default.
        """
        if ax is None:
            if figure is None:
                figure = plt.figure()
            ax = figure.add_axes([0, 0, 1, 1])

        # Plot each passband.
        for filter_name, filter_curve in self.lightcurves.items():
            # Check if we need to use the query times.
            if times is None:
                plot_times = filter_curve[:, 0]
                plot_values = filter_curve[:, 1]
            else:
                plot_times = times
                plot_values = np.interp(times, filter_curve[:, 0], filter_curve[:, 1], left=0.0, right=0.0)

            color = lsst_filter_plot_colors.get(filter_name, "black")
            ax.plot(plot_times, plot_values, color=color, label=filter_name)

        # Set the x and y axis labels.
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Filter value (nJy)")
        ax.set_title("Underlying Light Curves")
        ax.legend()


class BaseLightcurveBandTemplateModel(BandfluxModel, ABC):
    """A base class for light curve template models. This class is not meant to be used directly,
    but rather as a base for other light curve template models that may have additional functionality.
    It provides the basic structure (primarily SED basis functions) and validation for
    light curve-based SED models.

    The set of passbands used to configure the model MUST be the same as used
    to generate the SED (the wavelengths must match).

    Parameterized values include:
      * dec - The object's declination in degrees.
      * ra - The object's right ascension in degrees.
      * t0 - The t0 of the zero phase (if applicable), date.

    Attributes
    ----------
    sed_basis: SEDBasisModel, optional
        An SEDBasisModel mapping representing the fake SED basis functions for each filter.
        Only generated if passbands are provided.

    Parameters
    ----------
    passbands : Passband or PassbandGroup, optional
        The passband or passband group to use for defining the light curve. If provided, they
        will be used to create box-shaped SED basis functions for each filter.
    filters : list, optional
        A list of filter names that the model supports. If None then
        all available filters will be used.
    """

    def __init__(self, *, passbands=None, filters=None, **kwargs):
        super().__init__(**kwargs)

        # Convert a single passband to a PassbandGroup.
        if isinstance(passbands, Passband):
            passbands = PassbandGroup(given_passbands=[passbands])

        # Create the SED basis functions for each filter.
        if passbands is not None:
            self.sed_basis = SEDBasisModel.from_box_approximation(passbands, filters=filters)
        else:
            self.sed_basis = None

        # Check that t0 is set.
        if "t0" not in kwargs or kwargs["t0"] is None:
            raise ValueError("Light curve models require a t0 parameter.")

    def compute_sed_given_lc(self, lc, times, wavelengths, graph_state):
        """Compute the flux density for a given light curve at specified times and wavelengths.

        Parameters
        ----------
        lc : LightcurveBandData
            The light curve data to use for computing the flux density.
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
        if self.sed_basis is None:
            raise ValueError("SED basis functions are not defined for this model.")

        params = self.get_local_params(graph_state)

        # Shift the times for the model's t0 aligned with the light curve's reference epoch.
        shifted_times = times - params["t0"]

        flux_density = np.zeros((len(times), len(wavelengths)))
        for filter in lc.filters:
            # Compute the SED values for the wavelengths we are actually sampling.
            sed_waves = self.sed_basis.compute_sed(filter, wavelengths=wavelengths)

            # Compute the multipliers for the SEDs at different time steps along this light curve.
            # We use the light curve's baseline value for all times outside the light curve's range.
            sed_time_mult = lc.evaluate_bandfluxes(shifted_times, filter)

            # The contribution of this filter to the overall SED is the light curve's (interpolated)
            # value at each time multiplied by the SED values at each query wavelength.
            sed_flux = np.outer(sed_time_mult, sed_waves)
            flux_density += sed_flux

        # Return the total flux density from all light curves.
        return flux_density

    def plot_sed_basis(self, ax=None, figure=None):
        """Plot the basis functions for the SED.  This is a debugging
        function to help the user understand the SEDs produced by this
        model.

        Parameters
        ----------
        ax : matplotlib.pyplot.Axes or None, optional
            Axes, None by default.
        figure : matplotlib.pyplot.Figure or None
            Figure, None by default.
        """
        if self.sed_basis is None:
            raise ValueError("SED basis functions are not defined for this model.")
        self.sed_basis.plot(ax=ax, figure=figure)


class LightcurveTemplateModel(BaseLightcurveBandTemplateModel):
    """A model that generates either the SED or bandflux of a source based on
    given light curves in each band. When generating the bandflux, it interpolates
    the light curves directly. When generating the SED, the model uses a box-shaped SED
    for each filter such that the resulting flux density is equal to the light curve's
    value after passing through the passband filter.

    LightcurveTemplateModel supports both periodic and non-periodic light curves. If the
    light curve is not periodic then each light curve's given values will be interpolated
    during the time range of the light curve. Values outside the time range (before and
    after) will be set to the baseline value for that filter (0.0 by default).

    Periodic models require that each filter's light curve is sampled at the same times
    and that the value at the end of the light curve is equal to the value at the start
    of the light curve. The light curve epoch (lc_data_t0) is automatically set to the first
    time so that the t0 parameter corresponds to the shift in phase.

    The set of passbands used to configure the model MUST be the same as used
    to generate the SED (the wavelengths must match).

    Parameterized values include:
      * dec - The object's declination in degrees.
      * ra - The object's right ascension in degrees.
      * t0 - The t0 of the zero phase (if applicable), date.

    Notes
    -----
    If you are interested in generating SED-level data, use the SEDTemplateModel in
    src/lightcurvelynx/models/sed_template_model.py instead.

    Attributes
    ----------
    lightcurves : LightcurveBandData
        The data for the light curves, such as the times and bandfluxes in each filter.
    sed_values : dict
        A dictionary mapping filters to the SED basis values for that passband.
        These SED values are scaled by the light curve and added for the
        final SED.
    filters : list
        The list of filters in the light curves.

    Parameters
    ----------
    lightcurves : dict or numpy.ndarray
        The light curves can be passed as either:
        1) a LightcurveBandData instance,
        2) a dictionary mapping filter names to a (T, 2) array of the bandlfuxes in that filter
        where the first column is time and the second column is the flux density (in nJy), or
        3) a numpy array of shape (T, 3) array where the first column is time (in days), the
        second column is the bandflux (in nJy), and the third column is the filter.
    passbands : Passband or PassbandGroup or None
        The passband or passband group to use for defining the light curve. If provided (not None),
        these will be used to create box-shaped SED basis functions for each filter.
    lc_data_t0 : float
        The reference epoch of the input light curve. This is the time stamp of the input
        array that will correspond to t0 in the model. For periodic light curves, this either
        must be set to the first time of the light curve or set as 0.0 to automatically
        derive the lc_data_t0 from the light curve.
    periodic : bool
        Whether the light curve is periodic. If True, the model will assume that
        the light curve repeats every period.
        Default: False
    baseline : dict or None
        A dictionary of baseline bandfluxes for each filter. This is only used
        for non-periodic light curves when they are not active.
        Default: None
    """

    def __init__(
        self,
        lightcurves,
        passbands,
        lc_data_t0,
        *,
        periodic=False,
        baseline=None,
        **kwargs,
    ):
        # Store the light curve data, parsing out different formats if needed.
        if isinstance(lightcurves, LightcurveBandData):
            self.lightcurves = lightcurves
        else:
            self.lightcurves = LightcurveBandData(
                lightcurves,
                lc_data_t0,
                periodic=periodic,
                baseline=baseline,
            )
        self.filters = self.lightcurves.filters
        super().__init__(passbands=passbands, filters=self.filters, **kwargs)

        # Raise a warning if time extrapolation is provided but cannot be used.
        if "time_extrapolation" in kwargs and kwargs["time_extrapolation"] is not None:
            if periodic:
                logger.warning("time_extrapolation is provided, but is not used for periodic light curves. ")
            elif baseline is None:
                logger.warning(
                    "time_extrapolation is provided, but is not used for light curves without a baseline. "
                )

    def minphase(self, filter=None, **kwargs):
        """Get the minimum supported phase of the model (for this filter) in days.

        Parameters
        ----------
        filter : str
            The name of the filter (required). An error is raised if no value is provided.
        **kwargs : dict
            Additional keyword arguments, not used in this method.

        Returns
        -------
        minphase : float or None
            The minimum phase of the model (in days) or None
            if the model does not have a defined minimum phase.
        """
        if filter is None:
            raise ValueError("Filter must be provided to compute minphase.")

        if self.lightcurves.has_time_bounds:
            return self.lightcurves.min_times[filter]
        else:
            return None

    def maxphase(self, filter=None, **kwargs):
        """Get the maximum supported phase of the model (for this filter) in days.

        Parameters
        ----------
        filter : str
            The name of the filter (required). An error is raised if no value is provided.
        graph_state : GraphState, optional
            An object mapping graph parameters to their values. If provided,
            the function will use the graph state to compute the maximum wavelength.
        **kwargs : dict
            Additional keyword arguments, not used in this method.

        Returns
        -------
        maximum : float or None
            The maximum phase of the model (in days) or None
            if the model does not have a defined maximum phase.
        """
        if filter is None:
            raise ValueError("Filter must be provided to compute maxphase.")

        if self.lightcurves.has_time_bounds:
            return self.lightcurves.max_times[filter]
        else:
            return None

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
            A length T x N matrix of observer frame SED values (in nJy). These are generated
            from non-overlapping box-shaped SED basis functions for each filter and
            scaled by the light curve values.
        """
        return self.compute_sed_given_lc(
            self.lightcurves,
            times,
            wavelengths,
            graph_state,
        )

    def compute_bandflux(self, times, filter, state, **kwargs):
        """Evaluate the model at the passband level for a single, given graph state.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of observer frame timestamps in MJD.
        filter : str
            The name of the filter.
        state : GraphState
            An object mapping graph parameters to their values with num_samples=1.
        **kwargs : dict
            Additional keyword arguments, not used in this method.

        Returns
        -------
        bandfluxes : numpy.ndarray
            A length T matrix of observer frame passband fluxes (in nJy).
        """
        params = self.get_local_params(state)

        # Check that the filters are all supported by the model.
        if filter not in self.lightcurves.lightcurves:
            raise ValueError(f"Filter '{filter}' is not supported by LightcurveTemplateModel.")

        # Shift the times for the model's t0 aligned with the light curve's reference epoch.
        shifted_times = times - params["t0"]
        bandfluxes = self.lightcurves.evaluate_bandfluxes(shifted_times, filter)
        return bandfluxes

    def plot_lightcurves(self, times=None, ax=None, figure=None):
        """Plot the underlying light curves. This is a debugging
        function to help the user understand the SEDs produced by this
        model.

        Parameters
        ----------
        times : numpy.ndarray or None, optional
            An array of timestamps at which to plot the light curves.
            If None, the function uses the timestamps from each light curve.
        ax : matplotlib.pyplot.Axes or None, optional
            Axes, None by default.
        figure : matplotlib.pyplot.Figure or None
            Figure, None by default.
        """
        self.lightcurves.plot_lightcurves(times=times, ax=ax, figure=figure)


class MultiLightcurveTemplateModel(BaseLightcurveBandTemplateModel):
    """A MultiLightcurveTemplateModel randomly selects a light curve at each evaluation
    computes the flux from that source. The models can generate either the SED or
    bandflux of a source based of given light curves in each band. When generating
    the bandflux, the model interpolates the light curves directly. When generating the SED,
    the model uses a box-shaped SED for each filter such that the resulting flux density
    is equal to the light curve's value after passing through the passband filter.

    MultiLightcurveTemplateModel supports both periodic and non-periodic light curves. If the
    light curve is not periodic then each light curve's given values will be interpolated
    during the time range of the light curve. Values outside the time range (before and
    after) will be set to the baseline value for that filter (0.0 by default).

    Periodic models require that each filter's light curve is sampled at the same times
    and that the value at the end of the light curve is equal to the value at the start
    of the light curve. The light curve epoch is automatically set to the first time
    so that the t0 parameter corresponds to the shift in phase.

    The set of passbands used to configure the model MUST be the same as used
    to generate the SED (the wavelengths must match).

    Parameterized values include:
      * dec - The object's declination in degrees.
      * ra - The object's right ascension in degrees.
      * t0 - The t0 of the zero phase (if applicable), date.

    Attributes
    ----------
    lightcurves : list of LightcurveBandData
        The data for each set of light curves.
    sed_values : dict
        A dictionary mapping filters to the SED basis values for that passband.
        These SED values are scaled by the light curve and added for the
        final SED.
    all_filters : set
        A set of all filters used by the light curves. This is the union of all
        filters used by each light curve in the lightcurves list.

    Parameters
    ----------
    lightcurves : list of LightcurveBandData
        The data for each set of light curves. One light curve will be randomly selected
        at each evaluation.
    passbands : Passband or PassbandGroup or None, optional
        The passband or passband group to use for defining the light curve. If provided (not None),
        these will be used to create box-shaped SED basis functions for each filter.
        Default: None
    weights : numpy.ndarray, optional
        A length N array indicating the relative weight from which to select
        a light curve at random. If None, all light curves will be weighted equally.
    """

    def __init__(
        self,
        lightcurves,
        passbands=None,
        *,
        weights=None,
        **kwargs,
    ):
        # Validate the light curve input and create a union of all filters used.
        all_filters = set()
        for lc in lightcurves:
            if not isinstance(lc, LightcurveBandData):
                raise TypeError("Each light curve must be an instance of LightcurveBandData.")
            all_filters.update(lc.filters)
        self.filters = list(all_filters)
        self.lightcurves = lightcurves

        super().__init__(passbands=passbands, filters=self.filters, **kwargs)

        all_inds = [i for i in range(len(lightcurves))]
        self._sampler_node = GivenValueSampler(all_inds, weights=weights)
        self.add_parameter(
            "selected_lightcurve",
            value=self._sampler_node,
            allow_gradient=False,
            description="Index of the light curve selected for sampling.",
        )

        # Assemble a list of baseline values for each filter across all light curves.
        # Create a parameter to track the baseline values for the selected light curve. The node
        # will automatically fill in the correct baseline value based on the index given by
        # the selected_lightcurve parameter.
        for fltr in self.filters:
            baselines = [lc.baseline.get(fltr, 0.0) for lc in lightcurves]
            baseline_selector = GivenValueSelector(baselines, self.selected_lightcurve)
            self.add_parameter(
                f"baseline_{fltr}",
                value=baseline_selector,
                allow_gradient=False,
                description=f"Baseline value for filter {fltr} from the selected light curve.",
            )

    def __len__(self):
        """Get the number of light curves."""
        return len(self.lightcurves)

    @classmethod
    def from_lclib_file(cls, lightcurves_file, passbands, *, forced_lc_t0=None, filters=None, **kwargs):
        """Create a MultiLightcurveTemplateModel from a light curves file in LCLIB format.

        Parameters
        ----------
        lightcurves_file : str
            The path to the light curves file in LCLIB format.
        passbands : Passband or PassbandGroup
            The passband or passband group to use for defining the light curve.
        forced_lc_t0 : float or ndarray, optional
            By default we use the LCLIB convention of storing the light curves so the first time
            corresponds to the reference epoch (lc_data_t0) of the light curve. This can be overridden
            by providing a value for forced_lc_t0.
            Default: None
        filters : list of str, optional
            A list of filters to use for the light curves. If None, all filters will be used.
            Used to select a subset of filters that match the survey to simulate.
            Default: None
        **kwargs
            Additional keyword arguments to pass to the LightcurveBandData constructor, including
            the parameters for the model such as `dec`, `ra`, and `t0` and metadata
            such as `node_label`.

        Returns
        -------
        MultiLightcurveTemplateModel
            An instance of MultiLightcurveTemplateModel with the loaded light curves.
        """
        lightcurve_tables = read_lclib_data(lightcurves_file)
        if lightcurve_tables is None or len(lightcurve_tables) == 0:
            raise ValueError(f"Could not read light curves from file: {lightcurves_file}")

        if forced_lc_t0 is None:
            forced_lc_t0 = np.full(len(lightcurve_tables), None)
        elif np.isscalar(forced_lc_t0):
            forced_lc_t0 = np.full(len(lightcurve_tables), forced_lc_t0)
        elif len(forced_lc_t0) != len(lightcurve_tables):
            raise ValueError(
                "If provided as an array, forced_lc_t0 must have the same "
                "length as the number of light curves."
            )

        lightcurves = []
        for table, lc_t0 in tqdm(
            zip(lightcurve_tables, forced_lc_t0, strict=False), desc="Loading", unit="lc"
        ):
            lc_data = LightcurveBandData.from_lclib_table(table, forced_lc_t0=lc_t0, filters=filters)
            lightcurves.append(lc_data)

        # Add a citation for LCLIB if we loaded from an LCLIB file.
        cite_inline("LCLIB Data", f"LCLIB Data from the file {lightcurves_file}")

        return cls(lightcurves, passbands, **kwargs)

    def minphase(self, filter=None, graph_state=None, **kwargs):
        """Get the minimum supported phase of the model (for this filter) in days.

        Parameters
        ----------
        filter : str
            The name of the filter (required). An error is raised if no value is provided.
        graph_state : GraphState, optional
            An object mapping graph parameters to their values. If provided,
            the function will use the graph state to compute the minimum wavelength.
        **kwargs : dict
            Additional keyword arguments, not used in this method.

        Returns
        -------
        minphase : float or None
            The minimum phase of the model (in days) or None
            if the model does not have a defined minimum phase.
        """
        if filter is None:
            raise ValueError("Filter must be provided to compute minphase.")
        if graph_state is None:
            raise ValueError("Graph state must be provided to compute minphase.")
        if graph_state.num_samples > 1:
            raise ValueError("Graph state must have num_samples=1 to compute maxphase.")

        model_ind = self.get_param(graph_state, "selected_lightcurve")
        lc_model = self.lightcurves[model_ind]
        if lc_model.has_time_bounds:
            return lc_model.min_times[filter]
        else:
            return None

    def maxphase(self, filter=None, graph_state=None, **kwargs):
        """Get the maximum supported phase of the model (for this filter) in days.

        Parameters
        ----------
        filter : str
            The name of the filter (required). An error is raised if no value is provided.
        graph_state : GraphState, optional
            An object mapping graph parameters to their values. If provided,
            the function will use the graph state to compute the maximum wavelength.
        **kwargs : dict
            Additional keyword arguments, not used in this method.

        Returns
        -------
        maximum : float or None
            The maximum phase of the model (in days) or None
            if the model does not have a defined maximum phase.
        """
        if filter is None:
            raise ValueError("Filter must be provided to compute maxphase.")
        if graph_state is None:
            raise ValueError("Graph state must be provided to compute maxphase.")
        if graph_state.num_samples > 1:
            raise ValueError("Graph state must have num_samples=1 to compute maxphase.")

        model_ind = self.get_param(graph_state, "selected_lightcurve")
        lc_model = self.lightcurves[model_ind]
        if lc_model.has_time_bounds:
            return lc_model.max_times[filter]
        else:
            return None

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
            A length T x N matrix of observer frame SED values (in nJy). These are generated
            from non-overlapping box-shaped SED basis functions for each filter and
            scaled by the light curve values.
        """
        # Use the light curve selected by the sampler node to compute the flux density.
        model_ind = self.get_param(graph_state, "selected_lightcurve")
        return self.compute_sed_given_lc(
            self.lightcurves[model_ind],
            times,
            wavelengths,
            graph_state,
        )

    def compute_bandflux(self, times, filter, state, **kwargs):
        """Evaluate the model at the passband level for a single, given graph state.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of observer frame timestamps in MJD.
        filter : str
            The name of the filter.
        state : GraphState
            An object mapping graph parameters to their values with num_samples=1.
        **kwargs : dict
            Additional keyword arguments, not used in this method.

        Returns
        -------
        bandfluxes : numpy.ndarray
            A length T matrix of observer frame passband fluxes (in nJy).
        """
        params = self.get_local_params(state)
        model_ind = params["selected_lightcurve"]
        lc = self.lightcurves[model_ind]

        # Check that the filter is supported by the model.
        if filter not in lc.lightcurves:
            raise ValueError(f"Filter '{filter}' is not supported by LightcurveTemplateModel {model_ind}.")

        # Shift the times for the model's t0 aligned with the light curve's reference epoch.
        shifted_times = times - params["t0"]
        bandfluxes = lc.evaluate_bandfluxes(shifted_times, filter)
        return bandfluxes
