"""Utility functions for post processing the results data by adding statistics
columns and filtering on those columns."""

import numpy as np
import numpy.ma as ma
import pandas as pd
from nested_pandas import NestedFrame

from lightcurvelynx.astro_utils.mag_flux import flux2mag


def concat_results(results_list):
    """Concatenate a list of results into a single NestedFrame,
    updating the ID column to be unique across all results.

    Parameters
    ----------
    results_list : list of nested_pandas.NestedFrame
        The list of DataFrames to concatenate.

    Returns
    -------
    nested_pandas.NestedFrame
        The concatenated DataFrame.
    """
    result = pd.concat(results_list, ignore_index=True)

    # We need to update the ID column to be unique across all results.
    if "id" in result.columns:
        result["id"] = np.arange(len(result))
    return result


def results_drop_empty(results):
    """Drop empty lightcurves from the results DataFrame.

    Parameters
    ----------
    results : nested_pandas.NestedFrame
        The DataFrame containing lightcurve data.

    Returns
    -------
    nested_pandas.NestedFrame
        The DataFrame with empty lightcurves removed.
    """
    return results.dropna(subset=["lightcurve"])


def lightcurve_compute_snr(flux, fluxerr):
    """Compute the signal-to-noise ratio (SNR) for given flux and flux error arrays.

    Parameters
    ----------
    flux : array-like
        The flux values.
    fluxerr : array-like
        The flux error values.

    Returns
    -------
    result : np.ndarray
        The SNR values, with None for invalid entries (e.g., zero or negative flux error).
    """
    flux = np.asarray(flux)
    fluxerr = np.asarray(fluxerr)
    valid_mask = (flux > 0) & (fluxerr > 0)

    result = ma.masked_all(flux.shape)
    result[valid_mask] = flux[valid_mask] / fluxerr[valid_mask]
    return result


def lightcurve_compute_mag(flux, fluxerr):
    """Compute the AB magnitude and magnitude error for given flux and flux error arrays.

    Parameters
    ----------
    flux : array-like
        The flux values.
    fluxerr : array-like
        The flux error values.

    Returns
    -------
    tuple of np.ndarray
        The magnitude and magnitude error values, with None for invalid entries (e.g., non-positive flux).
    """
    flux = np.asarray(flux)
    fluxerr = np.asarray(fluxerr)
    valid_mask = (flux > 0) & (fluxerr > 0)

    mag = ma.masked_all(flux.shape)
    mag[valid_mask] = flux2mag(flux[valid_mask])

    magerr = ma.masked_all(flux.shape)
    magerr[valid_mask] = (2.5 / np.log(10)) * (fluxerr[valid_mask] / flux[valid_mask])

    return mag, magerr


def augment_single_lightcurve(results, *, min_snr=0.0, t0=None):
    """Add columns to a single lightcurve DataFrame with additional information
    about the light curve, including:
    - SNR = flux / fluxerr
    - detection flag (True if SNR >= min_snr, False otherwise)
    - AB magnitude
    - AB magnitude error = (2.5 / ln(10)) * (fluxerr / flux)
    - relative time = mjd - t0 (if t0 is provided)
    None is used for invalid entries, e.g. negative flux or zero flux error.

    Parameters
    ----------
    results : pandas.DataFrame
        The DataFrame containing lightcurve data. Modified in place.
    min_snr : float, optional
        Minimum SNR required to mark an entry as a detection. Default is 0.0.
    t0 : float or None, optional
        Reference time for the lightcurve.
    """
    if "flux" not in results.columns or "fluxerr" not in results.columns:
        raise ValueError("flux and fluxerr must be present in the light curve DataFrame.")
    flux = results["flux"]
    fluxerr = results["fluxerr"]

    snr = lightcurve_compute_snr(flux, fluxerr)
    results["snr"] = snr
    results["detection"] = [(x is not ma.masked and x >= min_snr) for x in snr]

    mag, magerr = lightcurve_compute_mag(flux, fluxerr)
    results["mag"] = mag
    results["magerr"] = magerr

    if t0 is not None and "mjd" in results.columns:
        results["time_rel"] = results["mjd"] - t0


def results_augment_lightcurves(results, *, min_snr=0.0):
    """Add columns to the results DataFrame with additional information
    about each light curve, including:
    - SNR = flux / fluxerr
    - detection flag (True if SNR >= min_snr, False otherwise)
    - AB magnitude
    - AB magnitude error = (2.5 / ln(10)) * (fluxerr / flux)
    - relative time = mjd - t0 (if t0 in the results table)
    None is used for invalid entries, e.g. negative flux or zero flux error.

    The input data frame can either be a single light curve (pandas.DataFrame)
    with columns "flux" and "fluxerr", or a NestedFrame (nested_pandas.NestedFrame)
    with a nested DataFrame column "lightcurve" that contains the "flux" and
    "fluxerr" columns.

    Parameters
    ----------
    results : pandas.DataFrame or nested_pandas.NestedFrame
        The DataFrame containing lightcurve data. Modified in place.
    min_snr : float, optional
        Minimum SNR required to mark an entry as a detection. Default is 0.0.
    """
    if not isinstance(results, NestedFrame) or "lightcurve" not in results.columns:
        raise ValueError("results must be a NestedFrame with a 'lightcurve' column.")
    if "flux" not in results["lightcurve"].nest.fields or "fluxerr" not in results["lightcurve"].nest.fields:
        raise ValueError("lightcurve.flux and lightcurve.fluxerr must be present in the DataFrame.")
    flux = results["lightcurve.flux"]
    fluxerr = results["lightcurve.fluxerr"]

    # Compute SNR and detection flag.
    snr = lightcurve_compute_snr(flux, fluxerr)
    results["lightcurve.snr"] = snr
    results["lightcurve.detection"] = [(x is not None and x >= min_snr) for x in snr]

    # Compute magnitude and magnitude error.
    mag, magerr = lightcurve_compute_mag(flux, fluxerr)
    results["lightcurve.mag"] = mag
    results["lightcurve.magerr"] = magerr

    # If t0 is provided as a column in results, compute relative time.
    if "t0" in results.columns and np.all(results["t0"]) and results["t0"].notna().all():
        if "mjd" not in results["lightcurve"].nest.fields:
            raise ValueError("lightcurve.mjd must be present in the DataFrame.")

        # Get the index for the t0 entry for each lightcurve MJD and use that
        # to subtract out the reference t0.
        t0 = np.asanyarray(results["t0"])
        t0_idx = np.array(results["lightcurve"]["mjd"].index)
        results["lightcurve.time_rel"] = results["lightcurve.mjd"] - t0[t0_idx]
