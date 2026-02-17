import matplotlib.pyplot as plt
import numpy as np
import pytest
from lightcurvelynx.utils.plotting import plot_bandflux_lightcurves, plot_flux_spectrogram, plot_lightcurves


def test_plot_lightcurves():
    """Test that we can plot light curves."""
    # Test minimal input
    fluxes = np.array([1.0, 2.0, 3.0])
    times = np.array([1.0, 2.0, 3.0])
    plot_lightcurves(fluxes, times)
    # ValueError if len(times) != len(fluxes)
    wrong_times = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        plot_lightcurves(fluxes, wrong_times)

    # ValueError if len(filters) != len(fluxes)
    wrong_filters = ["none"]
    with pytest.raises(ValueError):
        plot_lightcurves(fluxes, times, filters=wrong_filters)

    # ValueError if fluxerrs given and len(fluxerrs) != len(fluxes)
    wrong_fluxerrs = np.array([0.1, 0.2])
    with pytest.raises(ValueError):
        plot_lightcurves(fluxes, times, fluxerrs=wrong_fluxerrs)

    # Test with almost all inputs given:
    # - fluxerrs (same length as fluxes to pass the ValueError check)
    # - filters (same length as fluxes to pass the ValueError check)
    # - title
    fluxerrs = np.array([0.1, 0.2, 0.3])
    filters = np.array(["A", "B", "A"])
    title = "Test Title"
    plot_lightcurves(fluxes, times, fluxerrs=fluxerrs, filters=filters, title=title)

    # Plot in magnitudes.
    plot_lightcurves(fluxes, times, fluxerrs=fluxerrs, filters=filters, plot_magnitudes=True)

    # Test with all inputs given:
    # - ax (matplotlib axes object)
    # - figure (matplotlib figure object)
    fig, ax = plt.subplots()
    plot_lightcurves(fluxes, times, fluxerrs=fluxerrs, filters=filters, ax=ax, figure=fig, title=title)
    plt.close(fig)


def test_plot_bandflux_lightcurves():
    """Test that we can plot bandflux light curves."""
    # Test minimal input
    bandfluxes = np.array([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]])
    plot_bandflux_lightcurves(bandfluxes)

    # Test with all inputs given:
    # - times (np array of length T)
    # - ax (matplotlib axes object)
    # - figure (matplotlib figure object)
    # - title
    times = np.array([1.0, 2.0])
    fig, ax = plt.subplots()
    title = "Test Title"
    plot_bandflux_lightcurves(bandfluxes, times=times, ax=ax, figure=fig, title=title)
    plt.close(fig)


def test_plot_flux_spectrogram():
    """Test that we can plot a flux spectrogram."""
    # Test minimal input
    fluxes = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    plot_flux_spectrogram(fluxes)

    # Test with all inputs given:
    # - times (np array)
    # - ax (matplotlib axes object)
    # - figure (matplotlib figure object)
    # - title
    times = np.array([1.0, 2.0, 3.0, 4.0])
    fig, ax = plt.subplots()
    title = "Test Title"
    plot_flux_spectrogram(fluxes, times=times, ax=ax, figure=fig, title=title)
    plt.close(fig)
