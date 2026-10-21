import matplotlib.pyplot as plt
import numpy as np
import pytest

from lightcurvelynx.utils.plotting import (
    _build_colormap,
    _build_line_map,
    _build_symbol_map,
    plot_bandflux_lightcurves,
    plot_flux_spectrogram,
    plot_lightcurves,
)


def test_build_symbol_map():
    """Test that we can build a symbol map."""
    # Check the LSST recommended filters.
    filters = ["u", "g", "r", "i", "z", "y"]
    symbol_map = _build_symbol_map(filters)
    assert isinstance(symbol_map, dict)
    assert symbol_map["u"] == "o"
    assert symbol_map["g"] == "^"
    assert symbol_map["r"] == "v"
    assert symbol_map["i"] == "s"
    assert symbol_map["z"] == "*"
    assert symbol_map["y"] == "p"

    # Check an extended list of filters.
    filters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n"]
    expected = ["o", "^", "v", "s", "*", "p", "h", "H", "D", "d", "o", "^", "v", "s"]
    symbol_map = _build_symbol_map(filters)
    assert isinstance(symbol_map, dict)
    assert set(symbol_map.keys()) == set(filters)
    for i, f in enumerate(filters):
        assert symbol_map[f] == expected[i]


def test_build_line_map():
    """Test that we can build a line map."""
    # Check the LSST filters
    filters = ["u", "g", "r", "i", "z", "y"]
    line_map = _build_line_map(filters)
    assert isinstance(line_map, dict)
    assert set(line_map.keys()) == set(filters)
    for f in filters:
        assert isinstance(line_map[f], (str, tuple))

    # Check an extended list of filters.
    filters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n"]
    line_map = _build_line_map(filters)
    assert isinstance(line_map, dict)
    assert set(line_map.keys()) == set(filters)
    for f in filters:
        assert isinstance(line_map[f], (str, tuple))


def test_build_colormap():
    """Test that we can build a colormap."""
    for n_filters in [5, 10, 20, 50]:
        filters = [f"Filter {i}" for i in range(n_filters)]
        colormap = _build_colormap(filters)
        assert isinstance(colormap, dict)
        assert set(colormap.keys()) == set(filters)
        for color in colormap.values():
            assert isinstance(color, tuple) and len(color) == 4  # RGBA color

    # Test red and green only.
    filters = ["g", "r"]
    colormap = _build_colormap(filters)
    assert isinstance(colormap, dict)
    assert set(colormap.keys()) == set(filters)
    assert colormap["g"] == "green"
    assert colormap["r"] == "red"

    # Test the LSST standard filters.
    filters = ["u", "g", "r", "i", "z", "y"]
    colormap = _build_colormap(filters)
    assert isinstance(colormap, dict)
    assert set(colormap.keys()) == set(filters)
    assert colormap["u"] == "#1600ea"
    assert colormap["g"] == "#31de1f"
    assert colormap["r"] == "#b52626"
    assert colormap["i"] == "#370201"
    assert colormap["z"] == "#ba52ff"
    assert colormap["y"] == "#61a2b3"


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

    # Plot in magnitudes with an underlying model.
    underlying_model = {
        "A": np.array([1.5, 2.5, 3.5]),
        "B": np.array([2.0, 3.0, 4.0]),
        "times": np.array([1.0, 2.0, 3.0]),
    }
    plot_lightcurves(
        fluxes,
        times,
        fluxerrs=fluxerrs,
        filters=filters,
        plot_magnitudes=True,
        underlying_model=underlying_model,
    )

    # Test with all inputs given:
    # - ax (matplotlib axes object)
    # - figure (matplotlib figure object)
    fig, ax = plt.subplots()
    plot_lightcurves(fluxes, times, fluxerrs=fluxerrs, filters=filters, ax=ax, figure=fig, title=title)

    # Close all the open figures.
    plt.close("all")


def test_plot_lightcurves_many_filters():
    """Test that we can plot light curves with many filters."""
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

    # Plot in magnitudes with an underlying model.
    underlying_model = {
        "A": np.array([1.5, 2.5, 3.5]),
        "B": np.array([2.0, 3.0, 4.0]),
        "times": np.array([1.0, 2.0, 3.0]),
    }
    plot_lightcurves(
        fluxes,
        times,
        fluxerrs=fluxerrs,
        filters=filters,
        plot_magnitudes=True,
        underlying_model=underlying_model,
    )

    # Test with all inputs given:
    # - ax (matplotlib axes object)
    # - figure (matplotlib figure object)
    fig, ax = plt.subplots()
    plot_lightcurves(fluxes, times, fluxerrs=fluxerrs, filters=filters, ax=ax, figure=fig, title=title)

    # Close all the open figures.
    plt.close("all")


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

    # Close all the open figures.
    plt.close("all")


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

    # Close all the open figures.
    plt.close("all")
