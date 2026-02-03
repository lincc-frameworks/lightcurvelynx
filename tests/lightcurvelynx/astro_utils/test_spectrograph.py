import numpy as np
import pytest
from lightcurvelynx.astro_utils.spectrograph import SpectraPassbandGroup


def test_create_spectra_passband_group():
    """Test that we can create and query a SpectraPassbandGroup object."""
    sp_pbg = SpectraPassbandGroup(wave_start=3000, wave_end=11000, bin_width=5.0)
    assert sp_pbg.wave_start == 3000
    assert sp_pbg.wave_end == 11000
    assert sp_pbg.bin_width == 5.0
    assert sp_pbg.survey == "SpectraPassbandGroup"
    assert len(sp_pbg) == 1600  # (11000 - 3000) / 5 = 1600 bins
    assert np.array_equal(sp_pbg.waves, np.arange(3000 + 2.5, 11000, 5.0))

    l_val, h_val = sp_pbg.wave_bounds()
    assert l_val == 3000
    assert h_val == 11000

    # Check we can get the single "spectra" filter
    assert "spectra" in sp_pbg
    assert "any_other_filter" not in sp_pbg
    assert np.array_equal(sp_pbg.filters, ["spectra"])

    spectra_filter = sp_pbg["spectra"]
    assert isinstance(spectra_filter, SpectraPassbandGroup)
    assert spectra_filter is sp_pbg
    with pytest.raises(KeyError):
        _ = sp_pbg["any_other_filter"]

    assert str(sp_pbg) == "SpectraPassbandGroup (spectra) [3000A - 11000A]"

    # Two dimensional fluxes to bandfluxes
    values = np.random.random((10, len(sp_pbg)))
    bandfluxes = sp_pbg.fluxes_to_bandflux(values)
    assert np.allclose(bandfluxes, values)

    # Three dimensional fluxes to bandfluxes
    values_3d = np.random.random((4, 10, len(sp_pbg)))
    bandfluxes_3d = sp_pbg.fluxes_to_bandflux(values_3d)
    assert np.allclose(bandfluxes_3d, values_3d)


def test_spectra_passband_group_equals():
    """Test that we can compare two SpectraPassbandGroup objects for equality."""
    sp_pbg1 = SpectraPassbandGroup(wave_start=4000, wave_end=8000, bin_width=5.0)
    sp_pbg2 = SpectraPassbandGroup(wave_start=4000, wave_end=8000, bin_width=5.0)
    sp_pbg3 = SpectraPassbandGroup(wave_start=4000, wave_end=8000, bin_width=10.0)
    sp_pbg4 = SpectraPassbandGroup(wave_start=3000, wave_end=8000, bin_width=5.0)
    sp_pbg5 = SpectraPassbandGroup(wave_start=4000, wave_end=9000, bin_width=5.0)

    assert sp_pbg1 == sp_pbg2
    assert sp_pbg1 != sp_pbg3
    assert sp_pbg1 != sp_pbg4
    assert sp_pbg1 != sp_pbg5


def test_create_spectra_passband_group_fail():
    """Test that we fail to create a SpectraPassbandGroup object with invalid parameters."""
    with pytest.raises(ValueError):
        _ = SpectraPassbandGroup(wave_start=5000, wave_end=4000, bin_width=5.0)
    with pytest.raises(ValueError):
        _ = SpectraPassbandGroup(wave_start=4000, wave_end=5000, bin_width=-5.0)


def test_create_spectra_passband_group_with_scale():
    """Test that we can create and query a SpectraPassbandGroup object."""
    scale = np.array([0.5, 1.0, 1.0, 1.0, 0.8])
    sp_pbg = SpectraPassbandGroup(wave_start=4000, wave_end=5000, bin_width=200.0, scale=scale)
    assert np.allclose(sp_pbg.waves, np.array([4100.0, 4300.0, 4500.0, 4700.0, 4900.0]))
    assert sp_pbg.wave_start == 4000
    assert sp_pbg.wave_end == 5000
    assert sp_pbg.bin_width == 200.0

    # Two dimensional fluxes to bandfluxes
    input = np.array(
        [
            [10.0, 20.0, 30.0, 40.0, 50.0],
            [5.0, 15.0, 25.0, 35.0, 45.0],
        ]
    )
    expected = np.array(
        [
            [5.0, 20.0, 30.0, 40.0, 40.0],
            [2.5, 15.0, 25.0, 35.0, 36.0],
        ]
    )
    results = sp_pbg.fluxes_to_bandflux(input)
    assert np.allclose(results, expected)

    # Three dimensional fluxes to bandfluxes
    input = np.array(
        [
            [
                [10.0, 20.0, 30.0, 40.0, 50.0],
                [5.0, 15.0, 25.0, 35.0, 45.0],
            ],
            [
                [8.0, 18.0, 28.0, 38.0, 48.0],
                [4.0, 14.0, 24.0, 34.0, 44.0],
            ],
        ]
    )
    expected = np.array(
        [
            [
                [5.0, 20.0, 30.0, 40.0, 40.0],
                [2.5, 15.0, 25.0, 35.0, 36.0],
            ],
            [
                [4.0, 18.0, 28.0, 38.0, 38.4],
                [2.0, 14.0, 24.0, 34.0, 35.2],
            ],
        ]
    )
    results = sp_pbg.fluxes_to_bandflux(input)
    assert np.allclose(results, expected)

    # Test equality with scales.
    sp_pbg2 = SpectraPassbandGroup(wave_start=4000, wave_end=5000, bin_width=200.0, scale=scale)
    assert sp_pbg == sp_pbg2

    different_scale = np.array([0.5, 1.0, 1.0, 1.0, 0.9])
    sp_pbg3 = SpectraPassbandGroup(wave_start=4000, wave_end=5000, bin_width=200.0, scale=different_scale)
    assert sp_pbg != sp_pbg3

    # Test with a mismatched scale length.
    bad_scale = np.array([1.0, 0.8])
    with pytest.raises(ValueError):
        _ = SpectraPassbandGroup(wave_start=4000, wave_end=5000, bin_width=200.0, scale=bad_scale)
