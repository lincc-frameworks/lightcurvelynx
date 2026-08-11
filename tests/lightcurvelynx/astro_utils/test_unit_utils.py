import numpy as np
import pytest
from astropy import units as u
from lightcurvelynx.astro_utils.unit_utils import flam_to_fnu, fnu_to_flam


def test_flam_to_fnu():
    """Test flam to fnu conversion:
    fnu = flam * lam**2 / c
    """
    c = 299_792_458.0 * 1.0e10  # in AA/s
    wave = [100.0, 1.0e5]
    flam = [c * 1.0e-4, 1.0e5]
    fnu = flam_to_fnu(
        flam,
        wave,
        wave_unit=u.AA,
        flam_unit=u.erg / u.second / u.cm**2 / u.AA,
        fnu_unit=u.erg / u.second / u.cm**2 / u.Hz,
    )
    assert np.allclose(fnu, [1.0, 1.0e15 / c])

    # Include some tests using values from an older implementation to ensure that nothing has changed.
    flam = [697.713, 697.713, 697.713, 1013.715, 1013.715, 1013.715, 489.0754, 489.0754, 489.0754]
    wave = [5000, 5010, 5020, 5000, 5010, 5020, 5000, 5010, 5020]
    fnu = flam_to_fnu(
        flam,
        wave,
        wave_unit=u.AA,
        flam_unit=u.erg / u.second / u.cm**2 / u.AA,
        fnu_unit=u.nJy,
    )

    expected = [5.82e23, 5.84e23, 5.86e23, 8.45e23, 8.49e23, 8.52e23, 4.08e23, 4.09e23, 4.11e23]
    assert np.allclose(fnu, expected, rtol=1e-2)


def test_flam_to_fnu_matrix():
    """Test flam to fnu conversion when dealing with a matrix of observations."""
    c = 299_792_458.0 * 1.0e10  # in AA/s

    # Both wave and flam are matrices
    fnu = flam_to_fnu(
        [[c * 1.0e-4, 1.0e5], [1000.0, 2000.0], [c * 1.0e-4, c * 1.0e-4]],
        [[100.0, 1.0e5], [1000.0, 1.0e4], [1000.0, 1000.0]],
        wave_unit=u.AA,
        flam_unit=u.erg / u.second / u.cm**2 / u.AA,
        fnu_unit=u.erg / u.second / u.cm**2 / u.Hz,
    )

    # Compute expected from fnu = flam * lam**2 / c
    expected = np.array([[1.0, 1.0e15 / c], [1.0e9 / c, 2.0e11 / c], [100.0, 100.0]])
    assert np.allclose(fnu, expected)

    # Use a single vector of wavelengths.
    fnu = flam_to_fnu(
        [[c * 1.0e-4, 1.0e5], [1000.0, 2000.0], [c * 1.0e-4, c * 1.0e-4]],
        [100.0, 1.0e5],
        wave_unit=u.AA,
        flam_unit=u.erg / u.second / u.cm**2 / u.AA,
        fnu_unit=u.erg / u.second / u.cm**2 / u.Hz,
    )
    expected = np.array([[1.0, 1.0e15 / c], [1.0e7 / c, 2.0e13 / c], [1.0, 1.0e6]])
    assert np.allclose(fnu, expected)

    # Include some tests using values from an older implementation to ensure that nothing has changed.
    flam = [
        [148.55, 148.55, 148.55, 148.55],
        [414.47, 414.47, 414.47, 414.47],
        [491.29, 491.29, 491.29, 491.29],
    ]
    wave = [
        [6000, 6050, 7000, 7025],
        [6000, 6050, 7000, 7025],
        [6000, 6050, 7000, 7025],
    ]
    fnu = flam_to_fnu(
        flam,
        wave,
        wave_unit=u.AA,
        flam_unit=u.erg / u.second / u.cm**2 / u.AA,
        fnu_unit=u.nJy,
    )

    expected = [
        [1.78e23, 1.81e23, 2.43e23, 2.45e23],
        [4.98e23, 5.06e23, 6.77e23, 6.82e23],
        [5.90e23, 6.00e23, 8.03e23, 8.09e23],
    ]
    assert np.allclose(fnu, expected, rtol=1e-2)

    # Fail if the dimensions are not compatible.
    with pytest.raises(ValueError):
        _ = flam_to_fnu(
            [[c * 1.0e-4, 1.0e5], [1000.0, 2000.0], [c * 1.0e-4, c * 1.0e-4]],
            [[100.0, 1.0e5], [1000.0, 1.0e4]],
            wave_unit=u.AA,
            flam_unit=u.erg / u.second / u.cm**2 / u.AA,
            fnu_unit=u.erg / u.second / u.cm**2 / u.Hz,
        )

    with pytest.raises(ValueError):
        _ = flam_to_fnu(
            [[c * 1.0e-4, 1.0e5], [1000.0, 2000.0], [c * 1.0e-4, c * 1.0e-4]],
            [100.0, 200.0, 300.0],
            wave_unit=u.AA,
            flam_unit=u.erg / u.second / u.cm**2 / u.AA,
            fnu_unit=u.erg / u.second / u.cm**2 / u.Hz,
        )


def test_fnu_to_flam():
    """Test fnu to flam conversion using values from an older implementation to ensure
    that nothing has changed.
    """
    fnu = [5.82e23, 5.84e23, 5.86e23, 8.45e23, 8.49e23, 8.52e23, 4.08e23, 4.09e23, 4.11e23]
    wave = [5000, 5010, 5020, 5000, 5010, 5020, 5000, 5010, 5020]
    flam = fnu_to_flam(
        fnu,
        wave,
        wave_unit=u.AA,
        flam_unit=u.erg / u.second / u.cm**2 / u.AA,
        fnu_unit=u.nJy,
    )

    expected = [697.713, 697.713, 697.713, 1013.715, 1013.715, 1013.715, 489.0754, 489.0754, 489.0754]
    assert np.allclose(flam, expected, rtol=1e-2)

    # Test as a matrix.
    fnu = [
        [1.78e23, 1.81e23, 2.43e23, 2.45e23],
        [4.98e23, 5.06e23, 6.77e23, 6.82e23],
        [5.90e23, 6.00e23, 8.03e23, 8.09e23],
    ]
    wave = [
        [6000, 6050, 7000, 7025],
        [6000, 6050, 7000, 7025],
        [6000, 6050, 7000, 7025],
    ]
    flam = fnu_to_flam(
        fnu,
        wave,
        wave_unit=u.AA,
        flam_unit=u.erg / u.second / u.cm**2 / u.AA,
        fnu_unit=u.nJy,
    )

    expected = [
        [148.55, 148.55, 148.55, 148.55],
        [414.47, 414.47, 414.47, 414.47],
        [491.29, 491.29, 491.29, 491.29],
    ]
    assert np.allclose(flam, expected, rtol=1e-2)

    # The code works with a single vector of wavelengths.
    flam2 = fnu_to_flam(
        fnu,
        [6000, 6050, 7000, 7025],  # Wavelengths are a single vector, not a matrix.
        wave_unit=u.AA,
        flam_unit=u.erg / u.second / u.cm**2 / u.AA,
        fnu_unit=u.nJy,
    )
    assert np.allclose(flam2, expected, rtol=1e-2)

    # We fail with incompatible dimensions.
    with pytest.raises(ValueError):
        _ = fnu_to_flam(
            fnu,
            [[6000, 6050, 7000, 7025], [6000, 6050, 7000, 7025]],
            wave_unit=u.AA,
            flam_unit=u.erg / u.second / u.cm**2 / u.AA,
            fnu_unit=u.nJy,
        )


def test_flam_to_fnu_to_flam():
    """Test that flam_to_fnu(fnu_to_flam) is the identity."""
    rng = np.random.default_rng(None)
    n = 100
    waves = rng.uniform(low=100.0, high=1.0e5, size=n)
    flam0 = rng.lognormal(mean=0.0, sigma=1.0, size=n)
    fnu = flam_to_fnu(
        flam0,
        waves,
        wave_unit=u.AA,
        flam_unit=u.erg / u.second / u.cm**2 / u.AA,
        fnu_unit=u.nJy,
    )
    flam1 = fnu_to_flam(
        fnu,
        waves,
        wave_unit=u.AA,
        flam_unit=u.erg / u.second / u.cm**2 / u.AA,
        fnu_unit=u.nJy,
    )
    np.testing.assert_allclose(flam0, flam1)

    # We throw an error on mismatched dimensions.
    with pytest.raises(ValueError):
        _ = fnu_to_flam(
            fnu,
            waves[1:],
            wave_unit=u.AA,
            flam_unit=u.erg / u.second / u.cm**2 / u.AA,
            fnu_unit=u.nJy,
        )
