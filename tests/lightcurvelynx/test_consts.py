from numpy.testing import assert_allclose

from lightcurvelynx import consts


def test_parsec_to_cm():
    """Check PARSEC_TO_CM constant"""
    assert_allclose(consts.PARSEC_TO_CM, 3.08567758149137e18)


def test_gauss_eff_area2fwhm_sq():
    """Test GAUSS_EFF_AREA2FWHM_SQ to be close to 2.266

    Find 2.266 here:
    https://smtn-002.lsst.io/v/OPSIM-1171/index.html
    """
    assert_allclose(consts.GAUSS_EFF_AREA2FWHM_SQ, 2.266, atol=1e-3)


def test_angstrom_to_cm():
    """Check ANGSTROM_TO_CM constant"""
    assert_allclose(consts.ANGSTROM_TO_CM, 1e-8)


def test_cgs_fnu_unit_to_njy():
    """Check CGS_FNU_UNIT_TO_NJY constant"""
    assert_allclose(consts.CGS_FNU_UNIT_TO_NJY, 1e32)
