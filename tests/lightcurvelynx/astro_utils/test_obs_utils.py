import numpy as np

from lightcurvelynx.astro_utils.obs_utils import phot_eff_function, spec_eff_function


def test_phot_eff_function():
    """
    test that the phot_eff_function returns correct values.
    """

    snr = [1.0, 3.0, 10.0, 100.0]
    eff = phot_eff_function(snr)
    expected_eff = [0.0, 0.0, 1.0, 1.0]

    np.testing.assert_allclose(eff, expected_eff)


def test_spec_eff_function():
    """
    test the spec efficiency function using data extracted from Figure 4 in Kessler et al. 2019
    """

    imags = [20.01, 21.09, 21.67, 22.54]
    expected_eff = [1.0, 0.9, 0.67, 0.2]
    eff = spec_eff_function(imags)

    np.testing.assert_allclose(eff, expected_eff, atol=0.02)
