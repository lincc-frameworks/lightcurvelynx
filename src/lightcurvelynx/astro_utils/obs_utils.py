import numpy as np


def phot_eff_function(snr):
    """
    Photometric detection efficiency as a simple step function of snr.

    Parameters
    ----------
    snr: list or numpy.ndarray
        Signal to noise ratio of a list of observations.

    Returns
    -------
    eff: list or numpy.ndarray
        The photometric detection efficiency given snr.
    """

    snr = np.asarray(snr)
    eff = np.where(snr > 5, 1.0, 0.0)

    return eff


def spec_eff_function(peak_imag):
    """
    Spectroscopic follow-up efficiency as a function of peak i band magnitude.

    Parameters
    ----------
    peak_imag: list or numpy.ndarray
        Peak magnitude in i band.

    Returns
    -------
    eff: list or numpy.ndarray
        The spectroscopic efficiency given peak i band magnitude.
        Based on Equation (17) in Kessler et al. 2019
        s0, s1, s2 are fitted using data from Figure 4 in Kessler et al. 2019
    """

    s0 = 1.0
    s1 = 2.36
    s2 = 51.9

    peak_imag = np.asarray(peak_imag)
    eff = s0 * np.power((1.0 + np.exp(s1 * peak_imag - s2)), -1)

    return eff
