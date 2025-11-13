"""The ColorLaw used by SALT models as defined by in (Guy J., 2007)

It is adapted from sncosmo's SALT2ColorLaw class (but implemented in JAX):
https://github.com/sncosmo/sncosmo/blob/v2.10.1/sncosmo/salt2utils.pyx
"""

import numpy as np
from citation_compass import CiteClass

# Constants used in SALT2ColorLaw computations (from:
# https://github.com/sncosmo/sncosmo/blob/v2.10.1/sncosmo/salt2utils.pyx)
_SALT2CL_B = 4302.57  # B-band-ish wavelength
_SALT2CL_V = 5428.55  # V-band-ish wavelength
_WAVESCALE = 1.0 / (_SALT2CL_V - _SALT2CL_B)


class SALT2ColorLaw(CiteClass):
    """An object that applies the color law to the given wavelengths.

    References
    ----------
    sncosmo - https://zenodo.org/records/14714968

    Parameters
    ----------
    wave_min : float
        The minimum wavelength (in angstroms)
    wave_max : float
        The maximum wavelength (in angstroms)
    coeffs : list, numpy array, or jax array
        The <= 6 coefficients of the polynomial to use.

    Attributes
    ----------
    coeffs : JAX array
        The final 7 coefficients to use (based on the <= 6 provided in the parameters).
    scaled_wave_min : float
        The minimum wavelength shifted and scaled.
    scaled_wave_max : float
        The maximum wavelength shifted and scaled.
    value_at_min : float
        The value of the polynomial at wave_min.
    value_at_max : float
        The value of the polynomial at wave_max.
    _exponents : JAX array
        A precomputed array of the exponents to use.
    """

    def __init__(self, wave_min, wave_max, coeffs):
        try:
            import jax.numpy as jnp
        except ImportError as err:  # pragma: no cover
            raise ImportError(
                "JAX is required to use the SALT2ColorLaw class, please "
                "install with `pip install jax` or `conda install conda-forge::jax`"
            ) from err

        # Create the internal coefficient array. The new first entry is 1.0 minus the
        # sum of the given entries. The first six given entries are then listed.
        coeffs = np.array(coeffs)
        num_coeffs = min(len(coeffs), 6)

        padded_coeffs = np.zeros(7)
        padded_coeffs[1 : (num_coeffs + 1)] = np.array(coeffs)[0:num_coeffs]
        padded_coeffs[0] = 1.0 - np.sum(padded_coeffs)
        self.coeffs = jnp.asarray(padded_coeffs)

        # Compute the bounds for the wavelengths and the value of the polynomial at both bounds.
        self.scaled_wave_min = (wave_min - _SALT2CL_B) * _WAVESCALE
        self.scaled_wave_max = (wave_max - _SALT2CL_B) * _WAVESCALE

        self.exponents = jnp.arange(1, 8, 1)
        self.value_at_min = jnp.sum(self.coeffs * jnp.power(self.scaled_wave_min, self.exponents))
        self.value_at_max = jnp.sum(self.coeffs * jnp.power(self.scaled_wave_max, self.exponents))

        # Precompute the polynomials derivative at the min and max wavelength.
        dcoeffs = jnp.arange(2, 8, 1) * self.coeffs[1:7]
        dexponents = jnp.arange(1, 7, 1)
        self.deriv_at_min = jnp.sum(dcoeffs * jnp.power(self.scaled_wave_min, dexponents)) + self.coeffs[0]
        self.deriv_at_max = jnp.sum(dcoeffs * jnp.power(self.scaled_wave_max, dexponents)) + self.coeffs[0]

    @classmethod
    def from_file(cls, filename):
        """Create the SALT2ColorLaw object from data in a file.

        Parameters
        ----------
        filename : str
            The name of the file to load.
        """
        with open(filename, mode="r") as f:
            data = f.read().split()

        # The first line holds the number of coefficients N and the next N lines
        # each hold a single coefficient.
        num_coeff = int(data[0])
        coeffs = np.array(data[1 : (1 + num_coeff)], dtype=float)

        # The rest of the lines (if any) are meta-data with a label and value on each line.
        wave_min = 3000.0
        wave_max = 7000.0
        for i in range(1 + num_coeff, len(data), 2):
            if "min_lambda" in data[i]:
                wave_min = float(data[i + 1])
            elif "max_lambda" in data[i]:
                wave_max = float(data[i + 1])
            elif "version" in data[i]:
                version = int(data[i + 1])
                if version != 1:
                    raise RuntimeError(f"Unsupported version {version}.")

        return SALT2ColorLaw(wave_min, wave_max, coeffs)

    def apply(self, wavelengths):
        """Apply the color law to the given wavelengths.

        Parameters
        ----------
        wavelengths : array
            The wavelengths in angstroms.
        """
        try:
            import jax.numpy as jnp
        except ImportError as err:  # pragma: no cover
            raise ImportError(
                "JAX is required to use the SALT2ColorLaw class, please "
                "install with `pip install jax` or `conda install conda-forge::jax`"
            ) from err

        num_waves = len(wavelengths)
        shifted_wave = (jnp.asarray(wavelengths) - _SALT2CL_B) * _WAVESCALE

        # Compute the three cases of interest.
        # 1) If the shifted wave is past the lower bound, extrapolate a value based on
        #    the value and derivative at that bound.
        below = self.value_at_min + self.deriv_at_min * (shifted_wave - self.scaled_wave_min)
        # 2) If the shifted wave is past the upper bound, extrapolate a value based on
        #    the value and derivative at that bound.
        above = self.value_at_max + self.deriv_at_max * (shifted_wave - self.scaled_wave_max)
        # 3) If the shifted value is in the middle, use the polynomial.
        wave_T = jnp.reshape(shifted_wave, (num_waves, 1))
        coeffs_all = jnp.tile(self.coeffs, (num_waves, 1))
        middle = jnp.sum((coeffs_all * jnp.power(wave_T, self.exponents)).T, axis=0)

        result = -jnp.where(
            shifted_wave < self.scaled_wave_min,
            below,
            jnp.where(
                shifted_wave > self.scaled_wave_max,
                above,
                middle,
            ),
        )
        return result
