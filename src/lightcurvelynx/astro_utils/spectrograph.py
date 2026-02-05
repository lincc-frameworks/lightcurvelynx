"""The Spectrograph object stores information about a spectrograph's bins
and provides methods to compute fluxes for each bin.
"""

import numpy as np


class Spectrograph:
    """Models all of the bins of a spectrograph, producing bandfluxes for each
    bin in the spectra. This class operates similarly to a PassbandGroup, but
    only contains a single "filter" named "spectra" that contains all of the bins.

    Attributes
    ----------
    waves : np.ndarray
        The wavelengths at the center of each bin in Angstroms.
    instrument : str
        The instrument name for the spectra passband group. Default is "Spectrograph".
    scale : np.ndarray
        The multiplicative factor to apply to each bin's flux to capture sensor
        sensitivity, etc. If None, we use 1.0 for all bins.
    wave_min : float
        The minimum wavelength of the spectra in Angstroms.
    wave_max : float
        The maximum wavelength of the spectra in Angstroms.
    """

    def __init__(
        self,
        waves: np.array,
        *,
        scale: np.ndarray | None = None,
        instrument: str | None = None,
    ):
        if np.any(np.diff(waves) <= 0):
            raise ValueError("waves must be in strictly increasing order.")
        self.waves = waves
        self.instrument = instrument if instrument is not None else "Spectrograph"

        self.wave_min = self.waves[0] - 0.5 * self.bin_width(0)
        self.wave_max = self.waves[-1] + 0.5 * self.bin_width(len(self.waves) - 1)

        # Scale is the multiplicative factor to apply to each bin's flux. If None, we use 1.0 for all bins.
        if scale is None:
            scale = np.ones(len(self.waves))
        elif len(scale) != len(self.waves):
            raise ValueError("Scale array must have the same length as the number of bins in the spectra.")
        self.scale = scale

    def __str__(self) -> str:
        """Return a string representation of the spectra filter."""
        return f"{self.instrument} (spectra) [{self.wave_min}A - {self.wave_max}A]"

    def __len__(self) -> int:
        return len(self.waves)

    def __eq__(self, other) -> bool:
        """Determine if two passbands have equal values for the processed tables."""
        if len(self.waves) != len(other.waves):
            return False
        if not np.allclose(self.waves, other.waves):
            return False
        if not np.allclose(self.scale, other.scale):
            return False
        return True

    @classmethod
    def from_regular_grid(cls, wave_start: float, wave_end: float, bin_width: float, **kwargs):
        """Create a Spectrograph with regularly spaced bins.

        Parameters
        ----------
        wave_start : float
            The starting wavelength of the spectra in Angstroms.
        wave_end : float
            The ending wavelength of the spectra in Angstroms.
        bin_width : float
            The bin size of the spectra in Angstroms.
        **kwargs
            Additional keyword arguments to pass to the Spectrograph constructor.

        Returns
        -------
        Spectrograph
            A Spectrograph object with regularly spaced bins.
        """
        if wave_end <= wave_start:
            raise ValueError("wave_end must be greater than wave_start.")
        if bin_width <= 0:
            raise ValueError("bin_width must be positive.")

        # We use the wavelength at the center of each bin.
        bin_centers = np.arange(wave_start + bin_width / 2, wave_end, bin_width)
        return cls(bin_centers, **kwargs)

    def bin_width(self, index):
        """Get the width of the bin at the given index.

        Parameters
        ----------
        index : int
            The index of the bin.

        Returns
        -------
        float
            The width of the bin in Angstroms.
        """
        if index < 0 or index >= len(self.waves):
            raise IndexError(f"Index {index} out of bounds for {len(self.waves)} bin widths.")
        elif index == 0:
            # The center of the bin is at waves[0], so we assume the lower bound is symmetric
            # about that point: 2.0 * (center_1 - center_0) / 2.0
            return self.waves[1] - self.waves[0]
        elif index == len(self.waves) - 1:
            # The center of the bin is at waves[-1], so we assume the upper bound is symmetric
            # about that point 2.0 * (center_N-1 - center_N-2) / 2.0
            return self.waves[-1] - self.waves[-2]
        else:
            # Half the distance to the neighboring bins on either side.
            return (self.waves[index + 1] - self.waves[index - 1]) / 2

    def wave_bounds(self):
        """Get the minimum and maximum wavelength for this spectra.

        Returns
        -------
        min_wave : float
            The minimum wavelength.
        max_wave : float
            The maximum wavelength.
        """
        return self.wave_min, self.wave_max

    def evaluate(
        self,
        flux_density_matrix: np.ndarray,
    ) -> np.ndarray:
        """Calculate the measured values for each bin in the spectrograph.

        Parameters
        ----------
        flux_density_matrix : np.ndarray
            A 2D or 3D array of flux densities. If the array is 2D it contains a single sample where
            the rows are the T times and columns are M wavelengths in Angstroms. If the array is 3D
            it contains S samples and the values are indexed as (sample_num, time, wavelength).

        Returns
        -------
        measured : np.ndarray
            A 2D or 3D array. If the flux_density_matrix contains a single sample (2D input) then
            the function returns a 2D matrix where each row is a time and each column is the measurement
            at the corresponding wavelength bin. Otherwise the function returns a size S x T x B array
            where each entry corresponds to the measured value for a given sample at a given time and
            wavelength bin.
        """
        if flux_density_matrix.size == 0:
            raise ValueError("Empty flux density matrix used.")  # pragma: no cover
        if len(flux_density_matrix.shape) == 2:
            return flux_density_matrix * self.scale[np.newaxis, :]
        elif len(flux_density_matrix.shape) == 3:
            return flux_density_matrix * self.scale[np.newaxis, np.newaxis, :]
        else:
            raise ValueError("Invalid flux density matrix. Must be 2 or 3-dimensional.")  # pragma: no cover
