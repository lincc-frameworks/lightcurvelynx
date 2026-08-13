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
    waves_min : np.ndarray
        The start of each wavelength bin in Angstroms.
    waves_max : np.ndarray
        The end of each wavelength bin in Angstroms.
    num_bins : int
        The number of bins of the spectrograph.
    bin_widths : np.ndarray
        The width of each wavelength bin in Angstroms.
    query_waves : np.ndarray
        The points at which to evaluate the flux density of the spectral model in Angstroms.
    instrument : str
        The instrument name for the spectrograph. Default is "Spectrograph".
    scale : np.ndarray
        The multiplicative factor to apply to each bin's flux to capture sensor
        sensitivity, etc. If None, we use 1.0 for all bins.
    """

    def __init__(
        self,
        waves_min,
        waves_max,
        *,
        scale=None,
        instrument: str | None = None,
    ):
        # Check that the input arrays are valid and convert them to numpy arrays.
        self.waves_min = np.asarray(waves_min, dtype=float)
        self.waves_max = np.asarray(waves_max, dtype=float)
        self.num_bins = len(self.waves_min)
        if len(self.waves_max) != self.num_bins:
            raise ValueError("waves_min and waves_max must have the same length.")
        if np.any(np.diff(waves_min) <= 0):
            raise ValueError("waves_min must be in strictly increasing order.")
        if np.any(np.diff(waves_max) <= 0):
            raise ValueError("waves_max must be in strictly increasing order.")
        if np.any(waves_max <= waves_min):
            raise ValueError(
                "Each element of waves_max must be greater than the corresponding element of waves_min."
            )
        self.query_waves = (self.waves_min + self.waves_max) / 2

        # Compute the width of each bin and check that none of the bins have negative width.
        self.bin_widths = self.waves_max - self.waves_min
        if np.any(self.bin_widths <= 0):
            raise ValueError("Bins must have positive width.")

        # Scale is the multiplicative factor to apply to each bin's flux. If None, we use 1.0 for all bins.
        if scale is None:
            scale = np.ones(len(self.waves_min))
        elif len(scale) != len(self.waves_min):
            raise ValueError("Scale array must have the same length as the number of bins in the spectra.")
        self.scale = np.asarray(scale)

        # Save the other spectrograph properties if provided.
        self.instrument = instrument if instrument is not None else "Spectrograph"

    def __str__(self) -> str:
        """Return a string representation of the spectra filter."""
        return f"{self.instrument} (spectra) [{self.waves_min[0]}A - {self.waves_max[-1]}A]"

    def __len__(self) -> int:
        return self.num_bins

    def __eq__(self, other) -> bool:
        """Determine if two spectrographs have equal values for their internal data."""
        if self.num_bins != other.num_bins:
            return False
        if not np.allclose(self.waves_min, other.waves_min):
            return False
        if not np.allclose(self.waves_max, other.waves_max):
            return False
        if not np.allclose(self.bin_widths, other.bin_widths):  # pragma: no cover
            return False
        if self.instrument != other.instrument:  # pragma: no cover
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
        waves_min = bin_centers - bin_width / 2
        waves_max = bin_centers + bin_width / 2
        return cls(waves_min, waves_max, **kwargs)

    @classmethod
    def from_midpoints(cls, wave_midpoints, bin_width, **kwargs):
        """Create a Spectrograph from the midpoints and widths of the bins.

        Parameters
        ----------
        wave_midpoints : array-like
            The midpoints of each wavelength bin in Angstroms.
        bin_width : float or array-like
            The width of each wavelength bin in Angstroms.
        **kwargs
            Additional keyword arguments to pass to the Spectrograph constructor.

        Returns
        -------
        Spectrograph
            A Spectrograph object with bins defined by the midpoints and widths.
        """
        wave_midpoints = np.asarray(wave_midpoints, dtype=float)
        if bin_width is None:
            raise ValueError("bin_width must be provided.")
        if np.isscalar(bin_width):
            bin_widths = np.full(wave_midpoints.shape, float(bin_width), dtype=float)
        else:
            bin_widths = np.asarray(bin_width, dtype=float)

        if np.any(bin_widths <= 0):
            raise ValueError("All bin widths must be positive.")
        if len(wave_midpoints) != len(bin_widths):
            raise ValueError("wave_midpoints and bin_widths must have the same length.")

        waves_min = wave_midpoints - bin_widths / 2
        waves_max = wave_midpoints + bin_widths / 2
        return cls(waves_min, waves_max, **kwargs)

    def wave_bounds(self):
        """Get the minimum and maximum wavelength bin boundaries for this spectra.

        Returns
        -------
        min_wave : float
            The minimum wavelength.
        max_wave : float
            The maximum wavelength.
        """
        return self.waves_min[0], self.waves_max[-1]

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
        measured_flux : np.ndarray
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
