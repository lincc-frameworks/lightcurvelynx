"""The Passband and PassbandGroup objects store information about the filters used
to convert flux densities (as a function of wavelength) to bandfluxes. They also provide
methods for loading and manipulating passband data.
"""

import numpy as np


class SpectraPassbandGroup:
    """Models all of the bins of a spectrograph, producing bandfluxes for each
    bin in the spectra. This class operates similarly to a PassbandGroup, but
    only contains a single "filter" named "spectra" that contains all of the bins.

    Attributes
    ----------
    wave_start : float
        The starting wavelength of the spectra in Angstroms.
    wave_end : float
        The ending wavelength of the spectra in Angstroms.
    bin_width : float
        The bin size of the spectra in Angstroms.
    survey : str
        The survey name for the spectra passband group. Default is "SpectraPassbandGroup".
    waves : np.ndarray
        The wavelengths at the center of each bin in Angstroms.
    scale : np.ndarray
        The multiplicative factor to apply to each bin's flux to capture sensor
        sensitivity, etc. If None, we use 1.0 for all bins.
    """

    def __init__(
        self,
        wave_start: float,
        wave_end: float,
        bin_width: float,
        *,
        scale: np.ndarray | None = None,
        survey: str = "SpectraPassbandGroup",
    ):
        if wave_end <= wave_start:
            raise ValueError("wave_end must be greater than wave_start.")
        if bin_width <= 0:
            raise ValueError("bin_width must be positive.")
        self.wave_start = wave_start
        self.wave_end = wave_end
        self.bin_width = bin_width
        self.survey = survey

        # We use the wavelength at the center of each bin.
        self.waves = np.arange(wave_start + bin_width / 2, wave_end, bin_width)

        # Scale is the multiplicative factor to apply to each bin's flux. If None, we use 1.0 for all bins.
        if scale is None:
            scale = np.ones(len(self.waves))
        elif len(scale) != len(self.waves):
            print(scale)
            print(self.waves)
            raise ValueError("Scale array must have the same length as the number of bins in the spectra.")
        self.scale = scale

    def __str__(self) -> str:
        """Return a string representation of the spectra filter."""
        return f"{self.survey} (spectra) [{self.wave_start}A - {self.wave_end}A]"

    def __len__(self) -> int:
        return len(self.waves)

    def __eq__(self, other) -> bool:
        """Determine if two passbands have equal values for the processed tables."""
        if self.wave_start != other.wave_start:
            return False
        if self.wave_end != other.wave_end:
            return False
        if self.bin_width != other.bin_width:
            return False
        if not np.allclose(self.scale, other.scale):
            return False
        return True

    def __getitem__(self, key):
        """Return the passband for the given filter name or full name."""
        if key == "spectra":
            return self
        raise KeyError(f"Unknown passband {key} for SpectraFilter.")

    def __contains__(self, key):
        return key == "spectra"

    @property
    def filters(self) -> list:
        """Return a list of filter names in the passband group."""
        return ["spectra"]

    def wave_bounds(self):
        """Get the minimum and maximum wavelength for this spectra.

        Returns
        -------
        min_wave : float
            The minimum wavelength.
        max_wave : float
            The maximum wavelength.
        """
        return self.wave_start, self.wave_end

    def fluxes_to_bandflux(
        self,
        flux_density_matrix: np.ndarray,
    ) -> np.ndarray:
        """Calculate the 'bandflux' for each bin in the spectra.

        Parameters
        ----------
        flux_density_matrix : np.ndarray
            A 2D or 3D array of flux densities. If the array is 2D it contains a single sample where
            the rows are the T times and columns are M wavelengths in Angstroms. If the array is 3D
            it contains S samples and the values are indexed as (sample_num, time, wavelength).

        Returns
        -------
        bandfluxes : np.ndarray
            A 2D or 3D array. If the flux_density_matrix contains a single sample (2D input) then
            the function returns a 2D matrix where each row is a time and each column is the bandflux
            at the corresponding wavelength bin. Otherwise the function returns a size S x T x B array where
            each entry corresponds to the value for a given sample at a given time and wavelength bin.
        """
        if flux_density_matrix.size == 0:
            raise ValueError("Empty flux density matrix used.")
        if len(flux_density_matrix.shape) == 2:
            return flux_density_matrix * self.scale[np.newaxis, :]
        elif len(flux_density_matrix.shape) == 3:
            return flux_density_matrix * self.scale[np.newaxis, np.newaxis, :]
        else:
            raise ValueError("Invalid flux density matrix. Must be 2 or 3-dimensional.")
