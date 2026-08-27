"""The Spectrograph object stores information about a spectrograph's bins
and provides methods to compute fluxes for each bin.
"""

import numpy as np
from astropy import units as u

from lightcurvelynx.astro_utils.unit_utils import fnu_to_flam
import scipy

def gaussian_integral(nsig_low, nsig_high):
    """
    Compute the integral of a Gaussian between two limits in units of sigma.
    """
    return 0.5 * (scipy.special.erf(nsig_high / np.sqrt(2.0)) - scipy.special.erf(nsig_low / np.sqrt(2.0)))

class Spectrograph:
    """Models all of the bins of a spectrograph, producing bandfluxes for each
    bin in the spectra. This class operates similarly to a PassbandGroup, but
    only contains a single "filter" named "spectra" that contains all of the bins.

    Note
    ----
    This implementation requires the spectrograph to have non-overlapping bins
    that are provided in order of increasing wavelength.

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
        The points at which to evaluate the flux density of the spectral model in Angstroms. By
        default this is the midpoint of each bin (when `max_wave_step` is None or large enough
        to fully cover each bin).
    instrument : str
        The instrument name for the spectrograph. Default is "Spectrograph".
    scale : float | np.ndarray | None
        The multiplicative factor to apply to each bin's flux to capture sensor
        sensitivity, etc. If None, no scaling is applied.
    wavelength_resolution : np.ndarray
        The Gaussian sigma wavelength resolution for each bin in Angstroms.
    """

    def __init__(
        self,
        waves_min,
        waves_max,
        *,
        wavelength_resolution=None,
        instrument: str | None = None,
        scale=None,
        max_wave_step: float | None = None,
        oversample_factor: int = 10,
    ):
        """Initialize the Spectrograph object.

        Parameters
        ----------
        waves_min : array-like
            The start of each wavelength bin in Angstroms in order of increasing wavelength.
        waves_max : array-like
            The end of each wavelength bin in Angstroms in order of increasing wavelength.
        instrument : str, optional
            The instrument name for the spectrograph. Default is "Spectrograph".
        max_wave_step : float, optional
            The maximum allowed step size between wavelength points evaluated within each bin
            that are evaluated while computing the bin's flux. The smaller this value, the more
            accurate and expensive the integration. If None, a single sample per bin is used.
            Default: None
        oversample_factor: int, optional
            If max_wave_step is not provided, divide each bin into this many sub-bins to evaluate 
            the flux density of the object at a resolution higher than the spectrograph's.
            Default: 10
        scale : float | array-like, optional
            The multiplicative factor to apply to each bin's flux. If None, no additional scaling
            is applied.
            Default: None
        wavelength_resolution : np.ndarray
            The Gaussian sigma wavelength resolution for each bin in Angstroms.
        """
        # Check that the input arrays are valid and convert them to numpy arrays.
        self.waves_min = np.asarray(waves_min, dtype=float)
        self.waves_max = np.asarray(waves_max, dtype=float)
        self.num_bins = len(self.waves_min)

        oversample_factor = 1 if wavelength_resolution is None else oversample_factor
        self.wavelength_resolution = np.asarray(wavelength_resolution, dtype=float) \
            if wavelength_resolution is not None else np.zeros(self.num_bins, dtype=float)
        
        if self.num_bins <= 0:  # pragma: no cover
            raise ValueError("Spectrograph must have at least one bin.")
        if len(self.waves_max) != self.num_bins:
            raise ValueError("waves_min and waves_max must have the same length.")
        if np.any(self.waves_min[1:] < self.waves_max[:-1]):
            raise ValueError("Wavelength bins must be non-overlapping and in increasing order.")

        # Compute the width of each bin and check that none of the bins have negative width.
        self.bin_widths = self.waves_max - self.waves_min
        if np.any(self.bin_widths <= 0):
            raise ValueError("Bins must have positive width.")

        # check the oversample_factor
        if type(oversample_factor) is not int or oversample_factor < 1:
            raise ValueError(f"oversample_factor must be a positive integer, got {oversample_factor}.")

        

        # Compute the query wavelengths at which to evaluate the flux density of the object.
        # First, we pad the bins to account for smearing later.
        # Then, we evaluate at 10x the bin resolution.
        # However, if max_wave_step is provided AND
        # we need to split at least one bin, we will add multiple points per bin (evenly space
        # throughout the bin) until the maximum gap is LESS than max_wave_step.
        self._wave_to_bin_map = None
        padded_min, padded_max, padded_resolution, is_padding = self._compute_padded_bins()
        self.padded_min = padded_min
        self.padded_max = padded_max
        self.padded_widths = padded_max - padded_min
        self.padded_resolution = padded_resolution
        self.is_padding = is_padding
        self.num_padded_bins = len(self.padded_min)
        if (max_wave_step is None or np.max(self.bin_widths) <= max_wave_step) and oversample_factor == 1:
            self.query_waves = (self.padded_min + self.padded_max) / 2
            self._query_widths = self.padded_widths
            self._bin_counts = np.ones(self.num_padded_bins, dtype=int)
            wave_to_bin_map = list(range(self.num_padded_bins))
        else:
            if max_wave_step is not None and max_wave_step <= 0:
                    raise ValueError(f"max_wave_step must be positive, got {max_wave_step}.")

            # For each bin: compute the number of points that need to be sampled and
            # spread them evenly throughout the bin.
            query_waves = []
            wave_to_bin_map = []
            query_widths = []  # The width of each query wavelength point (used for integration).
            self._bin_counts = np.zeros(self.num_padded_bins, dtype=int)
            for bin_idx, (w_min, w_max) in enumerate(zip(self.padded_min, self.padded_max, strict=False)):

                # Determine the number of points based on argument used
                if max_wave_step is not None:
                    num_points = int(np.ceil((w_max - w_min) / max_wave_step))
                else:
                    num_points = oversample_factor

                self._bin_counts[bin_idx] = num_points

                # Split the bin into num_points equal-width sub-bins and evaluate at each
                # sub-bin's midpoint, so the evaluation points match the widths used for
                # integration below.
                sub_bin_edges = np.linspace(w_min, w_max, num_points + 1)
                wave_points = (sub_bin_edges[:-1] + sub_bin_edges[1:]) / 2
                query_waves.extend(wave_points)
                wave_to_bin_map.extend([bin_idx] * num_points)
                query_widths.extend([(w_max - w_min) / num_points] * num_points)
            self.query_waves = np.array(query_waves, dtype=float)
            self._query_widths = np.array(query_widths, dtype=float)

        # In the waves original order save the mapping from wave index to bin index and a mapping
        # of bin index to where the bin starts in the waves array.
        self._wave_to_bin_map = np.array(wave_to_bin_map, dtype=int)
        self._bin_starts = np.concatenate(([0], np.cumsum(self._bin_counts)[:-1]))

        # The query wavelengths should always be in increasing order.
        if not np.all(np.diff(self.query_waves) > 0):  # pragma: no cover
            raise ValueError("Query wavelengths are not in increasing order.")

        # Scale is the multiplicative factor to apply to each bin's flux.
        if scale is not None:
            if np.isscalar(scale):
                self.scale = np.full(self.num_bins, float(scale), dtype=float)
            else:
                if len(scale) != self.num_bins:
                    raise ValueError("Scale array must have the same length as the number of bins.")
                self.scale = np.asarray(scale)
        else:
            self.scale = None
        self.wavelength_resolution = wavelength_resolution if wavelength_resolution is not None else np.zeros(self.num_bins)

        # Save the other spectrograph properties if provided.
        self.instrument = instrument if instrument is not None else "Spectrograph"

        # Precompute the conversion factor from Fnu in nJy to Flam in erg/s/cm^2/AA
        # for each query wavelength.
        self._flam_conversion = fnu_to_flam(
            np.ones_like(self.query_waves),
            self.query_waves,
            wave_unit=u.AA,
            flam_unit=u.erg / u.s / u.cm**2 / u.AA,
            fnu_unit=u.nJy,
        )
        # compute the smear matrix, if there are wavelength resolutions
        self.smear_matrix = self._compute_smear_matrix() if self.wavelength_resolution is not None else None

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
        if self.bin_widths.shape != other.bin_widths.shape:  # pragma: no cover
            return False
        if not np.allclose(self.bin_widths, other.bin_widths):  # pragma: no cover
            return False
        if self.instrument != other.instrument:  # pragma: no cover
            return False
        if self.query_waves.shape != other.query_waves.shape:  # pragma: no cover
            return False
        if not np.allclose(self.query_waves, other.query_waves):  # pragma: no cover
            return False
        if not np.allclose(self.wavelength_resolution, other.wavelength_resolution):
            return False
        if self.scale is not None or other.scale is not None:
            if self.scale is None or other.scale is None:
                return False
            if not np.allclose(self.scale, other.scale):
                return False
        return True
    

    def _compute_padded_bins(self):
        """Compute the padded bins for the spectrograph.

        Parameters
        ----------
        oversampling : int, optional
            The factor by which to oversample the bins. Default is 10.

        Returns
        -------
        padded_bins : np.ndarray
            A 2D array of shape (num_bins, oversampling) representing the padded bins.
        """

        # determine the extended region
            # snana expanded by 2.5 sigma of the edge bins
        sigma_blue = self.wavelength_resolution[0]
        width_blue = self.bin_widths[0]
        num_pad_blue = int(2.5 * sigma_blue / width_blue) + 1 if sigma_blue > 0 else 0
        pad_blue_min = self.waves_min[0] - width_blue * np.arange(num_pad_blue, 0, -1)
        pad_blue_max = pad_blue_min + width_blue


        red_edge_idx = -2 if self.num_bins >= 2 else -1 # snana takes the 2nd to last bin
        sigma_red = self.wavelength_resolution[red_edge_idx]
        width_red = self.bin_widths[red_edge_idx]
        num_pad_red = int(2.5 * sigma_red / width_red) + 1 if sigma_red > 0 else 0
        pad_red_min = self.waves_max[-1] + width_red * np.arange(num_pad_red)
        pad_red_max = pad_red_min + width_red

        # combine the original bins with the padded bins
        padded_bins_min = np.concatenate((pad_blue_min, self.waves_min, pad_red_min))
        padded_bins_max = np.concatenate((pad_blue_max, self.waves_max, pad_red_max))

        padded_wavelength_resolution = np.concatenate([ # TODO: needed?
            np.full(num_pad_blue, sigma_blue),
            self.wavelength_resolution,
            np.full(num_pad_red, sigma_red)
        ])

        is_padding = np.concatenate(
            [
                np.ones(num_pad_blue, dtype=bool),
                np.zeros(self.num_bins, dtype=bool),
                np.ones(num_pad_red, dtype=bool),
            ]
        )
        return padded_bins_min, padded_bins_max, padded_wavelength_resolution, is_padding

    def _compute_smear_matrix(self, n_sigma=3):
        """Compute the smearing matrix for the spectrograph based on the wavelength resolution.

        Parameters
        ----------
        n_sigma : int, optional
            The number of standard deviations to consider for the Gaussian smearing. Default is 3.

        Returns
        -------
        smear_matrix : np.ndarray
            A 2D array of shape (num_bins, num_bins) representing the smearing matrix.
            where smear_matrix[i, j] represents the fraction of flux from bin i that smears into j
        """

        smear_matrix = np.zeros((self.num_padded_bins, self.num_padded_bins))
        for i in range(self.num_padded_bins):
            sigma = self.padded_resolution[i]

            # Determine the range of the smearing
            lambda_bin = self.padded_widths[i]
            n_bins_index = int(n_sigma * sigma / lambda_bin + 0.5)
            j_low = max(i - n_bins_index, 0)
            j_high = min(i + n_bins_index, self.num_padded_bins - 1)
            bin_center = (self.padded_min[i] + self.padded_max[i]) / 2

            # get smearing factor, if the bin is not padding
            for j in range(j_low, j_high + 1):
                # skip if in the padded region
                if self.is_padding[j]:
                    continue

                if sigma > 0:
                    # compute the limits of the Gaussian integral in units of sigma
                    lam_sig0 = (self.padded_min[j] - bin_center) / sigma
                    lam_sig1 = (self.padded_max[j] - bin_center) / sigma
                    smear_matrix[i, j] = gaussian_integral(lam_sig0, lam_sig1)
                else: # sigma == 0, no smearing to outer bins
                    smear_matrix[i, j] = 0.0 if i != j else 1.0

        return smear_matrix

    @property
    def bin_centers(self) -> np.ndarray:
        """Get the center of each wavelength bin in Angstroms."""
        return (self.waves_min + self.waves_max) / 2

    @property
    def bin_centers(self) -> np.ndarray:
        """Get the center of each wavelength bin in Angstroms."""
        return (self.waves_min + self.waves_max) / 2

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
        smear: bool = True,
    ) -> np.ndarray:
        """Calculate the bin-integrated flux for each bin in the spectrograph
        (in F_lambda units of erg/s/cm²).

        Parameters
        ----------
        flux_density_matrix : np.ndarray
            A 1D, 2D or 3D array of flux densities in nJy (Fnu). The last dimension contains the
            flux density values at the wavelengths specified by self.query_waves for a single
            sample. The other dimensions are used to represent multiple times (2D and 3D) and
            multiple objects (3D).
        smear : bool, optional
            Whether to smear the flux density values across the bins using the wavelength resolution. 
            Default is True.

        Returns
        -------
        measured_flux : np.ndarray
            An array of measure fluxes in F_lambda (units of erg/s/cm²) for each spectrograph bin.
            The array has the shape in the initial dimensions as `flux_density_matrix` and
            the last dimension corresponds to the number of spectrograph bins.
        """
        # Check that we have a valid flux density matrix.
        if flux_density_matrix.size == 0:
            raise ValueError("Empty flux density matrix used.")
        if flux_density_matrix.ndim < 1 or flux_density_matrix.ndim > 3:
            raise ValueError("Invalid flux density matrix. Must be 1, 2, or 3-dimensional.")

        num_query_waves = len(self.query_waves)
        if flux_density_matrix.shape[-1] != num_query_waves:
            raise ValueError(
                f"Flux density matrix has {flux_density_matrix.shape[-1]} wavelengths, "
                f"but the Spectrograph has {num_query_waves} wavelengths."
            )

        # Reshape the flux density matrix to a 2D array where each row corresponds to
        # a sample (e.g., if this is 3D then each of the first 2 dimensions are flattened
        # into one dimension) and each column corresponds to a wavelength.
        # We do this so we can efficiently perform integration, scaling, etc. on all samples
        # at once regardless of whether the input is 1D, 2D, or 3D.
        initial_dimensions = flux_density_matrix.shape[:-1]
        flux_density_flat = flux_density_matrix.reshape(-1, num_query_waves)

        # Convert the flux density from F_nu in nJy to F_lambda in erg/s/cm^2/Å before
        # integrating over each wavelength bin.
        flux_density_flat = flux_density_flat * self._flam_conversion[np.newaxis, :]

        # Convert the flux density at each query wavelength into a flux for each query bin.
        # We use rectangular interpolation, so the flux is just the flux density at the query
        # wavelength (center of the bin) multiplied by the width of the query bin.
        query_bin_flux_flat = flux_density_flat * self._query_widths[np.newaxis, :]

        # Compute the aggregate flux for each spectrograph bin. Note that because of how we
        # constructed the query wavelengths, the query bins perfectly (and evenly) cover each
        # spectrograph bin. and we can just sum the corresponding query bin fluxes to get the
        # integral for each spectrograph bin.
        if self._wave_to_bin_map is None:
            # We only queried the center points, so the bin fluxes equal the query fluxes.
            spectro_bin_flux_flat = query_bin_flux_flat
        else:
            # We sum batches of contiguous columns independently for each row. We can do
            # this efficiently using np.add.reduceat, since the input data is ordered such
            # that the query flux values for each bin are contiguous.
            spectro_bin_flux_flat = np.add.reduceat(query_bin_flux_flat, self._bin_starts, axis=1)

        # Reshape the bin flux density back to the original dimensions, but with the last
        # dimension corresponding to the number of bins.
        spectro_bin_flux = spectro_bin_flux_flat.reshape(*initial_dimensions, self.num_padded_bins)

        # Add per-bin smearing. By computing a B x B
        # smearing matrix in the __init__ method and then apply it here. This will allow us to model
        # the effects of the spectrograph's point spread function on the measured fluxes.
        if smear and self.smear_matrix is not None:
            spectro_bin_flux @= self.smear_matrix

        # get the flux in the spectrograph's native bins (not the padded bins)
        spectro_bin_flux = spectro_bin_flux[..., ~self.is_padding]

        # Multiply by any per-bin scaling factors.
        if self.scale is not None:
            spectro_bin_flux *= self.scale

        # Return the final result in the spectrograph's native bins.
        return spectro_bin_flux