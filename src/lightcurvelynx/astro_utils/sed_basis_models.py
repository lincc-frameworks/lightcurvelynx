"""Functions for creating *fake* SED basis functions for each filter.

These functions are primarily used to test and validate bandflux only models
by allowing them to compute some SED (that will be accurate when passed through
the filter curves). Users should NOT use these functions with transforms such
as redshift.
"""

import matplotlib.pyplot as plt
import numpy as np

from lightcurvelynx.astro_utils.passbands import Passband, PassbandGroup
from lightcurvelynx.consts import lsst_filter_plot_colors


class SEDBasisModel:
    """A simple class to hold SED basis model information.

    Attributes
    ----------
    wavelengths : np.ndarray
        A 1-dimensional array of wavelengths (in Angstroms).
    sed_values : dict
        A dictionary mapping filter names to their corresponding SED values
        in each wavelength (of all_waves).
    """

    def __init__(self, wavelengths, sed_values):
        self.wavelengths = wavelengths.copy()
        self.sed_values = {k: v.copy() for k, v in sed_values.items()}
        for filter, sed in sed_values.items():
            if len(sed) != len(wavelengths):
                raise ValueError(
                    f"Length of SED values for filter {filter} ({len(sed)}) does not match "
                    f"the length of wavelengths ({len(wavelengths)})."
                )

    def __len__(self):
        """Return the number of filters in the SED basis model."""
        return len(self.sed_values)

    @property
    def filters(self):
        """Return a list of all the filter names in the SED basis model."""
        return list(self.sed_values.keys())

    @classmethod
    def from_box_approximation(cls, passbands, filters=None):
        """Create box-shaped SED basis functions. For each passband this creates a box shaped SED
        that does not overlap with any other passband. The height of the SED is normalized
        such that the total flux density will be 1.0 after passing through the passband.

        Parameters
        ----------
        passbands : PassbandGroup
            The passband group to use for defining the light curve.
        filters : list, optional
            A list of filters to use for the model. If not provided, use all filters
            in the passband group.

        Returns
        -------
        sed_basis_values : SEDBasisModel
            The basis model for this set of filters.
        """
        if isinstance(passbands, Passband):
            passbands = PassbandGroup(given_passbands=[passbands])
        if filters is None:
            filters = passbands.filters

        # Mark which wavelengths are used by each passband.
        waves_per_filter = np.zeros((len(filters), len(passbands.waves)))
        for idx, filter in enumerate(filters):
            if filter not in passbands.filters:
                raise ValueError(f"Filter {filter} not found in passband group.")

            # Get all of the wavelengths that have a non-negligible transmission value
            # for this filter and find their indices in the passband group.
            is_significant = passbands[filter].normalized_system_response[:, 1] > 1e-5
            significant_waves = passbands[filter].waves[is_significant]
            indices = np.searchsorted(passbands.waves, significant_waves)

            # Mark all non-negligible wavelengths as used by this filter.
            waves_per_filter[idx, indices] = 1.0

        # Find which wavelengths are used by multiple filters.
        filter_counts = np.sum(waves_per_filter, axis=0)

        # Create the sed values for each wavelength.
        sed_basis_values = {}
        for idx, filter in enumerate(filters):
            # Get the wavelengths that are used by ONLY this filter.
            valid_waves = (waves_per_filter[idx, :] == 1) & (filter_counts == 1)
            if np.sum(valid_waves) == 0:
                raise ValueError(
                    f"Passband {filter} has no valid wavelengths where it: a) has a non-negligible "
                    "transmission value (>0.001) and b) does not overlap with another passband."
                )

            # Compute how much flux is passed through these wavelengths of this filter
            # and use this to normalize the sed values.
            filter_sed_basis = np.zeros((1, len(passbands.waves)))
            filter_sed_basis[0, valid_waves] = 1.0

            total_flux = passbands.fluxes_to_bandflux(filter_sed_basis, filter)
            if total_flux[0] <= 0:
                raise ValueError(f"Total flux for filter {filter} is {total_flux[0]}.")
            sed_basis_values[filter] = filter_sed_basis[0, :] / total_flux[0]

        sed_basis = cls(passbands.waves, sed_basis_values)
        return sed_basis

    def compute_sed(self, filter, wavelengths=None):
        """Compute the SED values for a given filter at a single time.

        Parameters
        ----------
        filter : str
            The filter for which to compute the SED.
        wavelengths : np.ndarray, optional
            A 1-dimensional array of wavelengths (in Angstroms) at which to
            compute the SED. If None, use the internal wavelengths.

        Returns
        -------
        np.ndarray
            A 1-dimensional array of SED values at each wavelength.
        """
        if wavelengths is None:
            wavelengths = self.wavelengths

        sed_waves = np.interp(
            wavelengths,  # The query wavelengths
            self.wavelengths,  # All of the passband group's wavelengths
            self.sed_values[filter],  # The SED values at each of the passband group's wavelengths
            left=0.0,  # Do not extrapolate in wavelength
            right=0.0,  # Do not extrapolate in wavelength
        )
        return sed_waves

    def compute_sed_from_bandfluxes(self, filters, bandfluxes, wavelengths=None):
        """Compute the SED values from given bandfluxes.

        Parameters
        ----------
        filters : list of str
            A length T array of filter names for each time.
        bandfluxes : np.ndarray
            A length T array of bandflux values for each filter (in nJy).
        wavelengths : np.ndarray, optional
            A length W array of wavelengths (in Angstroms) at which to
            compute the SED. If None, use the internal wavelengths.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of observer frame SED values (in nJy).
        """
        if wavelengths is None:
            wavelengths = self.wavelengths

        filters = np.asarray(filters)
        bandfluxes = np.asarray(bandfluxes)
        if filters.shape != bandfluxes.shape:
            raise ValueError("Length of filters and bandfluxes must match.")

        # Add in the SED from each filter at each time.
        flux_density = np.zeros((len(filters), len(wavelengths)), dtype=float)
        for filter in np.unique(filters):
            if filter not in self.sed_values:
                raise ValueError(f"Filter {filter} not found in SED basis model.")

            filter_mask = np.array(filters) == filter
            sed_values = self.compute_sed(filter, wavelengths)
            flux_density[filter_mask, :] = np.outer(bandfluxes[filter_mask], sed_values)

        return flux_density

    def plot(self, ax=None, figure=None):
        """Plot the basis functions for the SED.  This is a debugging
        function to help the user understand the SEDs produced by this
        model.

        Parameters
        ----------
        ax : matplotlib.pyplot.Axes or None, optional
            Axes, None by default.
        figure : matplotlib.pyplot.Figure or None
            Figure, None by default.
        """
        if ax is None:
            if figure is None:
                figure = plt.figure()
            ax = figure.add_axes([0, 0, 1, 1])

        # Plot each passband.
        for filter_name, filter_curve in self.sed_values.items():
            color = lsst_filter_plot_colors.get(filter_name, "black")
            ax.plot(self.wavelengths, filter_curve, color=color, label=filter_name)

        # Set the x and y axis labels.
        ax.set_xlabel("Wavelength (Angstroms)")
        ax.set_ylabel("SED (nJy)")
        ax.set_title("SED Basis Functions")
        ax.legend()
