"""A class to hold SED information."""

import numpy as np
from astropy import units as u
from citation_compass import cite_function

from lightcurvelynx.utils.io_utils import read_numpy_data, write_numpy_data


class SED:
    """A class to hold SED information.

    Attributes
    ----------
    wavelengths : np.ndarray
        The wavelength values of the SED.
    fluxes : np.ndarray
        The flux values of the SED.
    """

    def __init__(self, wavelengths, fluxes, **kwargs):
        self.wavelengths = np.array(wavelengths)
        self.fluxes = np.array(fluxes)

        if len(wavelengths) < 2:
            raise ValueError("SED must have at least two wavelength and flux values.")
        if len(wavelengths) != len(fluxes):
            raise ValueError("Wavelengths and fluxes must have the same length.")
        if not np.all(np.diff(wavelengths) > 0):
            raise ValueError("Wavelengths are not in sorted order.")

    @classmethod
    def from_file(cls, sed_file, **kwargs):
        """Load a static SED from a file containing a two column array where the
        first column is wavelength (in angstroms) and the second column is flux (in nJy).

        Parameters
        ----------
        sed_file : str or Path
            The path to the SED file to load.
        **kwargs : dict
            Additional keyword arguments to pass to the SED constructor.

        Returns
        -------
        SED
            An instance of SED with the loaded SED data.
        """
        # Load the SED data from the file (automatically detected format)
        sed_data = read_numpy_data(sed_file)
        if sed_data.ndim != 2 or sed_data.shape[1] != 2:
            raise ValueError(f"SED data from {sed_file} must be a two column array.")

        return cls(sed_data[:, 0], sed_data[:, 1], **kwargs)

    @classmethod
    @cite_function
    def from_synphot(cls, sp_model, waves=None, **kwargs):
        """Generate the spectrum from a given synphot model.

        References
        ----------
        synphot (ascl:1811.001)

        Parameters
        ----------
        sp_model : synphot.SourceSpectrum
            The synphot model to generate the spectrum from.
        waves : numpy.ndarray, optional
            A length N array of wavelengths (in angstroms) at which to sample the SED.
            If None, the SED will be sampled at the wavelengths defined in the synphot model.
        **kwargs : dict
            Additional keyword arguments to pass to the SED constructor.

        Returns
        -------
        SED
            An instance of SED with the generated SED data.
        """
        try:
            from synphot import units
        except ImportError as err:  # pragma: no cover
            raise ImportError(
                "synphot package is not installed be default. To use the synphot models, please "
                "install it. For example, you can install it with `pip install synphot`."
            ) from err

        if sp_model.z > 0.0:
            raise ValueError(
                "The synphot model must be defined at the rest frame (z=0.0). "
                f"Current redshift is {sp_model.z}."
            )

        if waves is None:
            waves = np.array(sp_model.waveset * u.angstrom)

        # Extract the SED data from the synphot model. Synphot models return flux in units
        # of PHOTLAM (photons s^-1 cm^-2 A^-1), so we convert to nJy.
        photlam_flux = sp_model(waves, flux_unit=units.PHOTLAM)
        sed_data = np.array(units.convert_flux(waves, photlam_flux, "nJy"))
        return cls(waves, sed_data, **kwargs)

    def minwave(self):
        """Get the minimum wavelength of the SED.

        Returns
        -------
        minwave : float
            The minimum wavelength of the SED (in angstroms).
        """
        return self.wavelengths[0]

    def maxwave(self):
        """Get the maximum wavelength of the SED.

        Returns
        -------
        maxwave : float
            The maximum wavelength of the SED (in angstroms).
        """
        return self.wavelengths[-1]

    def evaluate(self, wavelengths):
        """Evaluate the SED at the given wavelengths by interpolating the SED data.

        Parameters
        ----------
        wavelengths : numpy.ndarray, optional
            A length N array of wavelengths (in angstroms).

        Returns
        -------
        flux_density : numpy.ndarray
            A length N matrix of observer frame SED values (in nJy).
        """
        sed_fluxes = np.interp(
            wavelengths,
            self.wavelengths,
            self.fluxes,
            left=0.0,
            right=0.0,
        )
        return sed_fluxes

    def to_file(self, sed_file):
        """Save the SED to a file as a two column array where the first column is wavelength
        (in angstroms) and the second column is flux (in nJy).

        Parameters
        ----------
        sed_file : str or Path
            The path to the SED file to save.
        """
        sed_data = np.column_stack((self.wavelengths, self.fluxes))
        write_numpy_data(sed_file, sed_data)
