"""Models that generate a constant SED or bandflux at all times."""

import numpy as np
from astropy import units as u
from citation_compass import cite_function

from lightcurvelynx.math_nodes.given_sampler import GivenValueSampler
from lightcurvelynx.models.physical_model import BandfluxModel, SEDModel
from lightcurvelynx.utils.io_utils import read_numpy_data


class StaticSEDModel(SEDModel):
    """A StaticSEDModel randomly selects an SED at each evaluation and computes
    the flux from that SED at all time steps.

    Parameterized values include:
      * dec - The object's declination in degrees. [from BasePhysicalModel]
      * distance - The object's luminosity distance in pc. [from BasePhysicalModel]
      * ra - The object's right ascension in degrees. [from BasePhysicalModel]
      * redshift - The object's redshift. [from BasePhysicalModel]
      * t0 - The t0 of the zero phase, date. [from BasePhysicalModel. Not used.]

    Attributes
    ----------
    sed_values : list
       A list of SEDs from which to sample. Each SED is represented as a
       two row numpy-array where the first row is wavelength and the
       second is flux value.

    Parameters
    ----------
    sed_values : list or numpy array.
       A single SED or a list of SEDs from which to sample. Each SED is
       represented as a two row numpy-array where the first row is wavelength
       and the second is flux value.
    weights : numpy.ndarray, optional
        A length N array indicating the relative weight from which to select
        an SED at random. If None, all SEDs will be weighted equally.
    """

    def __init__(
        self,
        sed_values,
        weights=None,
        **kwargs,
    ):
        # If only a single SED was passed, then put it in a list by itself.
        if isinstance(sed_values, np.ndarray) and len(sed_values) == 2:
            self.sed_values = [sed_values]
        else:
            self.sed_values = sed_values

        # Validate the SED input data.
        for idx, sed in enumerate(self.sed_values):
            if not isinstance(sed, np.ndarray) or len(sed.shape) != 2 or sed.shape[0] != 2:
                raise ValueError(f"SED {idx} must be a two row numpy array of wavelength and flux.")
            if sed.shape[1] < 2:
                raise ValueError(f"SED {idx} must have at least 2 entries.")
            if not np.all(np.diff(sed[0, :]) > 0):
                raise ValueError(f"SED {idx} wavelenths are not in sorted order.")

        super().__init__(**kwargs)

        # Create a parameter that indicates which SED was sampled in each simulation.
        all_inds = [i for i in range(len(self.sed_values))]
        self._sampler_node = GivenValueSampler(all_inds, weights=weights)
        self.add_parameter(
            "selected_idx",
            value=self._sampler_node,
            allow_gradient=False,
            description="Index of the SED selected for sampling.",
        )

    def __len__(self):
        """Get the number of SED value."""
        return len(self.sed_values)

    @classmethod
    def from_file(cls, sed_file, **kwargs):
        """Load a static SED from a file containing a two column array where the
        first column is wavelength (in angstroms) and the second column is flux (in nJy).

        Parameters
        ----------
        sed_file : str or Path
            The path to the SED file to load.
        **kwargs : dict
            Additional keyword arguments to pass to the StaticSEDModel constructor.

        Returns
        -------
        StaticSEDModel
            An instance of StaticSEDModel with the loaded SED data.
        """
        # Load the SED data from the file (automatically detected format)
        sed_data = read_numpy_data(sed_file)
        if sed_data.ndim != 2 or sed_data.shape[1] != 2:
            raise ValueError(f"SED data from {sed_file} must be a two column array.")

        return cls(sed_values=sed_data.T, **kwargs)

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
            Additional keyword arguments to pass to the StaticSEDModel constructor.

        Returns
        -------
        StaticSEDModel
            An instance of StaticSEDModel with the generated SED data.
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
        return cls(np.vstack((waves, sed_data)), **kwargs)

    def minwave(self, graph_state=None):
        """Get the minimum wavelength of the model.

        Parameters
        ----------
        graph_state : GraphState, optional
            An object mapping graph parameters to their values. Not used
            for this model.

        Returns
        -------
        minwave : float or None
            The minimum wavelength of the model (in angstroms) or None
            if the model does not have a defined minimum wavelength.
        """
        idx = self.get_param(graph_state, "selected_idx")
        return self.sed_values[idx][0, 0]

    def maxwave(self, graph_state=None):
        """Get the maximum wavelength of the model.

        Parameters
        ----------
        graph_state : GraphState, optional
            An object mapping graph parameters to their values. Not used
            for this model.

        Returns
        -------
        maxwave : float or None
            The maximum wavelength of the model (in angstroms) or None
            if the model does not have a defined maximum wavelength.
        """
        idx = self.get_param(graph_state, "selected_idx")
        return self.sed_values[idx][0, -1]

    def compute_sed(self, times, wavelengths, graph_state):
        """Draw effect-free observer frame flux densities.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of observer frame timestamps in MJD.
        wavelengths : numpy.ndarray, optional
            A length N array of observer frame wavelengths (in angstroms).
        graph_state : GraphState
            An object mapping graph parameters to their values.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of observer frame SED values (in nJy).
        """
        # Use the SED selected by the sampler node to compute the flux density.
        model_ind = self.get_param(graph_state, "selected_idx")
        num_times = len(times)
        num_waves = len(wavelengths)

        # At each time step we interpolate SED at the query wavelengths.
        sed_fluxes = np.interp(
            wavelengths,
            self.sed_values[model_ind][0, :],
            self.sed_values[model_ind][1, :],
            left=0.0,
            right=0.0,
        )

        # We repeat the interpolated SED values at each time.
        flux_density = np.tile(sed_fluxes, num_times).reshape(num_times, num_waves)
        return flux_density


class StaticBandfluxModel(BandfluxModel):
    """A StaticBandfluxModel randomly selects a mapping of bandfluxes at each evaluation
    and uses that at all time steps.

    Parameterized values include:
      * dec - The object's declination in degrees. [from PhysicalModel]
      * distance - The object's luminosity distance in pc. [from PhysicalModel]
      * ra - The object's right ascension in degrees. [from PhysicalModel]
      * redshift - The object's redshift. [from PhysicalModel]
      * t0 - The t0 of the zero phase, date. [from PhysicalModel. Not used.]

    Attributes
    ----------
    bandflux_values : list of dict
       A list of bandflux mappings from which to sample. Each mapping is represented as a
       dictionary where the key is the filter name and the value is the bandflux (in nJy).

    Parameters
    ----------
    bandflux_values : dict or list
       A single bandflux mapping or a list of bandflux mappings from which to sample. Each mapping is
       represented as a dictionary where the key is the filter name and the value is the bandflux (in nJy).
    weights : numpy.ndarray, optional
        A length N array indicating the relative weight from which to select
        a model at random. If None, all models will be weighted equally.
    """

    def __init__(
        self,
        bandflux_values,
        weights=None,
        **kwargs,
    ):
        # If only a single bandflux mapping was passed, then put it in a list by itself.
        if isinstance(bandflux_values, dict):
            self.bandflux_values = [bandflux_values]
        else:
            self.bandflux_values = bandflux_values

        super().__init__(**kwargs)

        # Create a parameter that indicates which bandflux mapping was sampled in each simulation.
        all_inds = [i for i in range(len(self.bandflux_values))]
        self._sampler_node = GivenValueSampler(all_inds, weights=weights)
        self.add_parameter("selected_idx", value=self._sampler_node, allow_gradient=False)

    def __len__(self):
        """Get the number of band flux values."""
        return len(self.bandflux_values)

    def compute_bandflux(self, times, filter, state, rng_info=None):
        """Evaluate the model at the passband level for a single, given graph state.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of observer frame timestamps in MJD.
        filter : str
            The name of the filter.
        state : GraphState
            An object mapping graph parameters to their values with num_samples=1.
        rng_info : numpy.random._generator.Generator, optional
            A given numpy random number generator to use for this computation. If not
            provided, the function uses the node's random number generator.
        """
        # Get the selected bandflux mapping index and values.
        model_ind = self.get_param(state, "selected_idx")
        model_bandflux = self.bandflux_values[model_ind]

        # Fill in the bandflux values corresponding to the filter at each time.
        bandflux = model_bandflux[filter]
        return np.full(len(times), bandflux)
