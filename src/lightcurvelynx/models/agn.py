"""A model for an AGN.

Adapted from https://github.com/RickKessler/SNANA/blob/master/src/gensed_AGN.py
with the authors' permission.
"""

from os import urandom

import numpy as np
from astropy import constants
from citation_compass import cite_function
from scipy import integrate

from lightcurvelynx.base_models import FunctionNode
from lightcurvelynx.consts import ANGSTROM_TO_CM, CGS_FNU_UNIT_TO_NJY, M_SUN_G, PARSEC_TO_CM
from lightcurvelynx.math_nodes.np_random import NumpyRandomFunc
from lightcurvelynx.models.physical_model import SEDModel


def sample_damped_random_walk(times, tau_v, sf_inf, t0, rng=None):
    """Sample a damped random walk.

    Parameters
    ----------
    times : np.ndarray
        A length T array with the times at which to sample the damped random
        walk (in MJD).
    tau_v : np.ndarray
        A length W array with timescale of the damped random walk (in s).
    sf_inf : np.ndarray
        A length W array with structure function at infinity time in magnitude.
    t0 : float
        The initial time moment (in MJD).
    rng : np.random.Generator, optional
        The random number generator to use.
        Default: None

    Returns
    -------
    samples : float
        The sampled value of the damped random walk.
    """
    if rng is None:
        rng = np.random.default_rng()

    if len(sf_inf) != len(tau_v):
        raise ValueError(
            "The arrays sf_inf and tau_v must have the same length. "
            f"Received lengths: sf_inf={len(sf_inf)}, tau_v={len(tau_v)}"
        )
    samples = np.zeros((len(times), len(tau_v)))

    # Set the initial values.
    curr_t = t0
    delta_m = rng.random() * sf_inf

    # Iterate over each time step.
    for idx, time in enumerate(times):
        if time <= curr_t and idx != 0:
            raise ValueError("Times must be monotonically increasing.")

        dt = time - curr_t
        delta_m = delta_m * np.exp(-dt / tau_v) + sf_inf * np.sqrt(1 - np.exp(-2 * dt / tau_v)) * rng.random()
        samples[idx, :] = delta_m

    return samples


class AGN(SEDModel):
    """A model for an AGN.

    Parameterized values include:
      * accretion_rate - The accretion rate (ME_dot) at Eddington luminosity in g/s.
      * blackhole_accretion_rate - The accretion rate of the black hole in g/s.
      * blackhole_mass - The black hole mass in solar mass.
      * edd_ratio - The Eddington ratio.
      * dec - The object's declination in degrees. [from BasePhysicalModel]
      * distance - The object's luminosity distance in pc. [from BasePhysicalModel]
      * inclination - The inclination of the accretion disk in radians (sampled uniformly
        between 0 and pi/2).
      * L_bol - The bolometric luminosity in erg/s.
      * mag_i - The i band absolute magnitude.
      * ra - The object's right ascension in degrees. [from BasePhysicalModel]
      * redshift - The object's redshift. [from BasePhysicalModel]
      * t0 - The t0 of the zero phase, date. [from BasePhysicalModel]

    Parameters
    ----------
    t0 : float
        initial time moment in days.
    blackhole_mass : float
        The black hole mass in solar masses.
    edd_ratio: float
        Eddington ratio
    inclination_rad : float or None
        The inclination of the accretion disk in radians. If None then the value is
        sampled uniformly between 0 and pi/2.
        Default = None
    node_label : str, optional
        The label for the node in the model graph.
        Default: None
    seed : int, optional
        The seed to use for the random number generator.
        Default: None
    **kwargs : dict
        Additional keyword arguments.
    """

    def __init__(
        self,
        t0,
        blackhole_mass,
        edd_ratio,
        *,
        inclination_rad=None,
        node_label=None,
        seed=None,
        **kwargs,
    ):
        # We manually specify "node_label" as a parameter so it does not get
        # passed to the functions nodes below as part of kwargs.
        super().__init__(t0=t0, node_label=node_label, **kwargs)

        if "redshift" not in kwargs:
            raise ValueError("'redshift' parameter is required for the AGN model.")

        if "cosmology" not in kwargs and "distance" not in kwargs:
            raise ValueError(
                "At least one of 'cosmology' (str or astropy.cosmology object) or 'distance' (in pc) "
                "parameters must be provided."
            )

        # Add the parameters for the AGN. t0 already set in BasePhysicalModel.
        self.add_parameter(
            "blackhole_mass",
            blackhole_mass,
            description="The black hole mass in solar mass.",
            **kwargs,
        )
        self.add_parameter(
            "edd_ratio",
            edd_ratio,
            description="The Eddington ratio.",
            **kwargs,
        )
        if inclination_rad is None:
            inclination_rad = NumpyRandomFunc("uniform", low=0, high=np.pi / 2.0)
        self.add_parameter(
            "inclination_rad",
            inclination_rad,
            description="The inclination of the accretion disk in radians.",
            **kwargs,
        )

        # Add the derived parameters using FunctionNodes built from the object's static methods.
        # Each of these will be computed for each sample value of the input parameters.
        self.add_parameter(
            "blackhole_mass_gram",
            FunctionNode(lambda m_sun: m_sun * M_SUN_G, m_sun=self.blackhole_mass),
            description="The black hole mass in grams. Automatically computed from the blackhole_mass.",
            **kwargs,
        )
        self.add_parameter(
            "critical_accretion_rate",
            FunctionNode(self.compute_critical_accretion_rate, blackhole_mass=self.blackhole_mass_gram),
            description=(
                "The critical accretion rate (ME_dot) at Eddington luminosity in g/s. "
                "Automatically computed from the blackhole_mass_gram."
            ),
            **kwargs,
        )
        self.add_parameter(
            "blackhole_accretion_rate",
            FunctionNode(
                self.compute_blackhole_accretion_rate,
                accretion_rate=self.critical_accretion_rate,  # Pull from computed accretion rate
                edd_ratio=self.edd_ratio,  # Pull from sampled ratio
            ),
            description=(
                "The accretion rate of the black hole in g/s. Automatically computed from critical "
                "accretion rate and Eddington ratio."
            ),
            **kwargs,
        )
        self.add_parameter(
            "bolometric_luminosity",
            FunctionNode(
                self.compute_bolometric_luminosity,
                edd_ratio=self.edd_ratio,  # Pull from sampled ratio
                blackhole_mass=self.blackhole_mass_gram,  # Pull from the sampled mass
            ),
            description=(
                "The bolometric luminosity in erg/s. Automatically computed from Eddington "
                "ratio and blackhole mass.",
            ),
            **kwargs,
        )
        self.add_parameter(
            "mag_i",
            FunctionNode(
                self.compute_mag_i,
                bolometric_luminosity=self.bolometric_luminosity,  # Pull from computed value.
            ),
            description=(
                "The i band absolute magnitude. Automatically computed from the bolometric luminosity.",
            ),
            **kwargs,
        )
        self.add_parameter(
            "sf_inf_4000aa",
            FunctionNode(
                self.compute_sf_inf_4000aa,
                mag_i=self.mag_i,
                blackhole_mass=self.blackhole_mass_gram,
            ),
        )
        self.add_parameter(
            "tau_4000aa",
            FunctionNode(
                self.compute_tau_4000aa,
                mag_i=self.mag_i,
                blackhole_mass=self.blackhole_mass_gram,
            ),
        )

        # Instead of generating all the white noise values and storing them in the
        # graph state (which would be huge), we create a per model seed so we can
        # deterministically recreate the same noise values when needed.
        if seed is not None:
            self.add_parameter(
                "agn_seed",
                seed,
                description="The seed for the AGN random number generator.",
            )
        else:
            seed_generator = NumpyRandomFunc("integers", low=0, high=2**32 - 1)
            self.add_parameter(
                "agn_seed", seed_generator, description="The seed for the AGN random number generator."
            )

    # ------------------------------------------------------------------------
    # --- Static helper methods for computing the derived parameters. --------
    # ------------------------------------------------------------------------

    @staticmethod
    def compute_critical_accretion_rate(blackhole_mass):
        """Compute the critical accretion rate at Eddington luminosity.

        Parameters
        ----------
        blackhole_mass : float
            The black hole mass in g.

        Returns
        -------
        accretion_rate : float
            The accretion rate (ME_dot) at Eddington luminosity in g/s.
        """
        return 1.4e18 * blackhole_mass / M_SUN_G

    @staticmethod
    def compute_blackhole_accretion_rate(accretion_rate, edd_ratio):
        """Compute the accretion rate of the blackhole.

        Parameters
        ----------
        accretion_rate : float
            The accretion rate (ME_dot) at Eddington luminosity in g/s.
        edd_ratio : float
            The Eddington ratio.

        Returns
        -------
        bh_accretion_rate : float
            The accretion rate of the black hole in g/s.
        """
        return accretion_rate * edd_ratio

    @staticmethod
    def compute_bolometric_luminosity(edd_ratio, blackhole_mass):
        """Compute the bolometric luminosity of an AGN.

        Parameters
        ----------
        edd_ratio : float
            The Eddington ratio.
        blackhole_mass : float
            The black hole mass in g.

        Returns
        -------
        bolometric_luminosity : float
            The bolometric luminosity in erg/s.
        """
        return edd_ratio * 1.26e38 * blackhole_mass / M_SUN_G

    @staticmethod
    @cite_function
    def compute_sed_standard_disk(*, Mdot, nu, rin_to_rg, i, d, M):
        """Compute the flux based on a standard disk model.

        References
        ----------
        Lipunova, G., Malanchev, K., Shakura, N. (2018)
        https://doi.org/10.1007/978-3-319-93009-1_1

        All inputs are in CGS.

        Parameters
        ----------
        Mdot : float or np.ndarray
            The accretion rate at the previous time step.
        nu : float or np.ndarray
            The frequency.
        rin_to_rg : float or np.ndarray
            The inner radius of the accretion disk to gravitation radius ratio.
        i : float or np.ndarray
            The inclination in radians.
        d : float or np.ndarray
            The distance.
        M : float or np.ndarray
            The mass of the gravitating center.

        Returns
        -------
        flux : float
            The flux at the given time step.
        """
        # Compute the initial radius of the ring (r_0) and the effective temperature at r_0 (T_0).
        # Lipunova, G., Malanchev, K., Shakura, N. (2018). Page 33 for the main equation
        # DOI https://doi.org/10.1007/978-3-319-93009-1_1
        rin = constants.G.cgs.value * M / constants.c.cgs.value**2 * rin_to_rg
        r_0 = AGN.compute_r_0(rin)
        T_0 = AGN.compute_temp_at_r_0(M, Mdot, rin)

        h = constants.h.cgs.value
        k_B = constants.k_B.cgs.value
        c = constants.c.cgs.value
        # Set minimum x_in value for numerical stability
        x_in = np.maximum(h * nu / (k_B * T_0) * (rin / r_0) ** (3 / 4), 1e-6)
        # Set outer radius to infinity for now
        x_out = np.inf

        # large x in exponetial causes overflow, but 1/inf is zero.
        with np.errstate(over="ignore"):

            def _fun_integr(x):
                return (x ** (5 / 3)) / np.expm1(x)

            integ, _ = np.vectorize(lambda x_in, x_out: integrate.quad(_fun_integr, x_in, x_out))(x_in, x_out)

        return (
            (16 * np.pi)
            / (3 * d**2)
            * np.cos(i)
            * (k_B * T_0 / h) ** (8 / 3)
            * h
            * (nu ** (1 / 3))
            / (c**2)
            * (r_0**2)
            * integ
        ) * CGS_FNU_UNIT_TO_NJY

    @staticmethod
    @cite_function
    def compute_mag_i(bolometric_luminosity):
        """Compute the i band magnitude from the bolometric luminosity.

        References
        ----------
        Shen et al., 2013 - https://adsabs.harvard.edu/full/2013BASI...41...61S

        Parameters
        ----------
        bolometric_luminosity : float
            The bolometric luminosity in erg/s.

        Returns
        -------
        mag_i : float
            The i band magnitude.
        """
        # Adpated from Shen et al., 2013: https://adsabs.harvard.edu/full/2013BASI...41...61S
        return 90 - 2.5 * np.log10(bolometric_luminosity)

    @staticmethod
    @cite_function
    def compute_r_0(r_in):
        """Compute the initial radius of the ring (r_0) in a standard disk model
        given the inner radius.

        References
        ----------
        Lipunova, G., Malanchev, K., Shakura, N. (2018)
        https://doi.org/10.1007/978-3-319-93009-1_1

        Parameters
        ----------
        r_in : float
            The inner radius of the accretion disk.

        Returns
        -------
        r_0 : float
            The initial radius of the ring.
        """
        # Adapted from Lipunova, G., Malanchev, K., Shakura, N. (2018). Page 31
        # DOI https://doi.org/10.1007/978-3-319-93009-1_1
        return (7 / 6) ** 2 * r_in

    @staticmethod
    @cite_function
    def compute_structure_function_at_inf(sf_inf_4000aa, wavelength):
        """Adjust the structure function at infinity time for given wavelengths

        References
        ----------
        Suberlak et al. 2021 - DOI 10.3847/1538-4357/abc698

        Parameters
        ----------
        sf_inf_4000aa : float
            The structure function at infinity time for 4000 angstroms
        wavelength : np.ndarray
            A length W array with wavelength in cm.

        Returns
        -------
        result : np.ndarray
            A length W array with structure function at infinity time in magnitude.
        """
        return sf_inf_4000aa * 10 ** (-0.479) * wavelength / 4000e-8

    @staticmethod
    @cite_function
    def compute_sf_inf_4000aa(mag_i=-23, blackhole_mass=1e9 * M_SUN_G):
        """Compute the structure function at infinity time in magnitude, for 4000 A

        References
        ----------
        Suberlak et al. 2021 - DOI 10.3847/1538-4357/abc698

        Parameters
        ----------
        mag_i : float, optional
            The i band magnitude.
            Default: -23
        blackhole_mass : float, optional
            The black hole mass in g.
            Default: 1e9 * M_SUN_G

        Returns
        -------
        result : np.ndarray
            A length W array with structure function at infinity time
            in magnitude for 4000 angstroms.
        """
        # Equation and parameters for A=-0.51, B=-0.479, C=0.13, and D=0.18
        #  adopted from Suberlak et al. 2021: DOI 10.3847/1538-4357/abc698
        return 10 ** (-0.476 + 0.118 * (mag_i + 23) + 0.118 * np.log10(blackhole_mass / (1e9 * M_SUN_G)))

    @staticmethod
    @cite_function
    def compute_tau_4000aa(mag_i=-23, blackhole_mass=1e9 * M_SUN_G):
        """Compute the timescale (tau_v) at 4000 angstroms.

        References
        ----------
        Suberlak et al. 2021 - DOI 10.3847/1538-4357/abc698

        Parameters
        ----------
        mag_i : float, optional
            The i band magnitude.
            Default: -23
        blackhole_mass : float, optional
            The black hole mass in g.
            Default: 1e9 * M_SUN_G

        Returns
        -------
        tau_v : np.ndarray
            A length W array with the timescale for 4000 angstroms.
        """
        # Equation and parameters for A=2.4, B=0.17, C=0.03, and D=0.21 adopted
        # from Suberlak et al. 2021: DOI 10.3847/1538-4357/abc698
        return 10 ** (2.597 + 0.035 * (mag_i + 23) + 0.141 * np.log10(blackhole_mass / (1e9 * M_SUN_G)))

    @staticmethod
    @cite_function
    def compute_tau(tau_4000aa, wavelength):
        """Adjust the timescale (tau_v) for given wavelengths

        References
        ----------
        Suberlak et al. 2021 - DOI 10.3847/1538-4357/abc698

        Parameters
        ----------
        tau_4000aa : float
            The timescale for 4000 angstroms.
        wavelength : np.ndarray
            A length W array with wavelength in cm.
        """
        return tau_4000aa * 10**0.17 * wavelength / 4000e-8

    @staticmethod
    @cite_function
    def compute_temp_at_r_0(M, Mdot, r_in):
        """Compute the effective temperature at r0. This is the same as the maximum effective
        temperature at the disc surface (Tmax).

        References
        ----------
        Lipunova, G., Malanchev, K., Shakura, N. (2018)
        https://doi.org/10.1007/978-3-319-93009-1_1

        Parameters
        ----------
        M : float
            The mass of the gravitating centre.
        Mdot : float
            The accretion rate at the previous time step.
        r_in : float
            The inner radius of the accretion disk.

        Returns
        -------
        T_0 : float
            The effective temperature at r0.
        """
        # Lipunova, G., Malanchev, K., Shakura, N. (2018). Page 31
        # DOI https://doi.org/10.1007/978-3-319-93009-1_1
        sigma_sb = constants.sigma_sb.cgs.value
        G = constants.G.cgs.value
        return 2 ** (3 / 4) * (3 / 7) ** (7 / 4) * (G * M * Mdot / (np.pi * sigma_sb * r_in**3)) ** (1 / 4)

    def compute_sed(self, times, wavelengths, graph_state, **kwargs):
        """Draw effect-free observations for this object.

        Parameters
        ----------
        times : `numpy.ndarray`
            A length T array of rest frame timestamps.
        wavelengths : `numpy.ndarray`, optional
            A length N array of wavelengths (in angstroms).
        graph_state : `GraphState`
            An object mapping graph parameters to their values.
        **kwargs : `dict`, optional
            Any additional keyword arguments.

        Returns
        -------
        flux_density : `numpy.ndarray`
            A length T x N matrix of SED values (in nJy).
        """
        params = self.get_local_params(graph_state)
        waves_cm = wavelengths * ANGSTROM_TO_CM
        nu = constants.c.cgs.value / waves_cm
        bh_mass_g = params["blackhole_mass_gram"]

        # Get the seed for the random number generator. If not set, generate a random seed.
        seed = params["agn_seed"]
        if seed is None:
            seed = int.from_bytes(urandom(4), "big")
        rng = np.random.default_rng(seed)

        # Compute the parameters for these wavelengths.
        tau_v = self.compute_tau(params["tau_4000aa"], waves_cm)
        sf_inf = self.compute_structure_function_at_inf(params["sf_inf_4000aa"], waves_cm)

        # Compute the average flux of a standard disk model. Use a factor of 2 (two sides
        # of the disk) to get the total flux.
        fnu_average = self.compute_sed_standard_disk(
            Mdot=params["blackhole_accretion_rate"],  # g/s
            nu=nu,  # Hz
            rin_to_rg=6.0,  # rin / rg, no rotation for now
            i=params["inclination_rad"],  # radians
            d=params["distance"] * PARSEC_TO_CM,  # cm
            M=bh_mass_g,  # g
        )

        # Run the damped random walk.
        delta_m = sample_damped_random_walk(
            times,
            tau_v,
            sf_inf,
            params["t0"],
            rng=rng,
        )

        flux_density = 10 ** (-0.4 * delta_m) * fnu_average
        return flux_density
