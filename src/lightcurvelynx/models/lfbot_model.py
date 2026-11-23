import warnings

import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.modeling.models import BlackBody
from scipy.integrate import solve_ivp

from lightcurvelynx.math_nodes.np_random import NumpyRandomFunc
from lightcurvelynx.math_nodes.ra_dec_sampler import MilkyWayCoordSampler
from lightcurvelynx.models.physical_model import SEDModel


class LFBOTModel(SEDModel):
    """LFBOT model
    A Luminous Fast Blue Optical Transient (LFBOT) Model.

    The time evolution of the LFBOT is as modeled in Metzger 2022,
    which models a LFBOT as a merger between a Wolf-Rayet star
    and a black hole. The energy produced in this merger comes from
    a slow, equatorial outflow of mass and a fast, polar outflow of
    mass, so there are a slow and fast component contributing to the
    luminosity.

    Parameterized values include:

    * dec - The object's declination in degrees. [from BasePhysicalModel]
    * distance - The object's luminosity distance in pc. [from BasePhysicalModel]
    * ra - The object's right ascension in degrees. [from BasePhysicalModel]
    * redshift - The object's redshift. [from BasePhysicalModel]
    * t0 - The time of the merger, date. [from BasePhysicalModel]
    * WR_mass - The mass of the progenitor Wolf-Rayet star in solar masses.
    * BH_mass - The mass of the progenitor black hole in solar masses.
    * alpha - The viscosity of the accretion disk.
    * theta0 - The aspect ratio of the accretion disk.
    * p - The outflow index which helps describe the mass loss rate
    * eta - Engine efficiency
    * Mslow - Mass of the slow outflow in solar masses
    * vslow - Velocity of the slow outflow in cm/s
    * kap_slow - Opacity of the slow outflow in cm^2/g
    * Mfast - Mass of the fast outflow in solar masses
    * vfast - Velocity of the fast outflow as a fraction of the speed of light c
    * kap_fast - Opacity of the fast outflow in cm^2/g
    * phi0 - Solid angle fraction of slow outflow
    * N_orb - The number of orbits before the Wolf-Rayet-black hole merger

    Parameters
    ----------
    WR_mass : parameter, optional
        The mass of the Wolf-Rayet progenitor in solar masses
        If not provided, uniform-randomly sampled from the range
        10-40 solar masses.
    BH_mass : parameter, optional
        The mass of the black hole progenitor in solar masses
        If not provided, uniform-randomly sampled from the range
        1-20 solar masses.
    alpha : parameter, optional
        The effective viscosity of the accretion disk.
        Defaults to 0.03.
    theta0 : parameter, optional
        The aspect ratio of the accretion disk.
        Defaults to 0.33.
    p : parameter, optional
        The outflow index, defining the exponential relationship
        between mass loss and radius: dM/dt ~ r^p
        Defaults to 0.6.
    eta: parameter, optional
        Efficiency of conversion of mass into energy of accreted matter
        in E_acc ~ eta*M_acc*c^2
        Defaults to 0.03.
    Mslow: parameter, optional
        The mass of the slow-moving equatorial outflow
        in solar masses.
        Defaults to 9.8 solar mass.
    vslow: parameter, optional
        The speed of the slow-moving equatorial outflow
        in cm/s
        Defaults to 3.0e8 cm/s.
    kap_slow: parameter, optional
        The opacity of the slow-moving equatorial outflow
        in cm^2/g
        Defaults to 0.03 cm^2/g.
    Mfast: parameter, optional
        The mass of the fast-moving polar outflow
        in solar masses, constrained to be less than
        or equal to 1 solar mass.
        Defaults to 0.1 solar mass.
    vfast: parameter, optional
        The speed of the fast-moving polar outflow
        as a fraction of the speed of light
        Defaults to 0.2.
    kap_fast: parameter, optional
        The opacity of the fast-moving polar outflow
        in cm^2/g.
        Defaults to 0.2 cm^2/g.
    phi0: parameter, optional
        Solid angle fraction of the slow outflow.
        Defaults to 0.5
    N_orb: parameter, optional
        Number of orbits before merger.
        Defaults to 30.

    """

    def __init__(
        self,
        *,
        WR_mass=None,  # wolf-rayet star mass in solar masses
        BH_mass=None,  # in solar masses
        alpha=None,  # disk viscosity
        theta0=None,  # aspect ratio
        p=None,  # outflow index, Mdot(r) ~ r^p
        eta=None,  # engine efficiency
        Mslow=None,
        vslow=None,  # in cm/s
        kap_slow=None,  # cm2/g
        Mfast=None,  # * const.M_sun,
        vfast=None,  # * const.c,
        kap_fast=None,  # cm2/g
        phi0=None,  # solid-angle fraction of slow outflow
        N_orb=None,  # number of orbits before merger
        **kwargs,
    ):
        if (
            (kwargs.get("distance") is None)
            and (kwargs.get("redshift") is not None)
            and (kwargs.get("cosmology") is not None)
        ):
            warnings.warn("Distance is not provided, but will be computed from redshift and cosmology. ")

        super().__init__(**kwargs)

        if WR_mass is None:
            WR_mass = NumpyRandomFunc("uniform", low=10, high=40)
        if BH_mass is None:
            BH_mass = NumpyRandomFunc("uniform", low=1, high=20)
        if alpha is None:
            alpha = 0.03
        if theta0 is None:
            theta0 = 0.33
        if p is None:
            p = 0.6
        if eta is None:
            eta = 0.03
        if Mslow is None:
            Mslow = 9.8
        if vslow is None:
            vslow = 3.0e8
        if kap_slow is None:
            kap_slow = 0.03
        if Mfast is None:
            Mfast = 0.1
        if vfast is None:
            vfast = 0.2
        if kap_fast is None:
            kap_fast = 0.2
        if phi0 is None:
            phi0 = 0.5
        if N_orb is None:
            N_orb = 30
        if Mfast > 1:
            raise ValueError("Mfast must be less than or equal to 1 solar mass.")

        self.add_parameter(
            "WR_mass",
            value=WR_mass,
            description="The mass of the progenitor Wolf-Rayet star, in solar masses",
        )
        self.add_parameter(
            "BH_mass", value=BH_mass, description="The mass of the progenitor black hole, in solar masses"
        )
        self.add_parameter("alpha", value=alpha, description="Disk viscosity")
        self.add_parameter("theta0", value=theta0, description="Aspect ratio")
        self.add_parameter("p", value=p, description="Outflow index")
        self.add_parameter("eta", value=eta, description="Engine efficiency factor")
        self.add_parameter("Mslow", value=Mslow, description="Mass of 'slow' part of system")
        self.add_parameter("vslow", value=vslow, description="Velocity of slow outflow")
        self.add_parameter("kap_slow", value=kap_slow, description="Opacity of slow outflow")
        self.add_parameter("Mfast", value=Mfast, description="Mass of 'fast' outflow")
        self.add_parameter("vfast", value=vfast, description="Velocity of fast outflow")
        self.add_parameter("kap_fast", value=kap_fast, description="Opacity of fast outflow")
        self.add_parameter("phi0", value=phi0, description="Solid-angle fraction of slow outflow")
        self.add_parameter("N_orb", value=N_orb, description="Number of orbits before merger")
        self.add_parameter(
            "WR_R",
            value=0.8 * (WR_mass / 10) ** 0.58,  # Rsun,
            description="Radius of the WR star (pre-merger)",
        )

        if not self.has_valid_param("distance"):
            sampler = MilkyWayCoordSampler(node_label="mw")
            self.set_parameter("distance", value=sampler.distance_pc)
            if self.has_valid_param("ra") or self.has_valid_param("dec"):
                warnings.warn(
                    "Overwriting RA and Dec to match our distance distribution - \
                            \nto avoid this, input all three values"
                )
            self.set_parameter("ra", value=sampler.ra)
            self.set_parameter("dec", value=sampler.dec)

    def _lum_acc(self, t, state):
        """
        The equation describing the luminosity of the fast-outflow disk
        resulting from the Wolf Rayet star-black hole merger.
        From Brian D. Metzger (2022) DOI: 10.3847/1538-4357/ac6d59

        Parameters
        ----------
        t : float
            The time at which to evaluate the luminosity, seconds
        state : GraphState
            An object mapping graph parameters to their values.

        Returns
        -------
        L_acc : float
            The luminosity from the fast outflow at time t,
            ergs/s

        References
        -----------
        Brian D. Metzger (2022) DOI: 10.3847/1538-4357/ac6d59
        """
        params = self.get_local_params(state)
        Rd0 = 2.6 * params["WR_R"] * const.R_sun.cgs.value / 2
        Rin = 6 * const.G.cgs.value * params["BH_mass"] * const.M_sun.cgs.value / const.c.cgs.value**2

        t_visc0 = (1 / (params["alpha"] * params["theta0"] ** 2)) * np.sqrt(
            Rd0**3 / (const.G.cgs.value * params["BH_mass"] * const.M_sun.cgs.value)
        )

        tt = np.maximum(t, t_visc0)
        Mdot0 = params["WR_mass"] * const.M_sun.cgs.value / (t_visc0)

        Mdot_bh = (Mdot0 / 3) * (Rin / Rd0) ** params["p"] * (tt / t_visc0) ** (-4 * (params["p"] + 1) / 3)
        L_acc0 = params["eta"] * Mdot_bh * const.c.cgs.value**2

        L_acc = L_acc0 * np.exp(-((tt / t_visc0) ** 0.28))
        return L_acc

    def _lum_shock(self, t, state):
        """
        The equation describing the luminosity from the shock interaction of
        the Wolf-Rayet ejecta and the CSM.
        From Brian D. Metzger (2022) DOI: 10.3847/1538-4357/ac6d59

        Parameters
        ----------
        t : float
            The time at which to evaluate the luminosity, seconds
        state : GraphState
            An object mapping graph parameters to their values.

        Returns
        -------
        L_sh : float
            The luminosity from the shock interaction at time t,
            ergs/s

        References
        -----------
        Brian D. Metzger (2022) DOI: 10.3847/1538-4357/ac6d59
        """
        params = self.get_local_params(state)

        a_rlof = 2.6 * params["WR_R"] * const.R_sun.cgs.value
        Mpre = 0.1 * params["WR_mass"] * const.M_sun.cgs.value
        vpre = 0.5 * params["vslow"]

        P_orb = (
            2
            * np.pi
            * np.sqrt(
                a_rlof**3
                / (const.G.cgs.value * (params["WR_mass"] + params["BH_mass"]) * const.M_sun.cgs.value)
            )
        )
        t_run = params["N_orb"] * P_orb

        L_sh = (
            0.5 * Mpre * params["vslow"] ** 3 / (vpre * t_run) * np.exp(-params["vslow"] * t / (vpre * t_run))
        )
        return L_sh

    def _rhs(self, lnt, y, state):
        """
        The right hand side of equation describing the gradient of thermal
        energy resulting from the merger.
        From Brian D. Metzger (2022) DOI: 10.3847/1538-4357/ac6d59


        Parameters
        ----------
        lnt : float
            The natural log of the time at which to evaluate the
            luminosity, seconds
        y : list
            The initial values of E_eq and E_pol
        state : GraphState
             An object mapping graph parameters to their values.


        Returns
        -------
        [t * dE_eq, t * dE_pol] : list
            Incremental change in thermal energy, from equatorial mass flow
            and polar mass flow.
            ergs * s


        References
        -----------
        Brian D. Metzger (2022) DOI: 10.3847/1538-4357/ac6d59
        """
        t = np.exp(lnt)
        E_eq, E_pol = y
        params = self.get_local_params(state)
        R = params["vslow"] * t
        kapX = 2 * params["kap_fast"]  # X-ray opacity
        tdiff = (
            params["Mslow"] * const.M_sun.cgs.value * params["kap_slow"] / (4 * np.pi * R * const.c.cgs.value)
        )
        Lopt_sh = E_eq / (tdiff + R / const.c.cgs.value)

        dE_eq = -E_eq / t - Lopt_sh + self._lum_shock(t, state)

        R = params["vfast"] * const.c.cgs.value * t

        tdiff = (
            params["Mfast"] * const.M_sun.cgs.value * params["kap_fast"] / (4 * np.pi * R * const.c.cgs.value)
        )

        Lopt_rep = E_pol / (tdiff + R / const.c.cgs.value)
        tauX = params["Mfast"] * const.M_sun.cgs.value * kapX / (4 * np.pi * R**2)
        Lth = (1 - params["phi0"]) * (1 - np.exp(-tauX)) * self._lum_acc(t, state) + params[
            "phi0"
        ] * self._lum_acc(t, state)
        dE_pol = -E_pol / t - Lopt_rep + Lth
        return [t * dE_eq, t * dE_pol]

    def _fbot_luminosity(self, times, state):
        """
        Calculates FBOT luminosity by solving differential equation described
        in Metzger 2022 for thermal energy and luminosity.

        Parameters
        ----------
        times : 1-d numpy.ndarray
            The times at which to evaluate the luminosity, seconds

        state : GraphState
            An object mapping graph parameters to their values.

        Returns
        -------
        [R_slow, R_fast] : list
            Contains the lists of radii of the slow and fast outflow.
            cm

        [L_slow, L_fast] : list
            Contains the lists of luminosities of the slow and fast outflow.
            ergs/s

        References
        -----------
        Brian D. Metzger (2022) DOI: 10.3847/1538-4357/ac6d59
        """
        params = self.get_local_params(state)
        sol = solve_ivp(
            self._rhs,
            [np.log(times[0]), np.log(times[-1])],
            [0.0, 0.0],
            args=state,
            method="DOP853",
            rtol=1e-6,
            atol=1e30,
            dense_output=True,
        )

        R_slow, R_fast = params["vslow"] * times, params["vfast"] * const.c.cgs.value * times

        E_eq, E_pol = sol.sol(np.log(times))

        L_slow = E_eq / (
            (
                params["Mslow"]
                * const.M_sun.cgs.value
                * params["kap_slow"]
                / (4 * np.pi * R_slow * const.c.cgs.value)
            )
            + (R_slow / const.c.cgs.value)
        )

        L_fast = E_pol / (
            (
                params["Mfast"]
                * const.M_sun.cgs.value
                * params["kap_fast"]
                / (4 * np.pi * R_fast * const.c.cgs.value)
            )
            + (R_fast / const.c.cgs.value)
        )
        return [R_slow, R_fast], [L_slow, L_fast]

    def compute_sed(self, times, wavelengths, graph_state):
        """
        Constructs the SED by finding the blackbody temperature from luminosoities
        of the slow and fast outflows over the times in the times array and finding
        the flux for a blackbody at those temperatures and the distance of the model

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
        tot_flux_array : 2-d numpy.ndarray
            A length T x N matrix of observer frame SED values (in nJy).

        """
        state = graph_state
        params = self.get_local_params(graph_state)
        wavelengths = wavelengths * u.AA

        times_new = np.array(times - params["t0"]) * 86400.0  # time from merger in days to seconds

        if np.any(times_new <= 0.0):
            raise ValueError("The FBOTModel is only defined for times > t0.")

        Rs, Ls = self._fbot_luminosity(times_new, state)
        T_slow = (Ls[0] / (4 * np.pi * (Rs[0] ** 2) * const.sigma_sb.cgs.value)) ** 0.25
        T_fast = (Ls[1] / (4 * np.pi * (Rs[1] ** 2) * const.sigma_sb.cgs.value)) ** 0.25

        distance = params["distance"]
        if not isinstance(distance, u.Quantity):
            distance = distance * u.parsec

        array_slow = np.empty((len(T_slow), len(wavelengths)))
        array_fast = np.empty((len(T_fast), len(wavelengths)))

        for i, T in enumerate(T_slow):
            bbs = BlackBody(temperature=T * u.K)
            B_slow = bbs(wavelengths)
            F_slow = (np.pi * (Rs[0][i] * u.cm**2) / (distance.to(u.cm) ** 2)) * u.sr * B_slow
            array_slow[i] = F_slow.to(u.nJy)

            bbf = BlackBody(temperature=T_fast[i] * u.K)
            B_fast = bbf(wavelengths)
            F_fast = (np.pi * (Rs[1][i] * u.cm**2) / (distance.to(u.cm) ** 2)) * u.sr * B_fast
            array_fast[i] = F_fast.to(u.nJy)

        tot_flux_array = array_slow + array_fast
        return tot_flux_array
