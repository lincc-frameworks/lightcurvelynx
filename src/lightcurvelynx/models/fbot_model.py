import warnings

import numpy as np
from astropy import constants as const
from astropy import units as u  # angstroms, nanojanskies for flux density
from astropy.modeling.models import BlackBody
from scipy.integrate import solve_ivp

from lightcurvelynx.math_nodes.ra_dec_sampler import MilkyWayCoordSampler
from lightcurvelynx.models.physical_model import SEDModel


class FBOTModel(SEDModel):
    """FBOT model
    A Luminous Fast Blow Optical Transient (LFBOT) Model.

    The time evolution of the LFBOT is as modeled in Metzger 2022,
    which models a LFBOT as a merger between a Wolf-Rayet star
    and a black hole. The spectrum of the LFBOT is considered to
    be from the combined


    Parameterized values include:

    * dec - The object's declination in degrees. [from BasePhysicalModel]
    * distance - The object's luminosity distance in pc. [from BasePhysicalModel]
    * ra - The object's right ascension in degrees. [from BasePhysicalModel]
    * redshift - The object's redshift. [from BasePhysicalModel]
    * t0 - The t0 of the zero phase, date. [from BasePhysicalModel]
    * WR_mass - The mass of the progenitor Wolf-Rayet star in solar masses.
    * BH_mass - The mass of the progenitor black hole in solar masses.
    * alpha - The viscosity of the
    * theta0 -
    * p - The outflow index which helps describe the mass loss rate
    * eta - Engine efficiency
    * Mslow - in solar masses
    * vslow - in cm/s
    * kap_slow - opacity of the slow in cm^2/g
    * Mfast - in solar masses
    * vfast - fraction of the speed of light c
    * kap_fast - opacity of the fast in cm^2/g
    * phi0 - solid angle fraction of slow outflow
    * N_orb - the number of orbits before the Wolf-Rayet-black hole merger

    Parameters
    ----------
    WR_mass : parameter, optional
        The mass of the Wolf-Rayet progenitor in solar masses
        If not provided, the model uses
    BH_mass : parameter, optional

    alpha : parameter, optional

    theta0 : parameter, optional

    p : parameter, optional

    eta: parameter, optional

    Mslow: parameter, optional
        In solar masses

    vslow: parameter, optional
        in cm/s

    kap_slow: parameter, optional
        in cm^2/g

    Mfast: parameter, optional
        In solar masses

    vfast: parameter, optional
        fraction of the speed of light

    kap_fast: parameter, optional
        in cm^2/g

    phi0: parameter, optional

    N_orb: parameter, optional


    star_temp : parameter, optional
        The temperature of the star in Kelvins.
        If not provided, the model uses samples from the default pzflow model.
    star_radius : parameter, optional
        The radius of the star in cm.
        If not provided, the model uses samples from the default pzflow model.
    flare_fwhm : parameter, optional
        The full width half max of the flare in time (days).
        If not provided, the model uses samples from the default pzflow model.
    flare_amplitude: parameter, optional
        The maximum amplitude of the flare, relative to star flux.
        If not provided, the model uses samples from the default pzflow model.
    flare_temp : parameter, optional
        The temperature of the cold part of the flare in Kelvins.
        If not provided, uniform-randomly sampled between 9000 and 14000 K.
    balmer_jump_ratio : parameter, optional
        the ratio of the spectral intensity below the balmer jump to above the balmer jump
        (which is a blackbody at flare_temp). Defaults to 2

    """

    def __init__(
        self,
        *,
        WR_mass=10,  # wolf-rayet star mass in solar masses
        BH_mass=10,  # in solar masses
        alpha=0.03,  # disk viscosity
        theta0=0.33,  # aspect ratio
        p=0.6,  # outflow index, Mdot(r) ~ r^p
        eta=0.03,  # engine efficiency
        Mslow=9.8,
        vslow=3.0e8,  # in cm/s
        kap_slow=0.03,  # cm2/g
        Mfast=0.1,  # * const.M_sun,
        vfast=0.2,  # * const.c,
        kap_fast=0.2,  # cm2/g
        phi0=0.5,  # solid-angle fraction of slow outflow
        N_orb=30,  # number of orbits before merger
        **kwargs,
    ):
        if (
            (kwargs.get("distance") is None)
            and (kwargs.get("redshift") is not None)
            and (kwargs.get("cosmology") is not None)
        ):
            warnings.warn("Distance is not provided, but will be computed from redshift and cosmology. ")

        super().__init__(**kwargs)

        self.add_parameter(
            "WR_mass",
            value=WR_mass,
            description="The mass of the progenitor Wolf-Rayet star, in solar masses",
        )  # has to be between 10-40 Msun
        self.add_parameter(
            "BH_mass", value=BH_mass, description="The mass of the progenitor black hole, in solar masses"
        )  # has to be between 1-20 Msun
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
            value=0.8 * (WR_mass / 10) ** 0.58,  # * const.R_sun,
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
        The equation describing the luminosity of the fast-outflow disk.
        From Brian D. Metzger (2022) DOI: 10.3847/1538-4357/ac6d59


        Parameters
        ----------
        t : float
            The time to evaluate the luminosity at, seconds
        state : GraphState
            An object mapping graph parameters to their values.

        Returns
        -------
        L_acc : float
            The luminosity from the fast outflow at time, t

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
        # t_visc0 = t_visc0.decompose().value
        tt = np.maximum(t, t_visc0)
        Mdot0 = params["WR_mass"] * const.M_sun.cgs.value / (t_visc0)

        Mdot_bh = (Mdot0 / 3) * (Rin / Rd0) ** params["p"] * (tt / t_visc0) ** (-4 * (params["p"] + 1) / 3)
        L_acc0 = params["eta"] * Mdot_bh * const.c.cgs.value**2

        L_acc = L_acc0 * np.exp(-((tt / t_visc0) ** 0.28))
        return L_acc

    def _lum_shock(self, t, state):
        """
        The equation describing the luminosity from the shock interaction of
        the wolf-Rayet ejecta and the CSM.
        From Brian D. Metzger (2022) DOI: 10.3847/1538-4357/ac6d59

        Parameters
        ----------
        t : float
            The time to evaluate the luminosity at, seconds
        state : GraphState
            An object mapping graph parameters to their values.

        Returns
        -------
        L_sh : float
            The luminosity from the shock interaction at time t

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
        # t_run = t_run.decompose().value
        L_sh = (
            0.5 * Mpre * params["vslow"] ** 3 / (vpre * t_run) * np.exp(-params["vslow"] * t / (vpre * t_run))
        )
        return L_sh

    def _rhs(self, lnt, y, state):
        """
        The righthand side of equation describing the luminosity of the fast-outflow disk.
        From Brian D. Metzger (2022) DOI: 10.3847/1538-4357/ac6d59


        Parameters
        ----------
        lnt : float
            The natural log of the time to evaluate the luminosity at, seconds
        y : list
            The values of E_eq and E_pol to evaluate the equation at
        state : GraphState
             An object mapping graph parameters to their values.


        Returns
        -------
        [t * dE_eq, t * dE_pol] : list
            The luminosity from the fast outflow at time, t

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
        # Lopt_sh = Lopt_sh.decompose().value
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
        # Lth = Lth.decompose().value
        # Lopt_rep = Lopt_rep.decompose().value
        dE_pol = -E_pol / t - Lopt_rep + Lth
        return [t * dE_eq, t * dE_pol]

    def _fbot_luminosity(self, times, state):
        """
        Calculates FBOT luminosity by solving differential equation described


        Parameters
        ----------
        times : 1-d numpy.ndarray
            The time to evaluate the luminosity at, seconds

        state : GraphState
            An object mapping graph parameters to their values.

        Returns
        -------
        [R_slow, R_fast] : list
            The radii of the slow and fast outflow

        [L_slow, L_fast] : list
            The luminosities of the slow and fast outflow

        References
        -----------
        Brian D. Metzger (2022) DOI: 10.3847/1538-4357/ac6d59
        """
        params = self.get_local_params(state)
        # print("pre solve", times)
        # print([np.log(times[0]), np.log(times[-1])])
        sol = solve_ivp(
            self._rhs,
            [np.log(times[0]), np.log(times[-1])],
            [0.0, 0.0],
            args=state,
            method="DOP853",  # method="LSODA",  # method = "RK45", #,
            rtol=1e-8,
            atol=1e30,
            # t_eval = np.linspace(np.log(times[0]), np.log(times[-1]), len(times)),
            dense_output=True,
            # dense_output=True,
        )

        # t = np.logspace(np.log10(times[0]), np.log10(times[-1]), 800)
        R_slow, R_fast = params["vslow"] * times, params["vfast"] * const.c.cgs.value * times
        # print(times/86400)

        # print(sol.y)
        E_eq, E_pol = sol.sol(np.log(times))  # sol.y #sol.sol(np.log(times))
        # print(E_eq[0:10], E_pol[0:10])

        L_slow = E_eq / (
            params["Mslow"] * params["kap_slow"] / (4 * np.pi * R_slow * const.c.cgs.value)
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
        # print((R_fast / const.c.cgs.value))
        # L_slow = L_slow.decompose().value
        # L_fast = L_fast.decompose().value
        # print(L_slow[0:10], L_fast[0:10])
        return [R_slow, R_fast], [L_slow, L_fast]

    def compute_sed(self, times, wavelengths, graph_state):
        """
        Finds the SED


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

        # print("original times", len(times))

        times_new = np.array(times - params["t0"]) * 86400.0  # time from merger in days to seconds

        # print("new t", times_new)

        Rs, Ls = self._fbot_luminosity(times_new, state)
        # print("hi", Rs[0][1], Ls[0][1])
        T_slow = (Ls[0] / (4 * np.pi * (Rs[0] ** 2) * const.sigma_sb.cgs.value)) ** 0.25
        T_fast = (Ls[1] / (4 * np.pi * (Rs[1] ** 2) * const.sigma_sb.cgs.value)) ** 0.25

        # T_slow = T_slow.decompose().value
        # T_fast = T_fast.decompose().value

        # print("Temps",max(T_slow), max(T_fast))

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
