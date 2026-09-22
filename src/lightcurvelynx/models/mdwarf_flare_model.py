"""Models to simulate M-Dwarf Flares."""

import numpy as np

from lightcurvelynx.astro_utils.black_body import black_body_luminosity_density_per_solid
from lightcurvelynx.models.physical_model import SEDModel
from lightcurvelynx.math_nodes.np_random import NumpyRandomFunc
from scipy import special
from scipy.stats import binned_statistic
from astropy.modeling.models import BlackBody #there may be an inbuilt way to do this - check astro_utils
from astropy import units as u #angstroms, nanojanskies for flux density
from lightcurvelynx.astro_utils.pzflow_node import PZFlowNode
from astroquery.svo_fps import SvoFps
from lightcurvelynx.math_nodes.ra_dec_sampler import MilkyWayCoordSampler
from lightcurvelynx.math_nodes.basic_math_node import BasicMathNode





class MDwarfFlareModel(SEDModel):
    """An M-Dwarf Flare Model. The spectrum is modeled as a modified black body, where the flare is ~twice (or balmer_jump_ratio) as intense below the balmer jump.

    The time evolution of the M-Dwarf is modeled as in Mendoza et al. (2022), with free parameters for amplitude and full width half max of the flare. The SED and time evolution are treated as independent, with the normalized amplitude of the flare from the Mendoza model multiplied by the flare's contribution to the SED (and added to the star's constant contribution). 

    Parameterized values include:

    * dec - The object's declination in degrees. [from BasePhysicalModel]
    * distance - The object's luminosity distance in pc. [from BasePhysicalModel]
    * ra - The object's right ascension in degrees. [from BasePhysicalModel]
    * redshift - The object's redshift. [from BasePhysicalModel]
    * t0 - The t0 of the zero phase, date. [from BasePhysicalModel]

    Parameters
    ----------
    star_temp : parameter, optional
        The temperature of the star in Kelvins.
    star_radius : parameter, optional
        The radius of the star in cm.
    fwhm : parameter, optional
        The log of the full width half max of the flare in time.
    amplitude: parameter, optional
        The maximum amplitude of the flare, relative to star flux.
    flare_temp : parameter, optional
        The temperature of the cold part of the flare in Kelvins. Defaults to 9000 K. 
    balmer_jump_ratio : parameter, optional
        the ratio of the spectral intensity below the balmer jump to above the balmer jump (which is a blackbody at flare_temp). Defaults to 2

    """

    def __init__(
        self,
        *,
        star_temp=None,
        star_radius=None,
        fwhm=None,
        amplitude=None,
        flare_temp=None,
        balmer_jump_ratio=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if star_temp is None or star_radius is None or fwhm is None or amplitude is None:
            #must have put in all or none of the params
            if star_temp is not None or star_radius is not None or fwhm is not None or amplitude is not None:
                raise ValueError("must enter all or none of star_temp, star_radius, fwhm, and amplitude")
            from pzflow import Flow #conditional install
            flow = Flow(file='flare_flow.pzflow.pkl') #can do from PZFlowNode when we fix the version mismatch
            node = PZFlowNode(flow) 
            #the node has them in log space
            star_temp = BasicMathNode("10 ** log_teff", log_teff=node.logTeff)
            star_radius = BasicMathNode("10 ** log_radius", log_radius=node.logRadius)
            fwhm = BasicMathNode("10 ** log_fwhm", log_fwhm=node.logFWHM)
            amplitude = BasicMathNode("10 ** log_amp", log_amp=node.logAmp)


        if flare_temp is None:
            flare_temp = 9000 #default, Kelvin
            
        if balmer_jump_ratio is None:
            balmer_jump_ratio = 2 #default
            
        self.add_parameter(
            "star_temp", value=star_temp, description="The temperature of the star in Kelvins."
        )
        
        self.add_parameter(
            "star_radius", value=star_radius, description="The radius of the star in cm."
        )
        self.add_parameter(
            "fwhm",
            value=fwhm,
            description="The FWHM of the flare in days I think",
        ),
        self.add_parameter(
            "amplitude",
            value=amplitude,
            description="The amplitude of the flare.",
        )
        self.add_parameter(
            "balmer_jump_ratio",
            value=balmer_jump_ratio,
            description="The ratio of the spectral intensity below the balmer jump to above the balmer jump (which is a blackbody at flare_temp)",
        )
        self.add_parameter(
            "flare_temp",
            value=flare_temp,
            description="The temperature of the cold part of the flare in Kelvins."
        )
        if not self.has_valid_param("distance"):
            sampler = MilkyWayCoordSampler(seed=42, node_label="mw") #change seed?
            self.set_parameter("distance", value=sampler.distance_pc)
            if self.has_valid_param("ra") or self.has_valid_param("dec"):
                print("Overwriting RA and Dec to match our distance distribution - to avoid this, input all three values")
            self.set_parameter("ra", value=sampler.ra)
            self.set_parameter("dec", value=sampler.dec)


            
        
    

    def _flare_eqn(self, t,tpeak,fwhm,amplitude):
        '''
        The equation that defines the shape for the Continuous Flare Model. taken from Tovar Mendoza et al. (2022) DOI 10.3847/1538-3881/ac6fe6
        '''
        #Values were fit & calculated using MCMC 256 walkers and 30000 steps

        A,B,C,D1,D2,f1 = [0.9687734504375167,-0.251299705922117,0.22675974948468916,
                          0.15551880775110513,1.2150539528490194,0.12695865022878844]

        # We include the corresponding errors for each parameter from the MCMC analysis

        A_err,B_err,C_err,D1_err,D2_err,f1_err = [0.007941622683556804,0.0004073709715788909,0.0006863488251125649,
                                                  0.0013498012884345656,0.00453458098656645,0.001053149344530907 ]

        f2 = 1-f1

        eqn = ((1 / 2) * np.sqrt(np.pi) * A * C * f1 * np.exp(-D1 * t + ((B / C) + (D1 * C / 2)) ** 2)
                            * special.erfc(((B - t) / C) + (C * D1 / 2))) + ((1 / 2) * np.sqrt(np.pi) * A * C * f2
                            * np.exp(-D2 * t+ ((B / C) + (D2 * C / 2)) ** 2) * special.erfc(((B - t) / C) + (C * D2 / 2)))
        return eqn * amplitude


    def _flare_model(self, t,tpeak, fwhm, amplitude, upsample=False, uptime=10):
        '''
        Taken directly from Tovar Mendoza et al. (2022) DOI 10.3847/1538-3881/ac6fe6
        assumes kepler bandpass
        
        The Continuous Flare Model evaluated for single-peak (classical) flare events.
        Use this function for fitting classical flares with most curve_fit
        tools. Reference: Tovar Mendoza et al. (2022) DOI 10.3847/1538-3881/ac6fe6
    
        References
        --------------
        Tovar Mendoza et al. (2022) DOI 10.3847/1538-3881/ac6fe6
        Davenport et al. (2014) http://arxiv.org/abs/1411.3723
        Jackman et al. (2018) https://arxiv.org/abs/1804.03377
    
        Parameters
        ----------
        t : 1-d array
            The time array to evaluate the flare over
    
        tpeak : float
            The center time of the flare peak
    
        fwhm : float
            The Full Width at Half Maximum, timescale of the flare
    
        amplitude : float
            The amplitude of the flare
    
    
        Returns
        -------
        flare : 1-d array
            The flux of the flare model evaluated at each time
    
            A continuous flare template whose shape is defined by the convolution of a Gaussian and double exponential
            and can be parameterized by three parameters: center time (tpeak), FWHM, and ampitude
        '''
    
        t_new = (t-tpeak)/fwhm
    
        if upsample:
            dt = np.nanmedian(np.diff(np.abs(t_new)))
            timeup = np.linspace(min(t_new) - dt, max(t_new) + dt, t_new.size * uptime)
    
            flareup = flare_eqn(timeup,tpeak,fwhm,ampl)
    
            # and now downsample back to the original time...
    
            downbins = np.concatenate((t_new - dt / 2.,[max(t_new) + dt / 2.]))
            flare,_,_ = binned_statistic(timeup, flareup, statistic='mean',bins=np.sort(downbins))
        else:
    
            flare = self._flare_eqn(t_new,tpeak,fwhm,amplitude)
    
        return flare

    def _norm_flare_shape(self, t, tpeak, fwhm, **kwargs):
        """
        Peak-normalized flare shape: exactly 1.0 at t == tpeak.

        The Tovar Mendoza `amplitude` parameter is close to but not exactly
        the peak flux (the peak of flare_model(..., amplitude=1) is ~0.95,
        not 1.0),so we explicitly renormalize here rather than assume
        amplitude == peak height. This allows us to use the TESS amplitude
        distribution instead of what the parameter was originally meant for which was Kepler
        ...hopefully :)
        """
        peak = self._flare_model(np.array([tpeak]), tpeak, fwhm, amplitude=1.0, **kwargs)[0]
        return self._flare_model(t, tpeak, fwhm, amplitude=1.0, **kwargs) / peak

    
    def _build_spectrum_bb_with_balmer(self, wavelengths, temp_low=9000,  balmer_jump_ratio=2):
        '''
        Makes a spectrum for the flare temperature: 
        Blackbody at 9000 K (default) where below the balmer jump,
        the intensity is twice (or balmer_jump_ratio) as high
    
        returns units: erg / (Hz s sr cm**2 )
        '''
        if not isinstance(wavelengths, u.Quantity):
            wavelengths = wavelengths * u.AA
        if not isinstance(temp_low, u.Quantity):
            temp_low = temp_low * u.K
        bb_low = BlackBody(temperature=temp_low)
    
        intensity = bb_low(wavelengths)
        balmer_jump = 3645 * u.AA #angstroms
        
        
        intensity[wavelengths < balmer_jump] *= balmer_jump_ratio
        return intensity 
        
    def _tess_passband(self, wavelengths):
        """Interpolate TESS T(lambda) onto arbitrary wavelengths. 0 outside range."""
        #FIX?
        _filt = SvoFps.get_transmission_data('TESS/TESS.Red')
        TESS_WAVE = np.asarray(_filt['Wavelength']) * u.AA      # SVO gives this in Angstrom
        TESS_TRANS = np.asarray(_filt['Transmission'])           # dimensionless, 0-1

        wl_AA = wavelengths.to(u.AA).value
        return np.interp(wl_AA, TESS_WAVE.to(u.AA).value, TESS_TRANS,
                          left=0.0, right=0.0)

    def _tess_band_integrate(self, spectrum, wavelengths, axis=0):
        """
        Collapse a spectrum onto the TESS band: total_flux = integral of
        spectrum(lambda) * T(lambda) dlambda / lambda .

        Works on a plain 1D spectrum (n_wave,) or a stack with the wavelength
        axis at `axis` (e.g. an (n_wave, n_time) array — pass axis=0, the
        default).

        Parameters
        ----------
        spectrum : Quantity, wavelength axis at `axis`. Should be in terms of frequency
        wavelengths : Quantity, shape (n_wave,) — matches spectrum's wave axis
        axis : which axis of `spectrum` is the wavelength axis

        Returns
        -------
        Quantity, spectrum with the wavelength axis integrated out
        """
        T_lambda = self._tess_passband(wavelengths)
        wl = wavelengths .to(u.AA).value
        w = T_lambda * wl
        shape = [1] * spectrum.ndim
        shape[axis] = -1
        w = w.reshape(shape)
        # unit = spectrum.unit * u.AA #not sure if we need this?
        return np.trapezoid(spectrum.value / wl, wl, axis=axis) 

    def _quiescent_flux_no_distance(self, temp_star, wavelengths):
        """
        Quiescent stellar spectral intensity. it's a blackbody
    
        Parameters
        ----------
        temp_star : float or Quantity (assumed K if plain float)
        wavelengths : Quantity, shape (n_wave,)
    
        Returns
        -------
        Quantity, shape (n_wave,), units erg/(Hz s sr cm^2) 
        it is now in terms of frequency and not wavelength
        """
        if not isinstance(temp_star, u.Quantity):
            temp_star = temp_star * u.K
        bb = BlackBody(temperature=temp_star)
        I_lam = bb(wavelengths) 
        return  I_lam #earlier version had np.pi * u.sr * radius_star**2 *
    
    def compute_sed(self, times, wavelengths, graph_state): #try model.sample_parameters to generate single graph state
        """Draw effect-free observer frame flux densities.

        Reconstructs the flare's spectral flux F(lambda, t) from TESS-fit light-curve parameters, using the Tovar Mendoza (2022) time-domain shape and a blackbody+balmer jump spectral shape.

    equation : pi R_star**2 [(int of I_star P_tess dlambda/lambda) / (int of I_flare P_tess dlambda/lambda)]
    * lcmodel of t * I_flare


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
        params = self.get_local_params(graph_state) #should have everything that was instantiated
        num_times = len(times)
        num_waves = len(wavelengths)
        if not isinstance(wavelengths, u.Quantity):
            wavelengths = wavelengths * u.AA #angstroms
        
        flux_density = np.zeros((num_times, num_waves))

        constants = np.pi  * params["star_radius"]**2 * u.sr 
        if not isinstance(params["star_radius"], u.Quantity):
            constants = constants * (u.cm**2) 
            #we are losing the star radius units if we don't do this

        print('constants units', constants.unit)
        norm_shape = self._norm_flare_shape(times, params["t0"], params["fwhm"]) #if we want to upsample it we can add that here
        I_flare = self._build_spectrum_bb_with_balmer(wavelengths, temp_low=params["flare_temp"])
        print(I_flare.unit)
        integral_flare = self._tess_band_integrate(I_flare, wavelengths)
        I_star = self._quiescent_flux_no_distance(params["star_temp"], wavelengths) 
        integral_star = self._tess_band_integrate(I_star, wavelengths)
        flux_flare_no_distance = constants* (integral_star / integral_flare) * norm_shape[None, :] * I_flare[:, None]
        print('flux flare no distance', flux_flare_no_distance.unit)
        q_no_distance = self._quiescent_flux_no_distance(params["star_temp"], wavelengths)* constants   

        total_no_distance = q_no_distance[:, None] + flux_flare_no_distance  # erg/s/AA, no D yet
        print(total_no_distance.unit)
        if not isinstance(params["distance"], u.Quantity):
            distance = params["distance"] * u.parsec 
        else:
            distance = params["distance"]
        total_flux_at_earth = total_no_distance / ((distance.to(u.cm))**2)
        print(total_flux_at_earth.unit)
        print(distance.unit)
        print(((distance.to(u.cm))**2).unit)
        flux_density = (total_flux_at_earth).to(u.nJy, equivalencies=u.spectral_density(wavelengths[:,None])) 
        return flux_density
