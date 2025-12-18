import logging
from pathlib import Path

import numpy as np
import sncosmo
from astropy import units as u
from lightcurvelynx.astro_utils.noise_model import apply_noise
from lightcurvelynx.astro_utils.passbands import PassbandGroup
from lightcurvelynx.astro_utils.snia_utils import (
    DistModFromRedshift,
    HostmassX1Func,
    X0FromDistMod,
    num_snia_per_redshift_bin,
)
from lightcurvelynx.astro_utils.unit_utils import flam_to_fnu, fnu_to_flam
from lightcurvelynx.math_nodes.np_random import NumpyRandomFunc
from lightcurvelynx.math_nodes.scipy_random import SamplePDF
from lightcurvelynx.models.sncosmo_models import SncosmoWrapperModel
from lightcurvelynx.models.snia_host import SNIaHost
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


def construct_snia_source(oversampled_observations, zpdf):
    """Create a SNIA source/host pair with characteristics from an OpSim.

    Parameters
    ----------
    oversampled_observations : OpSim
        The opsim data to use.
    zpdf : interp1d
        The PDF for the redshift.

    Returns
    -------
    source : SEDModel
        The SEDModel to sample.
    """
    logger.info("Creating the source model.")

    # Get the range in which t0 can occur.
    t_min = oversampled_observations["observationStartMJD"].min()
    t_max = oversampled_observations["observationStartMJD"].max()

    # TODO: Extract the ra and dec center from the opsim in case that changes.
    # Currently we are relying on the fact the test opsim has all pointings
    # at (0.0, 0.0).

    # Create a host galaxy.
    host = SNIaHost(
        ra=NumpyRandomFunc("uniform", low=-0.5, high=0.5),  # all pointings RA = 0.0
        dec=NumpyRandomFunc("uniform", low=-0.5, high=0.5),  # all pointings Dec = 0.0
        hostmass=NumpyRandomFunc("uniform", low=7, high=12),
        redshift=SamplePDF(zpdf),
        node_label="host",
    )

    distmod_func = DistModFromRedshift(host.redshift, H0=73.0, Omega_m=0.3)
    x1_func = HostmassX1Func(host.hostmass)
    c_func = NumpyRandomFunc("normal", loc=0, scale=0.02)
    m_abs_func = NumpyRandomFunc("normal", loc=-19.3, scale=0.1)
    x0_func = X0FromDistMod(
        distmod=distmod_func,
        x1=x1_func,
        c=c_func,
        alpha=0.14,
        beta=3.1,
        m_abs=m_abs_func,
        node_label="x0_func",
    )

    sncosmo_modelname = "salt3"
    source = SncosmoWrapperModel(
        sncosmo_modelname,
        t0=NumpyRandomFunc("uniform", low=t_min, high=t_max),
        x0=x0_func,
        x1=x1_func,
        c=c_func,
        ra=NumpyRandomFunc("normal", loc=host.ra, scale=0.01),
        dec=NumpyRandomFunc("normal", loc=host.dec, scale=0.01),
        redshift=host.redshift,
        node_label="source",
    )
    return source


def load_and_register_passband(passbands_dir, to_use):
    """Load the passband from files and register with sncosmo.

    Parameters
    ----------
    passbands_dir : str
        The directory containing the passband files to use.
    to_use : list
        A list of the passbands to use.
        Example: ["g", "r"]

    Returns
    -------
    passbands : PassbandGroup
        The loaded and processed PassbandGroup.
    """
    passbands_dir = Path(passbands_dir)
    passband_list = []
    for band in to_use:
        file_path = passbands_dir / "LSST" / f"{band}.dat"
        passband_list.append({"filter_name": band, "table_path": file_path})
        logger.info(f"Loading band {band} from {file_path}")

    if len(passband_list) == 0:
        raise ValueError("No passbands being loaded.")

    # Do the actual loading and processing.
    passbands = PassbandGroup(
        passband_list,
        survey="LSST",
        units="nm",
        trim_quantile=0.001,
        delta_wave=1,
    )

    # Register sncosmo bandpasses
    for f, passband in passbands.passbands.items():
        sncosmo_bandpass = sncosmo.Bandpass(
            *passband.normalized_system_response.T, name=f"lightcurvelynx_{f}"
        )
        sncosmo.register(sncosmo_bandpass, force=True)

    return passbands


def draw_single_random_sn(
    source,
    opsim,
    passbands,
    state=None,
    rng_info=None,
):
    """Process a single random SN realization.

    Parameters
    ----------
    source : BasePhysicalModel
        The BasePhysicalModel to use for the flux computation.
    opsim : OpSim
        The OpSim for the simulations
    passbands : PassbandGroup
        The passbands to use in generating the observations.
    state : GraphState
        The sample values to use. If None resamples the state.
    rng_info : numpy.random._generator.Generator, optional
        A given numpy random number generator to use for this computation. If not
        provided, the function uses the default random number generator.

    Returns
    -------
    res : dict
        A dictionary of useful information about the run.
    """
    if state is None:
        state = source.sample_parameters(rng_info=rng_info)

    # Extract some important parameters that we need to use.
    ra = state["source"]["ra"]
    dec = state["source"]["dec"]
    t0 = state["source"]["t0"]
    z = state["source"]["redshift"]

    # Compute the rest wavelength information and save it.
    wave_obs = passbands.waves
    wavelengths_rest = wave_obs / (1.0 + z)
    res = {"wavelengths_rest": wavelengths_rest}

    # Find the times at which this source is seen.
    obs_index = np.array(opsim.range_search(ra, dec, radius=1.75))

    # Update obs_index to only include observations within SN lifespan
    phase_obs = opsim["time"][obs_index] - t0
    obs_index = obs_index[(phase_obs > -20 * (1.0 + z)) & (phase_obs < 50 * (1.0 + z))]

    # Extract the timing and filter information for those observations, changing the
    # match band names in passbands object.
    times = opsim["time"][obs_index].to_numpy()
    if len(times) == 0:
        logger.warning(f"No overlap time in opsim for (ra,dec)=({ra:.2f},{dec:.2f})")
    res["times"] = times

    filters = opsim["filter"][obs_index].to_numpy(str)
    filters = np.char.add("LSST_", filters)
    res["filters"] = filters

    # Compute the fluxes over all wavelengths.
    flux_nJy = source.evaluate_sed(times, wave_obs, graph_state=state, rng_info=rng_info)
    res["flux_nJy"] = flux_nJy
    res["flux_flam"] = fnu_to_flam(
        flux_nJy,
        wave_obs,
        wave_unit=u.AA,
        flam_unit=u.erg / u.second / u.cm**2 / u.AA,
        fnu_unit=u.nJy,
    )
    res["flux_fnu"] = flux_nJy

    # Compute the band_flixes over just the given filters.
    bandfluxes_perfect = source.evaluate_bandfluxes(passbands, times, filters, state)
    res["bandfluxes_perfect"] = bandfluxes_perfect

    bandfluxes_error = opsim.bandflux_error_point_source(bandfluxes_perfect, obs_index)
    res["bandfluxes_error"] = bandfluxes_error
    res["bandfluxes"] = apply_noise(bandfluxes_perfect, bandfluxes_error, rng=None)

    res["state"] = state

    return res


def run_snia_end2end(
    oversampled_observations,
    passbands_dir,
    solid_angle=0.0001,
    nsample=1,
    check_sncosmo=False,
    rng_info=None,
):
    """Test that we can sample and create SN Ia simulation using the salt3 model.

    Parameters
    ----------
    oversampled_observations : OpSim
        The opsim data to use.
    passbands_dir : str
        The name of the directory holding the passband information.
    solid_angle : float
        Solid angle for calculating number of SN.
    nsample : int
        The number of samples to test.
        Default:  1
    check_sncosmo : bool
        Run the simulation a second time directly with sncosmo and compare the answers.
        This should only be turned on for testing.
        Default: False
    rng_info : numpy.random._generator.Generator, optional
        A given numpy random number generator to use for this computation. If not
        provided, the function uses the default random number generator.

    Returns
    -------
    res_list : dict
        A dictionary of lists of sampling and result information.
    passbands : PassbandGroup
        The passbands used.
    """
    if rng_info is None:
        rng_info = np.random.default_rng()

    # Compute the distribution from which to sample the redshift.
    zmin = 0.1
    zmax = 0.4
    H0 = 70.0
    Omega_m = 0.3
    nsn, z = num_snia_per_redshift_bin(zmin, zmax, 100, H0=H0, Omega_m=Omega_m)
    zpdf = interp1d(z, nsn, bounds_error=False, fill_value=0)

    # Calculate nsample using SN Ia rate model
    if nsample is None and solid_angle is not None:
        nsn, _ = num_snia_per_redshift_bin(
            zmin, zmax, znbins=1, solid_angle=solid_angle, H0=H0, Omega_m=Omega_m
        )
        nsample = int(nsn[0])  # Since znbins=1 there is only one bin
        print(f"Drawing {nsample} samples from redshift {zmin} to {zmax}.")

    source = construct_snia_source(oversampled_observations, zpdf)
    passbands = load_and_register_passband(passbands_dir, to_use=["g", "r"])

    logger.info(f"Sampling {nsample} states.")
    sample_states = source.sample_parameters(num_samples=nsample)

    res_list = []
    for i in range(0, nsample):
        current_state = sample_states.extract_single_sample(i)
        res = draw_single_random_sn(
            source,
            opsim=oversampled_observations,
            passbands=passbands,
            state=current_state,
            rng_info=rng_info,
        )

        # Copy out important parameter values.
        p = {}
        for parname in ["t0", "x0", "x1", "c", "redshift", "ra", "dec"]:
            p[parname] = float(current_state["source"][parname])
        p["hostmass"] = current_state["host.hostmass"]
        p["distmod"] = current_state["x0_func.distmod"]
        res["parameter_values"] = p

        if check_sncosmo:
            saltpars = {"x0": p["x0"], "x1": p["x1"], "c": p["c"], "z": p["redshift"], "t0": p["t0"]}
            model = sncosmo.Model("salt3")
            model.update(saltpars)
            wave = passbands.waves
            time = res["times"]
            filters = res["filters"]

            flux_sncosmo = model.flux(time, wave)
            fnu_sncosmo = flam_to_fnu(
                flux_sncosmo,
                wave,
                wave_unit=u.AA,
                flam_unit=u.erg / u.second / u.cm**2 / u.AA,
                fnu_unit=u.nJy,
            )
            np.testing.assert_allclose(res["flux_nJy"], fnu_sncosmo, atol=1e-8, rtol=1e-6)
            np.testing.assert_allclose(res["flux_flam"], flux_sncosmo, atol=1e-30, rtol=1e-5)

            # Skip test for negative fluxes
            if np.all(flux_sncosmo > 0):
                sncosmo_band_names = np.char.add("lightcurvelynx_", filters)
                bandflux_sncosmo = model.bandflux(sncosmo_band_names, time, zpsys="ab", zp=8.9 + 2.5 * 9)
                np.testing.assert_allclose(res["bandfluxes_perfect"], bandflux_sncosmo, rtol=1e-1)

        res_list.append(res)

    return res_list, passbands
