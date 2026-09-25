import importlib
import numpy as np
from astropy import units as u
from lightcurvelynx.models.mdwarf_flare_model import MDwarfFlareModel

#test values with known output. first, no sampling

wavelengths = np.linspace(3000, 12000, 4000) 
times = np.linspace(0, 20, 201) #want peak time to actually be in times

def check_if_monotonic(values_array):
    '''
    checks whether an array is monotonic

    Parameters
    array: np.nparray

    Returns Bool
    
    '''
    return np.all(np.diff(values_array) > 0) | np.all(np.diff(values_array) < 0)


def test_mdwarf_flare_model_with_assigned_inputs():
    """
    Test the functions involved in creating and sampling an MDwarfFlareModel object.

    Using specific inputs and making sure the values are what we expect
    (Not sampling from parameter distributions)
    """
    star_temp=4000
    star_radius=.09*u.R_sun.to(u.cm)
    flare_fwhm=.1
    flare_amplitude=.03
    t0=10
    
    model = MDwarfFlareModel(star_temp=star_temp,                                star_radius=star_radius,                                  flare_fwhm=flare_fwhm, flare_amplitude=flare_amplitude, t0=t0, distance=100*u.parsec)
    
    assert np.isclose(model._quiescent_flux_no_distance(9000, wavelengths)[0].value,2.488e-26)
    
    assert np.isclose(model._build_spectrum_bb_with_balmer(wavelengths, .09*u.R_sun.to(u.cm) )[0].value,3842.055)

    I_lam_star= model._quiescent_flux_no_distance(9000*u.K, 
                                               wavelengths)
    
    assert np.isclose(model._tess_band_integrate(I_lam_star, wavelengths* u.AA), 7.009374891810094e-22)
    
    I_lam_flare = model._build_spectrum_bb_with_balmer(wavelengths, temp_low=9000) 
    
    assert np.isclose(model._tess_band_integrate(I_lam_flare, wavelengths* u.AA), 0.4724591162348664)

    sample = model.sample_parameters()
    values = model.evaluate_sed(times, wavelengths, sample)
    assert check_if_monotonic(values[:,0]) #should be no flare
    #that may not be a good check if the wavelength range is crazy but should work for normal ranges 
    assert not check_if_monotonic(values[:,100]) #when flare should exist
    
    #check that the peak amp of the flare is the same as the amplitude that we put in
    
    norm_shape = model._norm_flare_shape(times, t0, flare_fwhm) * flare_amplitude
    I_flare = model._build_spectrum_bb_with_balmer(wavelengths)
    I_star = model._quiescent_flux_no_distance(star_temp, wavelengths)
    integral_flare = model._tess_band_integrate(I_flare, wavelengths)
    integral_star = model._tess_band_integrate(I_star, wavelengths)
    constants = np.pi * star_radius**2 * u.sr * u.cm**2
    
    flux_flare_no_distance = constants * (integral_star / integral_flare) * norm_shape[None, :] * I_flare[:, None]
    q_no_distance = I_star * constants
    
    idx_peak = np.argmin(np.abs(times - t0))
    peak_flare_tess = model._tess_band_integrate(flux_flare_no_distance[:, idx_peak], wavelengths)
    star_tess = model._tess_band_integrate(q_no_distance, wavelengths)
    
    assert np.isclose(peak_flare_tess / star_tess, flare_amplitude, atol=.005)

    #test that compute_sed runs/has the shape we expect
    state = model.sample_parameters(num_samples=1)
    fluxes = model.evaluate_sed(times, wavelengths, state)

    assert fluxes.shape==(4000, 201)


def test_mdwarf_flare_model_parameter_sampling():
    """Test the sampling from parameter distributions for MDwarfFlareModel objects."""
    
    model = MDwarfFlareModel(t0=10)
    state = model.sample_parameters(num_samples=1)
    assert state["mw"]['distance_pc']>0
    assert 0 <= state["mw"]['ra'] 
    assert state["mw"]['ra'] <=360
    assert state["mw"]['dec'] >= -90
    assert state["mw"]['dec'] <= 90
    #there must be a better way to index these but this is what i have for now
    #star temp
    assert state["BasicMathNode:eval_2"]['function_node_result']>0
    #star radius - to make sure it's not in solar radii i assert >100
    assert state["BasicMathNode:eval_4"]['function_node_result']>100
    #flare_fwhm
    assert state["BasicMathNode:eval_5"]['function_node_result']>0
    assert state["BasicMathNode:eval_6"]['function_node_result']>0
    