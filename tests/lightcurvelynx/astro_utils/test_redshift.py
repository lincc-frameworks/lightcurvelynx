import numpy as np
from astropy.cosmology import WMAP9, Planck18
from lightcurvelynx.astro_utils.redshift import RedshiftDistFunc, redshift_to_distance
from lightcurvelynx.models.basic_models import StepModel


def test_redshifted_flux_densities() -> None:
    """Test that we correctly calculate redshifted values."""
    times = np.linspace(0, 100, 1000)
    wavelengths = np.array([100.0, 200.0, 300.0])
    t0 = 10.0
    t1 = 30.0
    brightness = 50.0

    for redshift in [0.0, 0.5, 2.0, 3.0, 30.0]:
        model_redshift = StepModel(brightness=brightness, t0=t0, t1=t1, redshift=redshift)
        values_redshift = model_redshift.evaluate_sed(times, wavelengths)

        for i, time in enumerate(times):
            if t0 <= time and time <= (t1 - t0) * (1 + redshift) + t0:
                # Note that the multiplication by (1+z) is due to the fact we are working in f_nu
                # units, instead of f_lambda units and may be unintuitive for users who are used to
                # working in f_lambda units. This factor can be derived by equaling the integrated
                # flux in f_nu unit before and after redshift is applied.
                assert np.all(values_redshift[i] == brightness * (1 + redshift))
            else:
                assert np.all(values_redshift[i] == 0.0)


def test_redshift_to_distance():
    """Test that we can convert the redshift to a distance using a given cosmology."""
    wmap9_val = redshift_to_distance(1100, cosmology=WMAP9)
    planck18_val = redshift_to_distance(1100, cosmology=Planck18)

    assert abs(planck18_val - wmap9_val) > 1000.0
    assert 13.0 * 1e12 < wmap9_val < 16.0 * 1e12
    assert 13.0 * 1e12 < planck18_val < 16.0 * 1e12


def test_redshift_dist_func_node():
    """Test the RedshiftDistFunc node."""
    node = RedshiftDistFunc(redshift=1100, cosmology=Planck18)
    state = node.sample_parameters()
    assert 13.0 * 1e12 < node.get_param(state, "function_node_result") < 16.0 * 1e12

    # Test that we can generate multiple samples.
    state = node.sample_parameters(num_samples=10)
    assert np.all(node.get_param(state, "function_node_result") > 13.0 * 1e12)
    assert np.all(node.get_param(state, "function_node_result") < 16.0 * 1e12)
