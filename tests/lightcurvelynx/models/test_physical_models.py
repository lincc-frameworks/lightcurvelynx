import numpy as np
import pytest
from astropy.cosmology import Planck18
from lightcurvelynx.astro_utils.passbands import PassbandGroup
from lightcurvelynx.math_nodes.given_sampler import GivenValueList
from lightcurvelynx.models.basic_models import ConstantSEDModel
from lightcurvelynx.models.physical_model import SEDModel


def test_sed_model():
    """Test that we can create a SEDModel."""
    # Everything is specified.
    model1 = SEDModel(ra=1.0, dec=2.0, redshift=0.0, t0=1.0)
    state = model1.sample_parameters()

    assert model1.get_param(state, "ra") == 1.0
    assert model1.get_param(state, "dec") == 2.0
    assert model1.get_param(state, "redshift") == 0.0
    assert model1.get_param(state, "t0") == 1.0
    assert model1.get_param(state, "distance") is None
    assert not model1.has_valid_param("distance")
    assert model1.apply_redshift

    # Only the t0 parameter is in the PyTree.
    pytree = model1.build_pytree(state)
    assert len(pytree["SEDModel_0"]) == 1

    # We can turn off the redshift computation.
    model1.set_apply_redshift(False)
    assert not model1.apply_redshift

    # Derive the distance from the redshift. t0 is not given.
    model2 = SEDModel(ra=1.0, dec=2.0, redshift=1100.0, cosmology=Planck18)
    state = model2.sample_parameters()
    assert model2.get_param(state, "ra") == 1.0
    assert model2.get_param(state, "dec") == 2.0
    assert model2.get_param(state, "redshift") == 1100.0
    assert 13.0 * 1e12 < model2.get_param(state, "distance") < 16.0 * 1e12
    assert model2.get_param(state, "t0") is None

    # Check that the RedshiftDistFunc node has the same computed value.
    # The syntax is a bit ugly because we are checking internal state.
    model2_val = model2.get_param(state, "distance")
    func_val = model2.setters["distance"].dependency.get_param(state, "function_node_result")
    assert model2_val == func_val

    # Neither distance nor redshift are specified.
    model3 = SEDModel(ra=1.0, dec=2.0)
    state = model3.sample_parameters()
    assert model3.get_param(state, "redshift") is None
    assert model3.get_param(state, "distance") is None
    assert not model3.apply_redshift

    # Redshift is specified but cosmology is not.
    model4 = SEDModel(ra=1.0, dec=2.0, redshift=1100.0)
    state = model4.sample_parameters()
    assert model4.get_param(state, "redshift") == 1100.0
    assert model4.get_param(state, "distance") is None


def test_sed_model_multi_samples():
    """Test that we generate multiple samples from a SEDModel."""
    # Everything is specified.
    model1 = SEDModel(ra=1.0, dec=2.0, redshift=0.5, t0=1.0)
    state = model1.sample_parameters(num_samples=10)

    assert np.all(model1.get_param(state, "ra") == 1.0)
    assert np.all(model1.get_param(state, "dec") == 2.0)
    assert np.all(model1.get_param(state, "redshift") == 0.5)
    assert np.all(model1.get_param(state, "t0") == 1.0)


def test_sed_model_evaluate_sed():
    """Test that we can evaluate a SEDModel."""
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    waves = np.array([4000.0, 5000.0])
    brightness = GivenValueList([10.0, 20.0, 30.0])
    static_source = ConstantSEDModel(brightness=brightness)

    # Providing no state should give a single sample.
    flux = static_source.evaluate_sed(times, waves)
    assert flux.shape == (5, 2)
    assert np.all(flux == 10.0)

    # Doing a single sample should give a single sample.
    state = static_source.sample_parameters(num_samples=1)
    flux = static_source.evaluate_sed(times, waves, graph_state=state)
    assert flux.shape == (5, 2)
    assert np.all(flux == 20.0)

    # We can do multiple samples.
    brightness.reset()
    state = static_source.sample_parameters(num_samples=3)
    flux = static_source.evaluate_sed(times, waves, graph_state=state)
    assert flux.shape == (3, 5, 2)
    assert np.all(flux[0, :, :] == 10.0)
    assert np.all(flux[1, :, :] == 20.0)
    assert np.all(flux[2, :, :] == 30.0)


def test_sed_model_evaluate_redshift():
    """Test that if we apply redshift to a model we get different flux values."""
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    waves = np.array([4000.0, 5000.0])
    static_source = ConstantSEDModel(brightness=10.0, redshift=0.5, t0=0.0)

    state = static_source.sample_parameters(num_samples=3)
    flux = static_source.evaluate_sed(times, waves, graph_state=state)
    assert flux.shape == (3, 5, 2)
    assert not np.any(flux[:, :, :] == 10.0)

    # The flux should be the same for all samples (not double applying redshift).
    assert len(np.unique(flux)) == 1


def test_sed_model_evaluate_bandflux(passbands_dir):
    """Test that band fluxes are computed correctly."""
    # It should work fine for any positive Fnu.
    f_nu = np.random.lognormal()
    static_source = ConstantSEDModel(brightness=f_nu)
    state = static_source.sample_parameters()
    passbands = PassbandGroup.from_preset(preset="LSST", table_dir=passbands_dir)
    n_passbands = len(passbands)

    times = np.arange(n_passbands, dtype=float)
    filters = np.array(sorted(passbands.passbands.keys()))

    # It should fail if no filters are provided.
    with pytest.raises(ValueError):
        _bandfluxes = static_source.evaluate_bandfluxes(passbands, times=times, filters=None, state=state)
    # It should fail if single passband is provided, but with multiple filter names.
    with pytest.raises(KeyError):
        _bandfluxes = static_source.evaluate_bandfluxes(passbands.passbands["LSST_r"], times, filters, state)

    bandfluxes = static_source.evaluate_bandfluxes(passbands, times, filters, state)
    assert bandfluxes.shape == (n_passbands,)
    np.testing.assert_allclose(bandfluxes, f_nu, rtol=1e-10)

    # If we use multiple samples, we should get a correctly sized array.
    n_samples = 21
    brightness_list = [1.5 * i for i in range(n_samples)]
    static_source2 = ConstantSEDModel(brightness=GivenValueList(brightness_list))
    state2 = static_source2.sample_parameters(num_samples=n_samples)
    bandfluxes2 = static_source2.evaluate_bandfluxes(passbands, times, filters, state2)
    assert bandfluxes2.shape == (n_samples, n_passbands)
    for idx, brightness in enumerate(brightness_list):
        np.testing.assert_allclose(bandfluxes2[idx, :], brightness, rtol=1e-10)
