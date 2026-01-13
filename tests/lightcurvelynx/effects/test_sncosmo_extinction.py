import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest
from lightcurvelynx.effects.sncosmo_extinction import SncosmoExtinctionEffect
from lightcurvelynx.math_nodes.given_sampler import GivenValueList
from lightcurvelynx.models.basic_models import ConstantSEDModel


def test_load_extinction_model():
    """Load an extinction model by string."""
    fm07_model = SncosmoExtinctionEffect.load_extinction_model("fm07")
    assert fm07_model is not None
    assert callable(fm07_model)

    # We can manually load the fm07_model into an ExtinctionEffect node.
    dust_effect = SncosmoExtinctionEffect(fm07_model, a_v=0.1, frame="rest")

    # We can apply the extinction effect to a set of fluxes.
    fluxes = np.full((10, 3), 1.0)
    wavelengths = np.array([7000.0, 5200.0, 4800.0])
    new_fluxes = dust_effect.apply(fluxes, wavelengths=wavelengths, a_v=0.1)
    assert new_fluxes.shape == (10, 3)
    assert np.all(new_fluxes < fluxes)

    # We fail if we are missing a required parameter.
    with pytest.raises(ValueError):
        _ = dust_effect.apply(fluxes, wavelengths=wavelengths)
    with pytest.raises(ValueError):
        _ = dust_effect.apply(fluxes, a_v=0.1)


def test_load_extinction_model_with_r_v():
    """Load an extinction model by string."""
    od94_model = SncosmoExtinctionEffect.load_extinction_model("odonnell94")
    assert od94_model is not None
    assert callable(od94_model)

    # We can manually load the od94_model into an ExtinctionEffect node.
    dust_effect = SncosmoExtinctionEffect(od94_model, a_v=0.1, r_v=3.1, frame="rest")

    # We can apply the extinction effect to a set of fluxes.
    fluxes = np.full((10, 3), 1.0)
    wavelengths = np.array([7000.0, 5200.0, 4800.0])
    new_fluxes = dust_effect.apply(fluxes, wavelengths=wavelengths, a_v=0.1, r_v=3.1)
    assert new_fluxes.shape == (10, 3)
    assert np.all(new_fluxes < fluxes)

    # We fail if we are missing a required parameter.
    with pytest.raises(ValueError):
        _ = dust_effect.apply(fluxes, wavelengths=wavelengths, r_v=3.1)
    with pytest.raises(ValueError):
        _ = dust_effect.apply(fluxes, a_v=0.1, r_v=3.1)
    with pytest.raises(ValueError):
        _ = dust_effect.apply(fluxes, wavelengths=wavelengths, a_v=0.1)


def test_set_frame():
    """Test that correct frame is set"""
    ext = SncosmoExtinctionEffect("odonnell94", a_v=0.1, r_v=3.1, frame="observer")
    assert ext.rest_frame is False

    with pytest.raises(ValueError):
        SncosmoExtinctionEffect("odonnell94", a_v=0.1, r_v=3.1, frame="InvalidFrame")


def test_pickle_extinction_model():
    """Test that we can pickle and unpickle an SncosmoExtinctionEffect object."""
    # Create two models: one defined by model name and the other with a given object.
    model_A = SncosmoExtinctionEffect("odonnell94", a_v=0.1, r_v=3.1, frame="rest")

    ext_model = SncosmoExtinctionEffect.load_extinction_model("odonnell94")
    model_B = SncosmoExtinctionEffect(ext_model, a_v=0.1, r_v=3.1, frame="rest")

    # Compute the some sample fluxes before and after extinction.
    org_fluxes = np.full((10, 3), 1.0)
    wavelengths = np.array([7000.0, 5200.0, 4800.0])
    ext_fluxes_1 = model_A.apply(org_fluxes, wavelengths=wavelengths, a_v=0.1, r_v=3.1)
    assert ext_fluxes_1.shape == (10, 3)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Pickle the model determined by a string.
        file_path = Path(tmpdir) / "test_model_a.pkl"
        assert not file_path.exists()

        with open(file_path, "wb") as f:
            pickle.dump(model_A, f)
        assert file_path.exists()

        with open(file_path, "rb") as f:
            loaded_model = pickle.load(f)
        assert loaded_model is not None

        ext_fluxes_2 = loaded_model.apply(org_fluxes, wavelengths=wavelengths, a_v=0.1, r_v=3.1)
        assert ext_fluxes_2.shape == (10, 3)
        assert np.allclose(ext_fluxes_1, ext_fluxes_2)

        # Now try to pickle the model defined with an actual extinction object.
        file_path = Path(tmpdir) / "test_model_b.pkl"
        assert not file_path.exists()

        with open(file_path, "wb") as f:
            pickle.dump(model_B, f)
        assert file_path.exists()

        with open(file_path, "rb") as f:
            loaded_model = pickle.load(f)
        assert loaded_model is not None

        ext_fluxes_2 = loaded_model.apply(org_fluxes, wavelengths=wavelengths, a_v=0.1, r_v=3.1)
        assert ext_fluxes_2.shape == (10, 3)
        assert np.allclose(ext_fluxes_1, ext_fluxes_2)


def test_constant_sncosmo_extinction():
    """Test that we can create and sample a SncosmoExtinctionEffect object."""
    # Use given ebv values. Usually these would be computed from a dustmap,
    # based on (RA, dec).
    a_v_node = GivenValueList([0.1, 0.2, 0.3, 0.4, 0.5])
    dust_effect = SncosmoExtinctionEffect("odonnell94", a_v=a_v_node, r_v=3.1, frame="rest")
    assert dust_effect.extinction_model is not None

    model = ConstantSEDModel(
        brightness=100.0,
        ra=0.0,
        dec=40.0,
        redshift=0.0,
    )
    model.add_effect(dust_effect)

    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    wavelengths = np.array([7000.0, 5200.0, 4800.0])  # Red, green, blue
    states = model.sample_parameters(num_samples=3)
    fluxes = model.evaluate_sed(times, wavelengths, states)

    # We check that all fluxes are reduced, and that higher a_v leads to
    # lower fluxes.
    assert fluxes.shape == (3, 5, 3)
    assert np.all(fluxes < 100.0)
    assert np.all(fluxes[0, :, :] > fluxes[1, :, :])
    assert np.all(fluxes[1, :, :] > fluxes[2, :, :])
