import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest
from lightcurvelynx.astro_utils.dustmap import ConstantHemisphereDustMap, DustmapWrapper
from lightcurvelynx.effects.extinction import ExtinctionEffect
from lightcurvelynx.math_nodes.given_sampler import GivenValueList
from lightcurvelynx.models.basic_models import ConstantSEDModel


def test_list_extinction_models():
    """List the available extinction models."""
    model_names = ExtinctionEffect.list_extinction_models()
    assert len(model_names) > 10
    assert "G23" in model_names
    assert "CCM89" in model_names


def test_load_extinction_model():
    """Load an extinction model by string."""
    g23_model = ExtinctionEffect.load_extinction_model("G23", Rv=3.1)
    assert g23_model is not None
    assert hasattr(g23_model, "extinguish")

    # We fail if we try to load a model that does not exist.
    with pytest.raises(KeyError):
        ExtinctionEffect.load_extinction_model("InvalidModel")

    # We can manually load the g23_model into an ExtinctionEffect node.
    dust_effect = ExtinctionEffect(g23_model, ebv=0.1, frame="rest")

    # We can apply the extinction effect to a set of fluxes.
    fluxes = np.full((10, 3), 1.0)
    wavelengths = np.array([7000.0, 5200.0, 4800.0])
    new_fluxes = dust_effect.apply(fluxes, wavelengths=wavelengths, ebv=0.1)
    assert new_fluxes.shape == (10, 3)
    assert np.all(new_fluxes < fluxes)

    # We fail if we are missing a required parameter.
    with pytest.raises(ValueError):
        _ = dust_effect.apply(fluxes, wavelengths=wavelengths)
    with pytest.raises(ValueError):
        _ = dust_effect.apply(fluxes, ebv=0.1)


def test_set_frame():
    """Test that correct frame is set"""
    ext = ExtinctionEffect("G23", ebv=0.1, frame="observer")
    assert ext.rest_frame is False

    with pytest.raises(ValueError):
        ExtinctionEffect("G23", ebv=0.1, frame="InvalidFrame")


def test_pickle_extinction_model():
    """Test that we can pickle and unpickle an ExtinctionEffect object."""
    # Create two models: one defined by model name and the other with a given object.
    F99_model = ExtinctionEffect("F99", Rv=3.1, frame="rest", ebv=0.1)

    ext_model = ExtinctionEffect.load_extinction_model("F99")
    F99_model_B = ExtinctionEffect(ext_model, Rv=3.1, frame="rest", ebv=0.1)

    # Compute the some sample fluxes before and after extinction.
    org_fluxes = np.full((10, 3), 1.0)
    wavelengths = np.array([7000.0, 5200.0, 4800.0])
    ext_fluxes_1 = F99_model.apply(org_fluxes, wavelengths=wavelengths, ebv=0.1)
    assert ext_fluxes_1.shape == (10, 3)

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test_f99_model.pkl"
        assert not file_path.exists()

        with open(file_path, "wb") as f:
            pickle.dump(F99_model, f)
        assert file_path.exists()

        with open(file_path, "rb") as f:
            loaded_F99_model = pickle.load(f)
        assert loaded_F99_model is not None

        ext_fluxes_2 = loaded_F99_model.apply(org_fluxes, wavelengths=wavelengths, ebv=0.1)
        assert ext_fluxes_2.shape == (10, 3)
        assert np.allclose(ext_fluxes_1, ext_fluxes_2)

        # We fail trying to pickle the model given the actual extinction object (instead of
        # the name), because we won't have enough information to unpickle.
        file_path2 = Path(tmpdir) / "test_f99_model2.pkl"
        with open(file_path2, "wb") as f2:
            with pytest.raises(ValueError):
                pickle.dump(F99_model_B, f2)


def test_constant_dust_extinction():
    """Test that we can create and sample a ExtinctionEffect object."""
    # Use given ebv values. Usually these would be computed from a dustmap,
    # based on (RA, dec).
    ebv_node = GivenValueList([0.1, 0.2, 0.3, 0.4, 0.5])
    dust_effect = ExtinctionEffect("CCM89", ebv=ebv_node, Rv=3.1, frame="rest")
    assert dust_effect.extinction_model is not None
    assert hasattr(dust_effect.extinction_model, "extinguish")

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

    assert fluxes.shape == (3, 5, 3)
    assert np.all(fluxes < 100.0)
    assert np.all(fluxes[0, :, :] > fluxes[1, :, :])
    assert np.all(fluxes[1, :, :] > fluxes[2, :, :])


def test_dustmap_chain():
    """Test that we can chain the dustmap computation and extinction effect."""
    model = ConstantSEDModel(
        brightness=100.0,
        ra=GivenValueList([45.0, 45.0, 45.0]),
        dec=GivenValueList([20.0, -20.0, 10.0]),
        redshift=0.0,
    )

    # Create a constant dust map for testing.
    dust_map = ConstantHemisphereDustMap(north_ebv=0.8, south_ebv=0.5)
    dust_map_node = DustmapWrapper(dust_map, ra=model.ra, dec=model.dec)

    # Create an extinction effect using the EBVs from that dust map.
    ext_effect = ExtinctionEffect(extinction_model="CCM89", ebv=dust_map_node, Rv=3.1, frame="rest")
    model.add_effect(ext_effect)

    # Sample the model.
    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    wavelengths = np.array([7000.0, 5200.0])
    states = model.sample_parameters(num_samples=3)
    fluxes = model.evaluate_sed(times, wavelengths, states)

    assert fluxes.shape == (3, 5, 2)
    assert np.all(fluxes < 100.0)
    assert np.allclose(fluxes[0, :, :], fluxes[2, :, :])
    assert np.all(fluxes[1, :, :] > fluxes[0, :, :])
