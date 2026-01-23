import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest
from lightcurvelynx.astro_utils.dustmap import ConstantHemisphereDustMap, DustmapWrapper
from lightcurvelynx.effects.extinction import ExtinctionEffect, _DustExtinctionWrapper
from lightcurvelynx.math_nodes.given_sampler import GivenValueList
from lightcurvelynx.models.basic_models import ConstantSEDModel


def test_list_dust_extinction_models():
    """List the available models in the dust_extinction package."""
    model_names = _DustExtinctionWrapper.list_extinction_models()
    assert len(model_names) > 10
    assert "G23" in model_names
    assert "CCM89" in model_names


def test_extinction_effect_dust_extinction():
    """Test that we can create a ExtinctionEffects from the dust_extinction package."""
    for model in ["CCM89", "F99", "G23"]:
        dust_effect = ExtinctionEffect(model, ebv=0.1, r_v=3.1, frame="rest", backend="dust_extinction")

        # We can apply the extinction effect to a set of fluxes.
        fluxes = np.full((10, 3), 1.0)
        wavelengths = np.array([7000.0, 5200.0, 4800.0])
        new_fluxes = dust_effect.apply(fluxes, wavelengths=wavelengths, ebv=0.1, r_v=3.1)
        assert new_fluxes.shape == (10, 3)
        assert np.all(new_fluxes < fluxes)

        # We fail if we are missing a required parameter (either ebv or wavelengths).
        with pytest.raises(ValueError):
            _ = dust_effect.apply(fluxes, wavelengths=wavelengths)
        with pytest.raises(ValueError):
            _ = dust_effect.apply(fluxes, ebv=0.1)

    # We fail if we try to use an invalid model.
    with pytest.raises(KeyError):
        _ = ExtinctionEffect("InvalidModel", ebv=0.1, r_v=3.1, frame="rest", backend="dust_extinction")


def test_extinction_effect_extinction():
    """Test that we can create a ExtinctionEffects from the extinction package."""
    for model in ["odonnell94", "calzetti00"]:
        dust_effect = ExtinctionEffect(model, ebv=0.1, r_v=3.1, frame="rest", backend="extinction")

        # We can apply the extinction effect to a set of fluxes.
        fluxes = np.full((10, 3), 1.0)
        wavelengths = np.array([7000.0, 5200.0, 4800.0])
        new_fluxes = dust_effect.apply(fluxes, wavelengths=wavelengths, ebv=0.1, r_v=3.1)
        assert new_fluxes.shape == (10, 3)
        assert np.all(new_fluxes < fluxes)

        # We fail if we are missing a required parameter (either ebv or wavelengths).
        with pytest.raises(ValueError):
            _ = dust_effect.apply(fluxes, wavelengths=wavelengths)
        with pytest.raises(ValueError):
            _ = dust_effect.apply(fluxes, ebv=0.1)

    # We fail if we try to use an invalid model.
    with pytest.raises(KeyError):
        _ = ExtinctionEffect("InvalidModel", ebv=0.1, r_v=3.1, frame="rest", backend="extinction")

    # We fail for a model that requires r_v if we don't provide it.
    with pytest.raises(ValueError):
        _ = ExtinctionEffect("odonnell94", ebv=0.1, frame="rest", backend="extinction")

    # For the fm07 model r_v is fixed, so we don't need to provide it. But we fail if we
    # try to override it with a different value.
    _ = ExtinctionEffect("fm07", ebv=0.1, frame="rest", backend="extinction")
    with pytest.raises(ValueError):
        _ = ExtinctionEffect("fm07", ebv=0.1, r_v=4.0, frame="rest", backend="extinction")


def test_bad_backend():
    """Test that we raise an error for an invalid backend."""
    with pytest.raises(ValueError):
        _ = ExtinctionEffect("CCM89", ebv=0.1, r_v=3.1, frame="rest", backend="invalid_backend")
    with pytest.raises(ValueError):
        _ = ExtinctionEffect("odonnell94", ebv=0.1, r_v=3.1, frame="rest")


def test_set_frame():
    """Test that correct frame is set"""
    ext = ExtinctionEffect("G23", ebv=0.1, r_v=3.1, frame="observer", backend="dust_extinction")
    assert ext.rest_frame is False

    with pytest.raises(ValueError):
        ExtinctionEffect("G23", ebv=0.1, r_v=3.1, frame="InvalidFrame", backend="dust_extinction")


def test_pickle_extinction_models(subtests):
    """Test that we can pickle and unpickle extinction effects regardless of backend."""
    for backend, model in zip(["dust_extinction", "extinction"], ["F99", "fm07"], strict=False):
        with subtests.test(backend=backend, model=model):
            model = ExtinctionEffect(model, r_v=3.1, frame="rest", ebv=0.1, backend=backend)

            # Compute the some sample fluxes before and after extinction.
            org_fluxes = np.full((10, 3), 1.0)
            wavelengths = np.array([7000.0, 5200.0, 4800.0])
            ext_fluxes_1 = model.apply(org_fluxes, wavelengths=wavelengths, ebv=0.1)
            assert ext_fluxes_1.shape == (10, 3)

            with tempfile.TemporaryDirectory() as tmpdir:
                file_path = Path(tmpdir) / f"test_{backend}_{model}_model.pkl"
                assert not file_path.exists()

                with open(file_path, "wb") as f:
                    pickle.dump(model, f)
                assert file_path.exists()

                with open(file_path, "rb") as f:
                    loaded_model = pickle.load(f)
                assert loaded_model is not None

                ext_fluxes_2 = loaded_model.apply(org_fluxes, wavelengths=wavelengths, ebv=0.1)
                assert ext_fluxes_2.shape == (10, 3)
                assert np.allclose(ext_fluxes_1, ext_fluxes_2)


def test_constant_extinction(subtests):
    """Test that we can create and sample an ExtinctionEffect object."""
    # Use given ebv values. Usually these would be computed from a dustmap,
    # based on (RA, dec).
    ebv_node = GivenValueList([0.1, 0.2, 0.3, 0.1, 0.2, 0.3])
    for backend, ext_name in zip(["dust_extinction", "extinction"], ["F99", "fm07"], strict=False):
        with subtests.test(backend=backend, model=ext_name):
            dust_effect = ExtinctionEffect(ext_name, r_v=3.1, frame="rest", ebv=ebv_node, backend=backend)

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

            # We check that all fluxes are reduced, and that higher ebv leads to
            # lower fluxes.
            assert fluxes.shape == (3, 5, 3)
            assert np.all(fluxes < 100.0)
            assert np.all(fluxes[0, :, :] > fluxes[1, :, :])
            assert np.all(fluxes[1, :, :] > fluxes[2, :, :])


def test_dustmap_chain(subtests):
    """Test that we can chain the dustmap computation and extinction effect."""
    for backend, ext_name in zip(["dust_extinction", "extinction"], ["F99", "fm07"], strict=False):
        with subtests.test(backend=backend, model=ext_name):
            model = ConstantSEDModel(
                brightness=100.0,
                ra=GivenValueList([45.0, 45.0, 45.0, 45.0, 45.0, 45.0]),
                dec=GivenValueList([20.0, -20.0, 10.0, 20.0, -20.0, 10.0]),
                redshift=0.0,
            )

            # Create a constant dust map for testing.
            dust_map = ConstantHemisphereDustMap(north_ebv=0.8, south_ebv=0.5)
            dust_map_node = DustmapWrapper(dust_map, ra=model.ra, dec=model.dec)

            # Create an extinction effect using the EBVs from that dust map.
            ext_effect = ExtinctionEffect(
                extinction_model=ext_name,
                ebv=dust_map_node,
                r_v=3.1,
                frame="rest",
                backend=backend,
            )
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
