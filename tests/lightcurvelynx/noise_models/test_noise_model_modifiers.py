import numpy as np
from lightcurvelynx.noise_models.base_noise_models import (
    ConstantFluxNoiseModel,
    PoissonFluxNoiseModel,
)
from lightcurvelynx.noise_models.noise_model_modifiers import (
    GivenLinearTransformFluxNoiseModel,
    LinearTransformFluxNoiseModel,
)

# Local helper class.
from lookup_only_obstable import LookupOnlyObsTable


def test_linear_transform_flux_noise_model_constant():
    """Test that LinearTransformFluxNoiseModel correctly scales and offsets the flux error
    computed by a ConstantFluxNoiseModel base noise model."""
    base_model = ConstantFluxNoiseModel(noise_level=1.0)
    scale_factor = 2.0
    flux_err_offset = 0.5
    model = LinearTransformFluxNoiseModel(
        base_noise_model=base_model,
        scale_factor=scale_factor,
        flux_err_offset=flux_err_offset,
    )

    bandflux = np.array([10.0, 20.0, 30.0])
    computed_flux_err = model.compute_flux_error(bandflux)
    assert np.allclose(
        computed_flux_err,
        scale_factor * np.full_like(bandflux, 1.0) + flux_err_offset,
    )


def test_linear_transform_flux_noise_model_poisson():
    """Test that LinearTransformFluxNoiseModel correctly scales and offsets the flux error
    computed by a PoissonFluxNoiseModel base noise model."""
    base_model = PoissonFluxNoiseModel()
    scale_factor = 1.25
    flux_err_offset = 0.1
    model = LinearTransformFluxNoiseModel(
        base_noise_model=base_model,
        scale_factor=scale_factor,
        flux_err_offset=flux_err_offset,
    )

    # Check that we copied over the required values from the base model.
    assert set(model.required_values) == {
        "exptime",
        "sky_bg_e",
        "psf_footprint",
        "zp",
        "read_noise",
        "dark_current",
    }

    bandflux = np.array([100.0, 200.0, 200.0])
    dummy_data = {
        "exptime": np.array([30.0, 35.0, 40.0]),
        "nexposure": np.array([1, 2, 1]),
        "sky_bg_e": np.array([100.0, 110.0, 120.0]),
        "psf_footprint": np.array([2.0, 2.5, 3.0]),
        "zp": np.array([25.0, 26.0, 27.0]),
        "read_noise": np.array([4.0, 4.5, 5.0]),
        "dark_current": np.array([0.01, 0.02, 0.03]),
        "zp_err_mag": np.array([0.001, 0.002, 0.003]),
    }
    obs_table = LookupOnlyObsTable(dummy_data)

    # Check that the model is compatible with our ObsTable.
    assert model.check_compatibility(obs_table, fail_on_incompatible=True)

    # Apply noise with the base model and the transformed model.
    base_flux, base_flux_err = base_model.apply_noise(
        bandflux,
        obs_table=obs_table,
        indices=np.array([0, 1, 2]),
        rng=np.random.default_rng(2024),
    )
    new_flux, new_flux_err = model.apply_noise(
        bandflux,
        obs_table=obs_table,
        indices=np.array([0, 1, 2]),
        rng=np.random.default_rng(2024),
    )

    assert np.allclose(new_flux_err, scale_factor * base_flux_err + flux_err_offset)
    assert not np.any(base_flux == new_flux)


def test_given_linear_transform_flux_noise_model_poisson():
    """Test that GivenLinearTransformFluxNoiseModel correctly scales and offsets the flux error
    computed by a PoissonFluxNoiseModel base noise model."""
    base_model = PoissonFluxNoiseModel()
    scale_factor = np.array([1.0, 2.0, 1.5])
    flux_err_offset = np.array([0.1, 0.5, 0.2])
    model = GivenLinearTransformFluxNoiseModel(
        base_noise_model=base_model,
        scale_factor_col="scale_factor",
        flux_err_offset_col="flux_err_offset",
    )

    # Check that we copied over the required values from the base model.
    assert set(model.required_values) == {
        "exptime",
        "sky_bg_e",
        "psf_footprint",
        "zp",
        "read_noise",
        "dark_current",
    }

    bandflux = np.array([100.0, 200.0, 200.0])
    dummy_data = {
        "exptime": np.array([30.0, 35.0, 40.0]),
        "nexposure": np.array([1, 2, 1]),
        "sky_bg_e": np.array([100.0, 110.0, 120.0]),
        "psf_footprint": np.array([2.0, 2.5, 3.0]),
        "zp": np.array([25.0, 26.0, 27.0]),
        "read_noise": np.array([4.0, 4.5, 5.0]),
        "dark_current": np.array([0.01, 0.02, 0.03]),
        "zp_err_mag": np.array([0.001, 0.002, 0.003]),
    }
    obs_table = LookupOnlyObsTable(dummy_data)

    # Check that the model is compatible with our ObsTable.
    assert model.check_compatibility(obs_table, fail_on_incompatible=True)

    # Apply noise with the base model and the transformed model. Without the scale_factor and flux_err_offset
    # columns in the obs_table, the model should use default values of 1.0 and 0.0, respectively.
    base_flux, base_flux_err = base_model.apply_noise(
        bandflux,
        obs_table=obs_table,
        indices=np.array([0, 1, 2]),
        rng=np.random.default_rng(2024),
    )
    new_flux, new_flux_err = model.apply_noise(
        bandflux,
        obs_table=obs_table,
        indices=np.array([0, 1, 2]),
        rng=np.random.default_rng(2024),
    )
    assert np.allclose(base_flux, new_flux)
    assert np.allclose(base_flux_err, new_flux_err)

    # We can now add the scale_factor and flux_err_offset columns to the obs_table and test that
    # the model correctly applies them.
    dummy_data["scale_factor"] = scale_factor
    dummy_data["flux_err_offset"] = flux_err_offset
    obs_table = LookupOnlyObsTable(dummy_data)
    new_flux, new_flux_err = model.apply_noise(
        bandflux,
        obs_table=obs_table,
        indices=np.array([0, 1, 2]),
        rng=np.random.default_rng(2024),
    )
    assert np.allclose(new_flux_err, scale_factor * base_flux_err + flux_err_offset)
    assert not np.any(base_flux == new_flux)
