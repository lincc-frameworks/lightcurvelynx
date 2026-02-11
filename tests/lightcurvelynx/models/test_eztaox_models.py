import numpy as np
import pytest
from lightcurvelynx.models.eztaox_models import EzTaoXWrapperModel

# We need to check if eztaox is installed before running the tests that depend on it
try:
    import eztaox.kernels.quasisep as ekq

    has_eztaox = True
except ImportError:
    ekq = None
    has_eztaox = False


@pytest.mark.skipif(not has_eztaox, reason="eztaox is not installed")
def test_eztaox_wrapper_model():
    """
    Test creating a simple EzTaoXWrapperModel and evaluating it at some
    times and wavelengths.
    """
    kernel = ekq.Exp(scale=2.2, sigma=0.02)

    source = EzTaoXWrapperModel(
        kernel,  # The kernel to use
        baseline_mags={"g": 20.0, "r": 19.5, "i": 19.0},
        band_list=["g", "r", "i"],
        log_kernel_param=[1.61, 0.0],
        log_amp_scale=[0.0, 0.0, 0.0],
        zero_mean=True,
        has_lag=False,
        ra=1.0,
        dec=1.0,
        redshift=0.1,
        node_label="source",
    )

    sampled_state = source.sample_parameters(num_samples=1)
    assert "source.eztaox_baseline_mag_g" in sampled_state
    assert "source.eztaox_baseline_mag_r" in sampled_state
    assert "source.eztaox_baseline_mag_i" in sampled_state
    assert "source.eztaox_baseline_mag_z" not in sampled_state
    assert "source.eztaox_log_kernel_param_0" in sampled_state
    assert "source.eztaox_log_kernel_param_1" in sampled_state
    assert "source.eztaox_log_kernel_param_2" not in sampled_state
    assert "source.eztaox_log_amp_scale_0" in sampled_state
    assert "source.eztaox_log_amp_scale_1" in sampled_state
    assert "source.eztaox_log_amp_scale_2" in sampled_state
    assert "source.eztaox_log_amp_scale_3" not in sampled_state

    # Evaluate the model at some times and wavelengths
    times = np.array([0.0, 1.0, 2.0, 3.0])
    filters = np.array(["g", "r", "g", "i"])
    bandfluxes = source.evaluate_bandfluxes(None, times, filters, sampled_state)
    assert bandfluxes.shape == (4,)
    assert np.all(bandfluxes > 0.0)

    # We can evaluate without a given graph state (a new one will be sampled).
    bandfluxes = source.evaluate_bandfluxes(None, times, filters, None)
    assert bandfluxes.shape == (4,)
    assert np.all(bandfluxes > 0.0)

    # We fail if the log_amp_scale length is incorrect
    with pytest.raises(ValueError):
        _ = EzTaoXWrapperModel(
            kernel,  # The kernel to use
            baseline_mags={"g": 20.0, "r": 19.5, "i": 19.0},
            band_list=["g", "r", "i"],
            log_kernel_param=[1.61, 0.0],
            log_amp_scale=[0.0, 0.0],  # Incorrect length
            zero_mean=True,
            has_lag=False,
            ra=1.0,
            dec=1.0,
            redshift=0.1,
            node_label="source",
        )


@pytest.mark.skipif(not has_eztaox, reason="eztaox is not installed")
def test_eztaox_wrapper_model_complex():
    """
    Test creating a simple EzTaoXWrapperModel with lag and non-zero mean, then
    evaluate it at some times and wavelengths
    """
    kernel = ekq.Exp(scale=2.2, sigma=0.02)

    source = EzTaoXWrapperModel(
        kernel,  # The kernel to use
        band_list=["g", "r", "i"],
        log_kernel_param=[1.61, 0.0],
        log_amp_scale=[0.0, 0.0, 0.0],
        zero_mean=False,
        mean_mag=[20.0, 19.5],
        has_lag=True,
        lag=[5.0, 10.0, 15.0],
        ra=1.0,
        dec=1.0,
        redshift=0.1,
        node_label="source",
    )

    sampled_state = source.sample_parameters(num_samples=1)

    # Evaluate the model at some times and wavelengths
    times = np.array([0.0, 1.0, 2.0, 3.0])
    filters = np.array(["g", "r", "g", "i"])
    bandfluxes = source.evaluate_bandfluxes(None, times, filters, sampled_state)
    assert bandfluxes.shape == (4,)
    assert np.all(bandfluxes > 0.0)

    # We fail with incorrect shapes of mean mag or lag.
    with pytest.raises(ValueError):
        _ = EzTaoXWrapperModel(
            kernel,  # The kernel to use
            band_list=["g", "r", "i"],
            log_kernel_param=[1.61, 0.0],
            log_amp_scale=[0.0, 0.0, 0.0],
            zero_mean=False,
            mean_mag=[20.0, 19.5, 18.0],  # Incorrect length
            has_lag=True,
            lag=[5.0, 10.0, 15.0],
            ra=1.0,
            dec=1.0,
            redshift=0.1,
            node_label="source",
        )
    with pytest.raises(ValueError):
        _ = EzTaoXWrapperModel(
            kernel,  # The kernel to use
            band_list=["g", "r", "i"],
            log_kernel_param=[1.61, 0.0],
            log_amp_scale=[0.0, 0.0, 0.0],
            zero_mean=False,
            mean_mag=[20.0, 19.5],
            has_lag=True,
            lag=[5.0, 10.0],  # Incorrect length
            ra=1.0,
            dec=1.0,
            redshift=0.1,
            node_label="source",
        )

    # We fail if we provide both baseline_mags and mean_mag
    with pytest.raises(ValueError):
        _ = EzTaoXWrapperModel(
            kernel,  # The kernel to use
            baseline_mags={"g": 20.0, "r": 19.5, "i": 19.0},
            band_list=["g", "r", "i"],
            log_kernel_param=[1.61, 0.0],
            log_amp_scale=[0.0, 0.0, 0.0],
            zero_mean=False,
            mean_mag=[20.0, 19.5],
            has_lag=False,
            ra=1.0,
            dec=1.0,
            redshift=0.1,
            node_label="source",
        )


@pytest.mark.skipif(not has_eztaox, reason="eztaox is not installed")
def test_eztaox_wrapper_model_seed():
    """
    Test creating a simple EzTaoXWrapperModel with a given seed.
    """
    kernel = ekq.Exp(scale=2.2, sigma=0.02)
    times = np.array([0.0, 1.0, 2.0, 3.0])
    filters = np.array(["g", "r", "g", "i"])

    # Create and evaluate a model.
    source1 = EzTaoXWrapperModel(
        kernel,  # The kernel to use
        baseline_mags={"g": 20.0, "r": 19.5, "i": 19.0},
        band_list=["g", "r", "i"],
        log_kernel_param=[1.61, 0.0],
        log_amp_scale=[0.0, 0.0, 0.0],
        zero_mean=True,
        has_lag=False,
        ra=1.0,
        dec=1.0,
        redshift=0.1,
        seed_param=42,
        node_label="source",
    )
    sampled_state1 = source1.sample_parameters(num_samples=1)
    bandfluxes1 = source1.evaluate_bandfluxes(None, times, filters, sampled_state1)

    # Create a second model with the same seed and evaluate it.
    source2 = EzTaoXWrapperModel(
        kernel,  # The kernel to use
        baseline_mags={"g": 20.0, "r": 19.5, "i": 19.0},
        band_list=["g", "r", "i"],
        log_kernel_param=[1.61, 0.0],
        log_amp_scale=[0.0, 0.0, 0.0],
        zero_mean=True,
        has_lag=False,
        ra=1.0,
        dec=1.0,
        redshift=0.1,
        seed_param=42,
        node_label="source",
    )
    sampled_state2 = source2.sample_parameters(num_samples=1)
    bandfluxes2 = source2.evaluate_bandfluxes(None, times, filters, sampled_state2)

    assert np.allclose(bandfluxes1, bandfluxes2)
