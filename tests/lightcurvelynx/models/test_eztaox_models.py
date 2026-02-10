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
def test_eztaox_wrapper_model(test_data_dir):
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
    print(sampled_state)
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
