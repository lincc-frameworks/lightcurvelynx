"""Test the MOSFiTWrapperModel."""

import numpy as np
import pytest
from citation_compass import find_in_citations
from lightcurvelynx.models.mosfit_models import MOSFiTWrapperModel
from lightcurvelynx.utils.extrapolate import ConstantPadding

TEST_PHASES = np.linspace(0.0, 100.0, 51)
TEST_WAVES = np.linspace(1000.0, 10000.0, 10)


class _ToyLynxSource:
    """A toy stand-in for mosfit.lynx.LynxSource with a light curve we control.

    The SED peaks at a phase of 20 days and falls off as 1 / wavelength, so tests can
    check the phase alignment and the wavelength dependence separately.

    Attributes
    ----------
    phases : numpy.ndarray
        The phase grid the source was built on (in days since explosion).
    wavelengths : numpy.ndarray
        The rest frame wavelength grid the source was built on (in angstroms).
    call_count : int
        The number of times compute_sed() has been called, used to check caching.
    """

    def __init__(self, phases=TEST_PHASES, wavelengths=TEST_WAVES):
        self.phases = np.asarray(phases, dtype=float)
        self.wavelengths = np.asarray(wavelengths, dtype=float)
        self.call_count = 0

    def compute_sed(self, parameters=None, **kwargs):
        """Return a toy rest frame SED in nJy at 10 pc."""
        self.call_count += 1
        parameters = parameters or {}
        if "bad_param" in parameters:
            raise ValueError("Not a free parameter of this model.")

        height = float(parameters.get("height", 1.0))
        width = float(parameters.get("width", 10.0))
        lightcurve = height * np.exp(-((self.phases - 20.0) ** 2) / width**2)
        return lightcurve[:, np.newaxis] * (1000.0 / self.wavelengths[np.newaxis, :])


class _BadShapeLynxSource(_ToyLynxSource):
    """A toy source that returns an SED on the wrong grid."""

    def compute_sed(self, parameters=None, **kwargs):
        """Return an SED with a shape the wrapper did not ask for."""
        return np.ones((len(self.phases) + 1, len(self.wavelengths)))


def _make_model(source=None, distance=10.0, t0=64350.0, **kwargs):
    """Create a MOSFiTWrapperModel backed by a toy source."""
    if source is None:
        source = _ToyLynxSource()
    return MOSFiTWrapperModel(
        "slsn",
        parameters={"height": 1000.0, "width": 10.0},
        phases=TEST_PHASES,
        wavelengths=TEST_WAVES,
        distance=distance,
        t0=t0,
        ra=0.0,
        dec=-10.0,
        source=source,
        node_label="source",
        **kwargs,
    )


def test_mosfit_wrapper_toy() -> None:
    """Test that we can create and evaluate a model backed by a toy source."""
    t0 = 64350.0
    source = _ToyLynxSource()
    model = _make_model(source=source, t0=t0)

    assert model.model_name == "slsn"
    assert set(model.source_param_names) == {"height", "width"}
    assert set(model.param_names) == {"height", "width"}

    state = model.sample_parameters()
    assert state["source"]["height"] == 1000.0
    assert state["source"]["width"] == 10.0
    assert state["source"]["t0"] == t0

    # The model is evaluated in phase space, so the peak sits 20 days after t0.
    times = np.array([0.0, 20.0, 60.0]) + t0
    fluxes = model.evaluate_sed(times, TEST_WAVES, graph_state=state)
    assert fluxes.shape == (3, len(TEST_WAVES))
    assert np.all(np.isfinite(fluxes))
    assert np.all(fluxes[0, :] < fluxes[1, :])
    assert np.all(fluxes[1, :] > fluxes[2, :])

    # The flux falls off with wavelength, as the toy source does.
    assert np.all(np.diff(fluxes[1, :]) < 0.0)

    # At a distance of 10 pc, the wrapper returns MOSFiT's own values unscaled. Compare
    # against the toy grid directly at a point that lies on the grid.
    grid_sed = _ToyLynxSource().compute_sed(parameters={"height": 1000.0, "width": 10.0})
    peak_index = np.searchsorted(TEST_PHASES, 20.0)
    assert np.allclose(fluxes[1, :], grid_sed[peak_index, :])


def test_mosfit_wrapper_bounds() -> None:
    """Test that the wrapper reports the bounds of its grid."""
    model = _make_model()
    assert model.minphase() == TEST_PHASES[0]
    assert model.maxphase() == TEST_PHASES[-1]
    assert model.minwave() == TEST_WAVES[0]
    assert model.maxwave() == TEST_WAVES[-1]


def test_mosfit_wrapper_distance_scaling() -> None:
    """Test that the SED is scaled from MOSFiT's 10 pc convention to the object's distance."""
    t0 = 64350.0
    times = np.array([20.0]) + t0

    near = _make_model(distance=10.0, t0=t0)
    far = _make_model(distance=100.0, t0=t0)

    near_flux = near.evaluate_sed(times, TEST_WAVES, graph_state=near.sample_parameters())
    far_flux = far.evaluate_sed(times, TEST_WAVES, graph_state=far.sample_parameters())

    assert np.all(near_flux > 0.0)
    assert np.allclose(far_flux, near_flux / 100.0)


def test_mosfit_wrapper_outside_grid() -> None:
    """Test that times outside the phase grid use the extrapolation model."""
    t0 = 64350.0
    state_times = np.array([-50.0, 20.0, 500.0]) + t0

    # The default time extrapolation is zero padding.
    model = _make_model(t0=t0)
    fluxes = model.evaluate_sed(state_times, TEST_WAVES, graph_state=model.sample_parameters())
    assert np.all(fluxes[0, :] == 0.0)
    assert np.all(fluxes[1, :] > 0.0)
    assert np.all(fluxes[2, :] == 0.0)

    # A different extrapolation model is used if the caller provides one.
    padded = _make_model(t0=t0, time_extrapolation=ConstantPadding(100.0))
    padded_fluxes = padded.evaluate_sed(state_times, TEST_WAVES, graph_state=padded.sample_parameters())
    assert np.all(padded_fluxes[0, :] == 100.0)
    assert np.all(padded_fluxes[2, :] == 100.0)


def test_mosfit_wrapper_outside_wavelength_grid() -> None:
    """Test that wavelengths outside the model's grid return zero with a warning."""
    t0 = 64350.0
    model = _make_model(t0=t0)
    state = model.sample_parameters()

    times = np.array([20.0]) + t0
    waves = np.array([500.0, 5000.0, 50000.0])
    with pytest.warns(UserWarning):
        fluxes = model.evaluate_sed(times, waves, graph_state=state)

    assert fluxes[0, 0] == 0.0
    assert fluxes[0, 1] > 0.0
    assert fluxes[0, 2] == 0.0


def test_mosfit_wrapper_caches_evaluations() -> None:
    """Test that repeated evaluations with the same parameters only run MOSFiT once."""
    t0 = 64350.0
    source = _ToyLynxSource()
    model = _make_model(source=source, t0=t0)
    state = model.sample_parameters()

    times = np.array([10.0, 20.0, 30.0]) + t0
    first = model.evaluate_sed(times, TEST_WAVES, graph_state=state)
    assert source.call_count == 1

    second = model.evaluate_sed(times, TEST_WAVES, graph_state=state)
    assert source.call_count == 1
    assert np.allclose(first, second)

    # Changing a parameter value invalidates the cache.
    other = _make_model(source=source, t0=t0)
    other.set_parameter("height", 2000.0)
    other_flux = other.evaluate_sed(times, TEST_WAVES, graph_state=other.sample_parameters())
    assert source.call_count == 2
    assert np.allclose(other_flux, 2.0 * first)


def test_mosfit_wrapper_citations() -> None:
    """Test that we can recover the MOSFiT citation."""
    citations = find_in_citations("MOSFiTWrapperModel")
    assert len(citations) >= 1
    for citation in citations:
        assert "https://ui.adsabs.harvard.edu/abs/2018ApJS..236....6G/abstract" in citation


def test_mosfit_wrapper_invalid_setup() -> None:
    """Test that invalid wrapper configurations are rejected."""
    # A parameter given both in the dictionary and as a keyword argument.
    with pytest.raises(ValueError):
        _ = MOSFiTWrapperModel(
            "slsn",
            parameters={"redshift": 0.05},
            redshift=0.1,
            distance=1.0e8,
            source=_ToyLynxSource(),
        )

    # No way to determine the luminosity distance.
    with pytest.raises(ValueError):
        _ = MOSFiTWrapperModel("slsn", parameters={"height": 1.0}, source=_ToyLynxSource())

    # Grids that are too short, not strictly increasing, or not 1-d.
    for bad_grid in ([10.0], [10.0, 5.0, 20.0], np.ones((2, 2))):
        with pytest.raises(ValueError):
            _ = MOSFiTWrapperModel("slsn", phases=bad_grid, distance=10.0, source=_ToyLynxSource())
        with pytest.raises(ValueError):
            _ = MOSFiTWrapperModel("slsn", wavelengths=bad_grid, distance=10.0, source=_ToyLynxSource())


def test_mosfit_wrapper_evaluation_errors() -> None:
    """Test that errors from MOSFiT and bad output shapes are reported clearly."""
    t0 = 64350.0
    times = np.array([20.0]) + t0

    # An error raised inside MOSFiT is wrapped with guidance.
    model = MOSFiTWrapperModel(
        "slsn",
        parameters={"bad_param": 1.0},
        phases=TEST_PHASES,
        wavelengths=TEST_WAVES,
        distance=10.0,
        t0=t0,
        source=_ToyLynxSource(),
    )
    with pytest.raises(RuntimeError, match="Error evaluating the MOSFiT model"):
        _ = model.evaluate_sed(times, TEST_WAVES, graph_state=model.sample_parameters())

    # An SED on a grid we did not ask for is rejected rather than silently reshaped.
    bad_shape = _make_model(source=_BadShapeLynxSource(), t0=t0)
    with pytest.raises(ValueError, match="returned an SED of shape"):
        _ = bad_shape.evaluate_sed(times, TEST_WAVES, graph_state=bad_shape.sample_parameters())


def test_mosfit_wrapper_invalid_distance() -> None:
    """Test that a non-positive luminosity distance is rejected at evaluation time."""
    t0 = 64350.0
    model = _make_model(distance=10.0, t0=t0)
    state = model.sample_parameters()
    state.set("source", "distance", -1.0, force_copy=True)

    with pytest.raises(ValueError, match="invalid luminosity distance"):
        _ = model.evaluate_sed(np.array([20.0]) + t0, TEST_WAVES, graph_state=state)


def test_mosfit_wrapper_real_source(tmp_path, monkeypatch) -> None:
    """Test that we can build and evaluate a real MOSFiT model end to end."""
    pytest.importorskip(
        "mosfit.lynx",
        reason="Requires MOSFiT >= 2.1, the first release providing the mosfit.lynx module.",
    )

    # MOSFiT copies its `modules` directory into the working directory when a model is
    # built, so run from a temporary directory to keep the repository clean.
    monkeypatch.chdir(tmp_path)

    t0 = 64350.0
    phases = np.linspace(0.0, 100.0, 21)
    waves = np.linspace(2000.0, 12000.0, 11)
    model = MOSFiTWrapperModel(
        "slsn",
        phases=phases,
        wavelengths=waves,
        distance=10.0,
        t0=t0,
        node_label="source",
    )

    state = model.sample_parameters()
    times = np.linspace(0.0, 100.0, 11) + t0
    fluxes = model.evaluate_sed(times, waves, graph_state=state)

    assert fluxes.shape == (len(times), len(waves))
    assert np.all(np.isfinite(fluxes))
    assert np.all(fluxes >= 0.0)
    assert np.any(fluxes > 0.0)


def test_mosfit_wrapper_applies_redshift() -> None:
    """Test that LightCurveLynx, and not MOSFiT, applies the redshift.

    MOSFiT's Lynx output is a rest frame SED at 10 pc, so the wrapper leaves
    ``apply_redshift`` on and lets the superclass do the time dilation, the wavelength
    shift and the distance dimming.
    """
    t0 = 64350.0
    redshift = 1.0
    model = MOSFiTWrapperModel(
        "slsn",
        parameters={"height": 1000.0, "width": 10.0},
        phases=TEST_PHASES,
        wavelengths=TEST_WAVES,
        redshift=redshift,
        cosmology="Planck18",
        t0=t0,
        source=_ToyLynxSource(),
        node_label="source",
    )
    assert model.apply_redshift

    state = model.sample_parameters()
    distance = state["source"]["distance"]
    assert distance > 1.0e8  # Roughly 6.8 Gpc at z=1.

    # The toy source peaks 20 days after the explosion in the rest frame, so the
    # observed peak is time dilated to 20 * (1 + z) days after t0.
    times = np.array([20.0, 40.0]) + t0
    waves = np.array([4000.0, 8000.0])
    fluxes = model.evaluate_sed(times, waves, graph_state=state)
    assert fluxes[1, 0] > fluxes[0, 0]

    # The flux is dimmed by the luminosity distance relative to MOSFiT's 10 pc output.
    grid_sed = _ToyLynxSource().compute_sed(parameters={"height": 1000.0, "width": 10.0})
    assert np.all(fluxes < grid_sed.max())
