import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord, SkyOffsetFrame
from numpy.testing import assert_allclose, assert_array_equal

from lightcurvelynx.astro_utils.passbands import Passband, PassbandGroup
from lightcurvelynx.effects.basic_effects import ScaleFluxEffect
from lightcurvelynx.effects.effect_model import EffectModel
from lightcurvelynx.math_nodes.basic_math_node import BasicMathNode
from lightcurvelynx.math_nodes.given_sampler import GivenValueList
from lightcurvelynx.math_nodes.np_random import NumpyRandomFunc
from lightcurvelynx.models.basic_models import LinearTimeModel, LinearWavelengthModel
from lightcurvelynx.models.physical_model import BandfluxModel
from lightcurvelynx.models.strong_lens_model import (
    ResolvedStrongLensModel,
    _LensImageExpansionNode,
    _LensImagePositionNode,
)

# Two systems with fixed width 4. System 0 has 3 active images, given out of arrival order
# and with a delay tie. System 1 has 2 active images, and its 0.0 padding is smaller than its
# active delays, so it must not be mistaken for the earliest image.
IMAGE_X = np.array([[0.5, -0.4, 0.1, np.nan], [1.0, -1.0, 0.0, 0.0]])
IMAGE_Y = np.array([[0.2, 0.3, -0.6, np.nan], [0.0, 0.5, 0.0, 0.0]])
MAGNIFICATIONS = np.array([[2.0, 4.0, 1.5, np.nan], [3.0, 0.5, 0.0, 0.0]])
TIME_DELAYS = np.array([[30.0, 10.0, 30.0, np.nan], [5.0, 12.0, 0.0, 0.0]])
NUM_IMAGES = np.array([3, 2])

# Expected per-image rows: arrival order (stable for ties), delays normalized to the earliest image.
EXPECTED_SYSTEM_ID = np.array([0, 0, 0, 1, 1])
EXPECTED_IMAGE_ID = np.array([0, 1, 2, 0, 1])
EXPECTED_IMAGE_X = np.array([-0.4, 0.5, 0.1, 1.0, -1.0])
EXPECTED_IMAGE_Y = np.array([0.3, 0.2, -0.6, 0.0, 0.5])
EXPECTED_MAGNIFICATION = np.array([4.0, 2.0, 1.5, 3.0, 0.5])
EXPECTED_TIME_DELAY = np.array([0.0, 20.0, 20.0, 0.0, 7.0])


def _given(values):
    """Return a stateless GivenValueList so repeated or batched sampling is reproducible."""
    return GivenValueList(values, stateful=False)


def _two_system_image_inputs():
    """Return the image-array setters for the two test systems."""
    return {
        "image_x": _given(IMAGE_X),
        "image_y": _given(IMAGE_Y),
        "macro_magnifications": _given(MAGNIFICATIONS),
        "time_delays": _given(TIME_DELAYS),
        "num_images": _given(NUM_IMAGES),
    }


def test_expansion_sorts_normalizes_and_drops_padding():
    """Test that each system expands into its active images in arrival order."""
    node = _LensImageExpansionNode(**_two_system_image_inputs(), node_label="images")
    state = node.sample_parameters(num_samples=2)

    assert state.num_samples == 5
    assert_array_equal(state["images"]["system_id"], EXPECTED_SYSTEM_ID)
    assert_array_equal(state["images"]["image_id"], EXPECTED_IMAGE_ID)
    assert_array_equal(state["images"]["image_x"], EXPECTED_IMAGE_X)
    assert_array_equal(state["images"]["image_y"], EXPECTED_IMAGE_Y)
    assert_array_equal(state["images"]["macro_magnification"], EXPECTED_MAGNIFICATION)
    assert_array_equal(state["images"]["time_delay"], EXPECTED_TIME_DELAY)


def test_expansion_repeats_per_system_values():
    """Test that values sampled before expansion are repeated once per image."""
    inputs = _two_system_image_inputs()
    state = _LensImageExpansionNode(**inputs, node_label="images").sample_parameters(num_samples=2)
    rows = state[inputs["image_x"].node_string]["function_node_result"]
    assert_array_equal(rows, IMAGE_X[EXPECTED_SYSTEM_ID])


def test_expansion_uses_full_width_without_num_images():
    """Test that all entries are active when num_images is not given."""
    node = _LensImageExpansionNode(
        image_x=[1.0, 2.0],
        image_y=[0.0, 0.0],
        macro_magnifications=[1.0, 2.0],
        time_delays=[3.0, 1.0],
        node_label="images",
    )
    state = node.sample_parameters(num_samples=1)
    assert state.num_samples == 2
    assert_array_equal(state["images"]["image_id"], [0, 1])
    assert_array_equal(state["images"]["image_x"], [2.0, 1.0])
    assert_array_equal(state["images"]["time_delay"], [0.0, 2.0])


def test_expansion_single_image_stores_scalars():
    """Test that a one-image system becomes a single row of scalar values."""
    node = _LensImageExpansionNode(
        image_x=[0.7, np.nan],
        image_y=[0.1, np.nan],
        macro_magnifications=[1.5, np.nan],
        time_delays=[4.0, np.nan],
        num_images=1,
        node_label="images",
    )
    state = node.sample_parameters(num_samples=1)
    assert state.num_samples == 1
    expected = {
        "system_id": 0,
        "image_id": 0,
        "image_x": 0.7,
        "image_y": 0.1,
        "macro_magnification": 1.5,
        "time_delay": 0.0,
    }
    for name, value in expected.items():
        assert np.isscalar(state["images"][name])
        assert state["images"][name] == value


def test_expansion_system_id_includes_sample_offset():
    """Test that system_id continues from the state's sample offset (batched simulation)."""
    node = _LensImageExpansionNode(
        image_x=[0.0, 1.0],
        image_y=[0.0, 1.0],
        macro_magnifications=[1.0, 1.0],
        time_delays=[0.0, 1.0],
        node_label="images",
    )
    state = node.sample_parameters(num_samples=2, sample_offset=10)
    assert_array_equal(state["images"]["system_id"], [10, 10, 11, 11])
    assert_array_equal(state["images"]["image_id"], [0, 1, 0, 1])


def test_expansion_accepts_integer_valued_float_counts():
    """Test that float-typed image counts such as 3.0 are used as counts."""
    inputs = _two_system_image_inputs() | {"num_images": _given(np.array([3.0, 2.0]))}
    state = _LensImageExpansionNode(**inputs, node_label="images").sample_parameters(num_samples=2)
    assert_array_equal(state["images"]["system_id"], EXPECTED_SYSTEM_ID)
    assert_array_equal(state["images"]["time_delay"], EXPECTED_TIME_DELAY)


POSITION_NAMES = ["source_ra", "source_dec", "source_x", "source_y", "image_x", "image_y"]
POSITION_CASES = [
    (10.0, 0.0, 2.0, -1.5, -1.0, 2.5),
    (0.0001, 30.0, -1.0, 0.5, -3.0, 1.0),  # image RA wraps below 0 to ~359.9995
    (45.0, 89.9, 2.0, -1.5, -1.2, 2.8),
    (200.0, -89.9, -2.5, 1.0, 1.5, -2.0),
]


def _lens_frame_offsets(ra, dec, lens_ra, lens_dec):
    """Return (east, north) offsets in arcsec of (ra, dec) in the lens-centred frame."""
    frame = SkyOffsetFrame(origin=SkyCoord(lens_ra * u.deg, lens_dec * u.deg))
    offset = SkyCoord(ra * u.deg, dec * u.deg).transform_to(frame)
    return offset.lon.wrap_at(180 * u.deg).to_value(u.arcsec), offset.lat.to_value(u.arcsec)


@pytest.mark.parametrize(", ".join(POSITION_NAMES), POSITION_CASES)
def test_position_node_recovers_lens_frame_offsets(
    source_ra, source_dec, source_x, source_y, image_x, image_y
):
    """Test that the source and image sit at their given offsets in the derived lens frame."""
    node = _LensImagePositionNode(
        source_ra=source_ra,
        source_dec=source_dec,
        source_x=source_x,
        source_y=source_y,
        image_x=image_x,
        image_y=image_y,
        node_label="positions",
    )
    params = node.sample_parameters(num_samples=1)["positions"]

    sx, sy = _lens_frame_offsets(source_ra, source_dec, params["lens_ra"], params["lens_dec"])
    assert sx == pytest.approx(source_x, abs=1e-6)
    assert sy == pytest.approx(source_y, abs=1e-6)
    ix, iy = _lens_frame_offsets(params["ra"], params["dec"], params["lens_ra"], params["lens_dec"])
    assert ix == pytest.approx(image_x, abs=1e-6)
    assert iy == pytest.approx(image_y, abs=1e-6)
    assert 0.0 <= params["ra"] < 360.0


def test_position_node_matches_scalar_rows():
    """Test that the vectorized multi-row result matches row-by-row results."""
    cases = np.array(POSITION_CASES)
    inputs = {name: _given(cases[:, i]) for i, name in enumerate(POSITION_NAMES)}
    state = _LensImagePositionNode(**inputs, node_label="positions").sample_parameters(num_samples=len(cases))
    for row, case in enumerate(POSITION_CASES):
        single = _LensImagePositionNode(**dict(zip(POSITION_NAMES, case, strict=True)), node_label="row")
        expected = single.sample_parameters(num_samples=1)["row"]
        for name in ("lens_ra", "lens_dec", "ra", "dec"):
            assert state["positions"][name][row] == pytest.approx(expected[name], abs=1e-9)


SOURCE_RA = np.array([10.0, 250.0])
SOURCE_DEC = np.array([-30.0, 45.0])
SOURCE_T0 = np.array([100.0, 200.0])
SOURCE_X = np.array([0.3, -0.2])
SOURCE_Y = np.array([0.1, 0.4])


def _make_source(**overrides):
    """Return a LinearTimeModel source: flux = 10 + 2 * max(0, t - t0), flat in wavelength."""
    params = {
        "linear_base": 10.0,
        "linear_scale": 2.0,
        "t0": _given(SOURCE_T0),
        "ra": _given(SOURCE_RA),
        "dec": _given(SOURCE_DEC),
        "redshift": 0.0,
        "node_label": "source",
    }
    return LinearTimeModel(**(params | overrides))


def _make_lens(source=None, **image_overrides):
    """Return a ResolvedStrongLensModel over the two test systems."""
    source = _make_source() if source is None else source
    return ResolvedStrongLensModel(
        source,
        source_x=_given(SOURCE_X),
        source_y=_given(SOURCE_Y),
        node_label="lens",
        **(_two_system_image_inputs() | image_overrides),
    )


def test_resolved_lens_rows_and_parameters():
    """Test the per-image parameters exposed by the wrapper."""
    state = _make_lens().sample_parameters(num_samples=2)
    assert state.num_samples == 5
    params = state["lens"]
    ids = EXPECTED_SYSTEM_ID
    assert_array_equal(params["system_id"], ids)
    assert_array_equal(params["image_id"], EXPECTED_IMAGE_ID)
    assert_array_equal(params["image_x"], EXPECTED_IMAGE_X)
    assert_array_equal(params["image_y"], EXPECTED_IMAGE_Y)
    assert_array_equal(params["macro_magnification"], EXPECTED_MAGNIFICATION)
    assert_array_equal(params["time_delay"], EXPECTED_TIME_DELAY)
    assert_array_equal(params["source_x"], SOURCE_X[ids])
    assert_array_equal(params["source_y"], SOURCE_Y[ids])
    assert_allclose(params["t0"], SOURCE_T0[ids] + EXPECTED_TIME_DELAY)
    assert_array_equal(params["redshift"], 0.0)

    ix, iy = _lens_frame_offsets(params["ra"], params["dec"], params["lens_ra"], params["lens_dec"])
    assert_allclose(ix, EXPECTED_IMAGE_X, atol=1e-6)
    assert_allclose(iy, EXPECTED_IMAGE_Y, atol=1e-6)
    sx, sy = _lens_frame_offsets(SOURCE_RA[ids], SOURCE_DEC[ids], params["lens_ra"], params["lens_dec"])
    assert_allclose(sx, SOURCE_X[ids], atol=1e-6)
    assert_allclose(sy, SOURCE_Y[ids], atol=1e-6)


def test_source_parameters_are_shared_per_system():
    """Test that a random source parameter is drawn once per system."""
    source = _make_source(linear_base=NumpyRandomFunc("uniform", low=0.0, high=1.0))
    state = _make_lens(source).sample_parameters(num_samples=2, rng_info=np.random.default_rng(0))
    base = state["source"]["linear_base"]
    assert base[0] == base[1] == base[2]
    assert base[3] == base[4]
    assert base[0] != base[3]


def test_random_source_t0_is_shared_and_delayed_per_image():
    """Test that each image's t0 is the shared source t0 plus its delay."""
    source = _make_source(t0=NumpyRandomFunc("uniform", low=0.0, high=100.0))
    state = _make_lens(source).sample_parameters(num_samples=2, rng_info=np.random.default_rng(1))
    source_t0 = state["source"]["t0"]
    assert source_t0[0] == source_t0[2]
    assert source_t0[3] == source_t0[4]
    assert_allclose(state["lens"]["t0"], source_t0 + EXPECTED_TIME_DELAY)


def test_wrapper_observer_frame_effect_parameters_are_per_image():
    """Test that an observer-frame effect added to the wrapper draws per image."""
    lens = _make_lens()
    flux_scale = NumpyRandomFunc("uniform", low=1.0, high=2.0)
    lens.add_effect(ScaleFluxEffect(flux_scale=flux_scale, rest_frame=False))
    state = lens.sample_parameters(num_samples=2, rng_info=np.random.default_rng(2))
    assert len(np.unique(state["lens"]["flux_scale"])) == 5


def test_wrapper_rest_frame_effect_is_shared_per_system():
    """Test that a rest-frame effect added to the wrapper goes to the source and is shared."""
    lens = _make_lens()
    flux_scale = NumpyRandomFunc("uniform", low=1.0, high=2.0)
    lens.add_effect(ScaleFluxEffect(flux_scale=flux_scale, rest_frame=True))
    state = lens.sample_parameters(num_samples=2, rng_info=np.random.default_rng(3))
    scale = state["source"]["flux_scale"]
    assert scale[0] == scale[1] == scale[2]
    assert scale[3] == scale[4]
    assert scale[0] != scale[3]


def test_rest_frame_effect_linked_to_image_position_varies_per_image():
    """Test that a rest-frame effect parameter linked to the image RA is per image."""
    lens = _make_lens()
    flux_scale = BasicMathNode("1.0 + ra / 360.0", ra=lens.ra)
    lens.add_effect(ScaleFluxEffect(flux_scale=flux_scale, rest_frame=True))
    state = lens.sample_parameters(num_samples=2)
    assert_allclose(state["source"]["flux_scale"], 1.0 + state["lens"]["ra"] / 360.0)


def test_source_model_is_not_modified():
    """Test that wrapping leaves the source's setters and effects untouched."""
    source = _make_source()
    setters_before = dict(source.setters)
    _make_lens(source)
    assert list(source.setters) == list(setters_before)
    assert all(source.setters[name] is setter for name, setter in setters_before.items())
    assert source.rest_frame_effects == []
    assert source.obs_frame_effects == []


TIMES = np.array([90.0, 105.0, 130.0, 210.0])
WAVES = np.array([4000.0, 5000.0])


def _expected_linear_flux(times):
    """Return the magnified, delayed LinearTimeModel flux for each expected image row, shape (5, T)."""
    image_t0 = SOURCE_T0[EXPECTED_SYSTEM_ID] + EXPECTED_TIME_DELAY
    linear = 10.0 + 2.0 * np.maximum(0.0, times[np.newaxis, :] - image_t0[:, np.newaxis])
    return EXPECTED_MAGNIFICATION[:, np.newaxis] * linear


class _AddTimeEffect(EffectModel):
    """Observer-frame test effect that adds the observation time to the flux."""

    def __init__(self):
        super().__init__(rest_frame=False)

    def apply(self, flux_density, times=None, wavelengths=None, **kwargs):
        return flux_density + times[:, np.newaxis]

    def apply_bandflux(self, bandfluxes, *, times=None, filters=None, **kwargs):
        return bandfluxes + times


class _LinearBandfluxSource(BandfluxModel):
    """Bandflux test source: flux = linear_base + linear_scale * max(0, t - t0), plus 1.0 in "r"."""

    def __init__(self, linear_base, linear_scale, **kwargs):
        super().__init__(**kwargs)
        self.add_parameter("linear_base", linear_base)
        self.add_parameter("linear_scale", linear_scale)

    def compute_bandflux(self, times, filter, state):
        params = self.get_local_params(state)
        offset = 1.0 if filter == "r" else 0.0
        return params["linear_base"] + params["linear_scale"] * np.maximum(0.0, times - params["t0"]) + offset


def _make_bandflux_source():
    """Return a bandflux source with the same parameters as _make_source()."""
    return _LinearBandfluxSource(
        linear_base=10.0,
        linear_scale=2.0,
        t0=_given(SOURCE_T0),
        ra=_given(SOURCE_RA),
        dec=_given(SOURCE_DEC),
        node_label="source",
    )


def test_evaluate_sed_matches_magnified_delayed_source():
    """Test that each image row is the magnified source at t - delay."""
    lens = _make_lens()
    state = lens.sample_parameters(num_samples=2)
    flux = lens.evaluate_sed(TIMES, WAVES, state)
    assert flux.shape == (5, 4, 2)
    assert_allclose(flux, np.repeat(_expected_linear_flux(TIMES)[:, :, np.newaxis], 2, axis=2))


def test_wrapper_sed_effect_sees_observed_times():
    """Test that wrapper observer-frame effects get the unshifted observation times."""
    lens = _make_lens()
    lens.add_effect(_AddTimeEffect())
    state = lens.sample_parameters(num_samples=2)
    flux = lens.evaluate_sed(TIMES, WAVES, state)
    assert_allclose(flux[:, :, 0], _expected_linear_flux(TIMES) + TIMES[np.newaxis, :])


def test_redshift_is_applied_once():
    """Test that a redshifted source is converted once, by the source itself."""
    source = _make_source(redshift=1.0)
    lens = _make_lens(source)
    state = lens.sample_parameters(num_samples=2)
    flux = lens.evaluate_sed(TIMES, WAVES, state)
    for row, row_state in enumerate(state):
        expected = EXPECTED_MAGNIFICATION[row] * source.evaluate_sed(
            TIMES - EXPECTED_TIME_DELAY[row], WAVES, row_state
        )
        assert_allclose(flux[row], expected)


def test_evaluate_sed_rejects_bandflux_source():
    """Test that SED evaluation is refused for a bandflux-only source."""
    lens = _make_lens(_make_bandflux_source())
    state = lens.sample_parameters(num_samples=2)
    with pytest.raises(TypeError):
        lens.evaluate_sed(TIMES, WAVES, state)


def test_minwave_maxwave_forward_to_source():
    """Test that wavelength bounds come from the source."""
    source = LinearWavelengthModel(
        linear_base=1.0, linear_scale=0.0, min_wave=1000.0, max_wave=9000.0, t0=0.0, ra=10.0, dec=-30.0
    )
    lens = ResolvedStrongLensModel(
        source,
        source_x=0.0,
        source_y=0.0,
        image_x=[0.5, -0.5],
        image_y=[0.0, 0.0],
        macro_magnifications=[1.0, 1.0],
        time_delays=[0.0, 1.0],
    )
    assert lens.minwave() == 1000.0
    assert lens.maxwave() == 9000.0


def test_single_image_system_evaluates_as_one_row():
    """Test that a one-image system evaluates to a (T, W) SED."""
    source = LinearTimeModel(linear_base=10.0, linear_scale=2.0, t0=100.0, ra=10.0, dec=-30.0, redshift=0.0)
    lens = ResolvedStrongLensModel(
        source,
        source_x=0.0,
        source_y=0.0,
        image_x=[0.4, np.nan],
        image_y=[0.0, np.nan],
        macro_magnifications=[3.0, np.nan],
        time_delays=[5.0, np.nan],
        num_images=1,
    )
    state = lens.sample_parameters(num_samples=1)
    assert state.num_samples == 1
    flux = lens.evaluate_sed(TIMES, WAVES, state)
    assert flux.shape == (4, 2)
    assert_allclose(flux[:, 0], 3.0 * (10.0 + 2.0 * np.maximum(0.0, TIMES - 100.0)))


FILTERS = np.array(["g", "r", "g", "r"])


def _toy_passbands():
    """Return a two-filter toy passband group in Angstroms."""
    return PassbandGroup(
        [
            Passband(np.array([[4000.0, 0.5], [5000.0, 0.75], [6000.0, 0.25]]), "toy", "g"),
            Passband(np.array([[5000.0, 0.25], [6000.0, 0.5], [7000.0, 0.75]]), "toy", "r"),
        ]
    )


class _WavelengthTiltEffect(EffectModel):
    """Observer-frame test effect with no apply_bandflux: multiplies flux by wavelength / 5000 Å."""

    def __init__(self):
        super().__init__(rest_frame=False)

    def apply(self, flux_density, times=None, wavelengths=None, **kwargs):
        return flux_density * (wavelengths[np.newaxis, :] / 5000.0)


def test_evaluate_bandfluxes_sed_source():
    """Test bandfluxes for an SED source (a flat SED integrates to its own value)."""
    lens = _make_lens()
    state = lens.sample_parameters(num_samples=2)
    fluxes = lens.evaluate_bandfluxes(_toy_passbands(), TIMES, FILTERS, state)
    assert fluxes.shape == (5, 4)
    assert_allclose(fluxes, _expected_linear_flux(TIMES))


def test_sed_only_wrapper_effect_works_on_bandflux_path():
    """Test that a wavelength-dependent wrapper effect is applied to the SED before integration."""
    lens = _make_lens()
    lens.add_effect(_WavelengthTiltEffect())
    state = lens.sample_parameters(num_samples=2)
    passbands = _toy_passbands()
    fluxes = lens.evaluate_bandfluxes(passbands, TIMES, FILTERS, state)
    expected = _expected_linear_flux(TIMES)
    for col, filter_name in enumerate(FILTERS):
        passband = passbands[filter_name]
        tilt = passband.fluxes_to_bandflux((passband.waves / 5000.0)[np.newaxis, :])[0]
        assert_allclose(fluxes[:, col], expected[:, col] * tilt)


def test_evaluate_bandfluxes_bandflux_source():
    """Test a bandflux source: magnified at t - delay, wrapper effect at observed times."""
    lens = _make_lens(_make_bandflux_source())
    lens.add_effect(_AddTimeEffect())
    state = lens.sample_parameters(num_samples=2)
    fluxes = lens.evaluate_bandfluxes(_toy_passbands(), TIMES, FILTERS, state)
    filter_offset = np.where(FILTERS == "r", 1.0, 0.0)
    expected = (
        _expected_linear_flux(TIMES)
        + EXPECTED_MAGNIFICATION[:, np.newaxis] * filter_offset
        + TIMES[np.newaxis, :]
    )
    assert_allclose(fluxes, expected)
