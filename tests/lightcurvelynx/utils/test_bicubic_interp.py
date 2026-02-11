import os

import jax.numpy as jnp
import numpy as np
import pytest
from lightcurvelynx.utils.bicubic_interp import (
    BicubicAxis,
    BicubicInterpolator,
    _kernel_value,
    expand_to_cross_products,
)


def test_expand_to_cross_products():
    """Test that we can create cross product arrays."""
    x_vals = np.arange(5, 10)
    y_vals = np.arange(-3, 3)
    num_x = len(x_vals)
    num_y = len(y_vals)

    # Manually fill in the expected results.
    expected_x = np.zeros(num_x * num_y)
    expected_y = np.zeros(num_x * num_y)
    count = 0
    for row in range(num_x):
        for col in range(num_y):
            expected_x[count] = x_vals[row]
            expected_y[count] = y_vals[col]
            count += 1

    all_x, all_y = expand_to_cross_products(x_vals, y_vals)
    assert np.array_equal(all_x, expected_x)
    assert np.array_equal(all_y, expected_y)


def test_bicubic_range():
    """Test that we can create and query a BicubicAxis object."""
    xvals = BicubicAxis([0.1 * i for i in range(100)])
    assert xvals.min_val == 0.0
    assert xvals.max_val == 9.9
    assert xvals.num_vals == 100
    assert xvals.step_size == pytest.approx(0.1)

    # Depending on the version of JAX and whether other tests have set the precision,
    # the step size might be slightly different than 0.1, which gives slightly different
    # in some cases. So we force step size to exactly 0.1.
    xvals.step_size = 0.1

    # Test that we can run out_of_bounds
    values = [-1.0, 2.0, 11.0, 5.0, -0.1, 3.5, 4.0]
    expected = [True, False, True, False, True, False, False]
    is_out = xvals.out_of_bounds(values)
    assert np.array_equal(is_out, expected)

    # Test that we can run find_indices
    ix = xvals.find_indices(values)
    assert np.array_equal(ix, [0, 20, 98, 50, 0, 35, 40])

    # Run a larger find_indices search.
    n_x = len(xvals)
    x_q = jnp.asarray([0.04 * i - 0.1 for i in range(400)])
    ix = xvals.find_indices(x_q)
    assert jnp.all(ix >= 0)
    assert jnp.all(ix <= n_x - 2)

    # Test that the indices are correct (using a small tolerance to account for
    # floating point issues).
    assert jnp.all((ix == n_x - 2) | (x_q < xvals.values[ix + 1] + 1e-6))
    assert jnp.all((ix == 0) | (xvals.values[ix] - 1e-6 <= x_q))

    # Test that we fail for invalid BicubicAxis range objects.
    with pytest.raises(ValueError):
        _ = BicubicAxis([0.0, 0.1])  # Too few points
    with pytest.raises(ValueError):
        _ = BicubicAxis([0.0, 0.2, 0.1])  # Not sorted


def test_bicubic_range_non_regular():
    """Test that we can create and query a BicubicAxis object
    without regular steps in the values."""
    xvals = BicubicAxis([0.1, 0.2, 0.4, 0.5, 0.6, 0.65, 0.7])
    assert xvals.min_val == 0.1
    assert xvals.max_val == 0.7
    assert xvals.num_vals == 7
    assert xvals.step_size is None

    # Test that we can run out_of_bounds
    values = [0.05, 0.15, 0.25, 0.8, 0.5]
    expected = [True, False, False, True, False]
    is_out = xvals.out_of_bounds(values)
    assert np.array_equal(is_out, expected)

    # Test that we can run find_indices
    ix = xvals.find_indices(values)
    assert np.array_equal(ix, [0, 0, 1, 5, 3])


def test_bicubic_interpolator_simple():
    """Test that we can create and query a bicubic interpolator."""
    x_vals = np.arange(1.0, 5.0, 1.0)
    y_vals = np.arange(0.0, 3.0, 0.5)
    z_vals = np.full((len(x_vals), len(y_vals)), 1.0)
    interp_obj = BicubicInterpolator(x_vals, y_vals, z_vals)

    x_queries = np.arange(0.5, 5.5, 0.5)
    y_queries = np.arange(-0.5, 3.5, 0.5)
    results = interp_obj(x_queries, y_queries)

    # Everything within the grid should be 1.0 and everything
    # outside the grid should be 0.0.
    for i, x_q in enumerate(x_queries):
        for j, y_q in enumerate(y_queries):
            if (x_q < 1.0) or (x_q > 4.0) or (y_q < 0.0) or (y_q > 2.5):
                assert np.abs(results[i, j]) < 1e-8
            else:
                assert np.abs(results[i, j] - 1.0) < 1e-8


def test_bicubic_interpolator():
    """Test that we can create and query a bicubic interpolator."""
    x_vals = np.arange(1.0, 5.0, 1.0)
    num_x = len(x_vals)

    y_vals = np.arange(-1.0, 2.0, 0.5)
    num_y = len(y_vals)

    # Make a wavey plane of values.
    z_vals = np.empty((num_x, num_y))
    for i in range(num_x):
        for j in range(num_y):
            if j % 2 == 0:
                z_vals[i, j] = i + 0.2 * j
            else:
                z_vals[i, j] = i - 0.1 * j

    # Pick some query points.
    x_q = jnp.asarray([-0.5, 1.0, 1.5, 2.0, 2.1, 3.4, 4.0, 5.5])
    y_q = jnp.asarray([-1.5, 0.0, 0.5, 1.1, 2.0, 3.5])

    # Expected values (as computed by the sncosmo version)
    expected = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.4, -0.3, 0.54, 0.0, 0.0],
            [0.0, 0.9, 0.2, 1.04, 0.0, 0.0],
            [0.0, 1.4, 0.7, 1.54, 0.0, 0.0],
            [0.0, 1.5, 0.8, 1.64, 0.0, 0.0],
            [0.0, 2.8, 2.1, 2.94, 0.0, 0.0],
            [0.0, 3.4, 2.7, 3.54, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    # Do the JAX interpolation.
    interp_obj = BicubicInterpolator(x_vals, y_vals, z_vals)
    res_jax = interp_obj(x_q, y_q)
    assert jnp.allclose(expected, res_jax, rtol=1e-3, atol=1e-4)


def test_bicubic_interpolator2():
    """Test that we can create and query a bicubic interpolatorusing other values."""
    x_vals = [0.25 * i for i in range(100)]
    y_vals = [j / 8.0 for j in range(320)]
    z_vals = jnp.reshape(jnp.arange(len(x_vals) * len(y_vals)), (len(x_vals), len(y_vals)))

    x_q = jnp.asarray([-0.5, 0.0, 1.0, 22.2, 49.0, 50.5])
    y_q = jnp.asarray([-1.0, 5.0, 13.4, 21.5, 41.0])

    # Expected values (as computed by the sncosmo version)
    expected = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 40.0, 107.19999695, 172.0, 0.0],
            [0.0, 1320.0, 1387.19999695, 1452.0, 0.0],
            [0.0, 28456.00097656, 28523.20097351, 28588.00097656, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    # Do the JAX interpolation.
    interp_obj = BicubicInterpolator(x_vals, y_vals, z_vals)
    res_jax = interp_obj(x_q, y_q)

    assert jnp.allclose(expected, res_jax, rtol=1e-5, atol=1e-5)


def test_kernel_values():
    """Test that we can evaluate the kernel with the JAX function
    and get the same results as with the original function."""
    input = jnp.asarray([-1.5, -1.1, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    expected = jnp.asarray([-0.0625, -0.0405, 1.0, 0.5625, 0.0, -0.0625, 0.0, 0.0])
    all_results = _kernel_value(input)
    assert jnp.allclose(expected, all_results)


def test_bad_bicubic_interpolator():
    """Test that we catch failure cases for creating a bicubic interpolator."""
    good_x = [1.0, 1.1, 1.2]
    good_y = [1.0, 1.1, 1.2]
    good_z = [[1.0, 1.1, 1.2], [1.0, 1.1, 1.2], [1.0, 1.1, 1.2]]
    _ = BicubicInterpolator(good_x, good_y, good_z)

    # No too few x or y
    with pytest.raises(ValueError):
        _ = BicubicInterpolator([1.0], good_y, good_z)
    with pytest.raises(ValueError):
        _ = BicubicInterpolator(good_x, [1.0], good_z)

    # The x and y values must be in sorted order.
    z = np.full((3, 4), 1.0)
    _ = BicubicInterpolator([1.0, 2.0, 3.0], [0.0, 0.5, 1.0, 1.5], z)
    with pytest.raises(ValueError):
        _ = BicubicInterpolator([2.0, 1.0, 3.0], [0.0, 0.5, 1.0, 1.5], z)
    with pytest.raises(ValueError):
        _ = BicubicInterpolator([1.0, 2.0, 3.0], [0.0, 2.5, 1.0, 1.5], z)

    # Sizes must match z.
    with pytest.raises(ValueError):
        _ = BicubicInterpolator([1.0, 2.0, 3.0, 4.0], [0.0, 2.5, 1.0, 1.5], z)
    with pytest.raises(ValueError):
        _ = BicubicInterpolator([1.0, 2.0, 3.0], [0.0, 2.5, 1.0], z)


def test_load_bicubic_interp_from_file(test_data_dir):
    """Test loading a bicubic interpolator object from a file."""
    filename = os.path.join(test_data_dir, "truncated-salt2-h17/salt2_template_0.dat")
    test_obj = BicubicInterpolator.from_grid_file(filename)

    assert len(test_obj.x_vals) == 26
    assert len(test_obj.y_vals) == 401
    assert test_obj.z_vals.shape == (26, 401)
