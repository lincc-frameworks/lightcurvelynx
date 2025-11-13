"""The BicubicInterpolator is used by SALT models.

It is adapted from sncosmo's BicubicInterpolator class (but implemented in JAX):
https://github.com/sncosmo/sncosmo/blob/v2.10.1/sncosmo/salt2utils.pyx
"""

try:
    import jax.numpy as jnp
    from jax import jit, vmap
except ImportError as err:  # pragma: no cover
    raise ImportError(
        "JAX is required to use the BicubicInterpolator class, please "
        "install with `pip install jax` or `conda install conda-forge::jax`"
    ) from err

from lightcurvelynx.utils.io_utils import read_grid_data


@jit
def _kernel_value(input_vals):
    """A vectorized (JAX-ized) form of the kernval method from:
    https://github.com/sncosmo/sncosmo/blob/v2.10.1/sncosmo/salt2utils.pyx

    Uses:
        A = -0.5
        B = A + 2.0 = 1.5
        C = A + 3.0 = 2.5
    """
    x = jnp.abs(input_vals)

    # Start with the case of 1.0 <= x <= 2.0.
    result = -0.5 * (-4.0 + x * (8.0 + x * (-5.0 + x)))

    # Override cases where x < 1.0
    result = jnp.where(x < 1.0, x * x * (1.5 * x - 2.5) + 1.0, result)

    # Override cases where x > 2.0
    result = jnp.where(x > 2.0, 0.0, result)
    return result


def _compute_linear(ix, iy, ax, ay, z_vals):
    """Computes the linear interpolation (used for the edges). This operation is
    broken out into a function so it can be jit compiled and vmapped.

    Parameters
    ----------
    ix, iy : int
        The location of the current point in the array of z values.
    ax, ay : float
        The current point distance relative to the respective nodes in the z array.
    z_vals : JAX Array
        An 2d matrix of z values.

    Returns
    -------
    float
        The result of the linear interpolation at this point.
    """
    return (1.0 - ax) * ((1.0 - ay) * z_vals[ix][iy] + ay * z_vals[ix][iy + 1]) + ax * (
        (1.0 - ay) * z_vals[ix + 1][iy] + ay * z_vals[ix + 1][iy + 1]
    )


def _compute_cubic(ix, iy, dx, dy, z_vals):
    """Computes the bicubic interpolation (used for the center). This operation is
    broken out into a function so it can be jit compiled and vmapped.

    Parameters
    ----------
    ix, iy : int
        The location of the current point in the array of z values.
    dx, dy : float
        The current point distance relative to the respective nodes in the z array.
    z_vals : JAX Array
        An 2d matrix of z values.

    Returns
    -------
    float
        The result of the bicubic interpolation at this point.
    """
    wx = _kernel_value(jnp.asarray([dx - 1.0, dx, dx + 1.0, dx + 2.0]))
    wy = _kernel_value(jnp.asarray([dy - 1.0, dy, dy + 1.0, dy + 2.0]))

    return (
        wx[0]
        * (
            wy[0] * z_vals[ix - 1][iy - 1]
            + wy[1] * z_vals[ix - 1][iy]
            + wy[2] * z_vals[ix - 1][iy + 1]
            + wy[3] * z_vals[ix - 1][iy + 2]
        )
        + wx[1]
        * (
            wy[0] * z_vals[ix][iy - 1]
            + wy[1] * z_vals[ix][iy]
            + wy[2] * z_vals[ix][iy + 1]
            + wy[3] * z_vals[ix][iy + 2]
        )
        + wx[2]
        * (
            wy[0] * z_vals[ix + 1][iy - 1]
            + wy[1] * z_vals[ix + 1][iy]
            + wy[2] * z_vals[ix + 1][iy + 1]
            + wy[3] * z_vals[ix + 1][iy + 2]
        )
        + wx[3]
        * (
            wy[0] * z_vals[ix + 2][iy - 1]
            + wy[1] * z_vals[ix + 2][iy]
            + wy[2] * z_vals[ix + 2][iy + 1]
            + wy[3] * z_vals[ix + 2][iy + 2]
        )
    )


@jit
def expand_to_cross_products(x_vals, y_vals):
    """Create the unraveled arrays representing the grid of points from each access.

    Parameters
    ----------
    x_vals : array-like
        A length N array of the x values.
    y_vals : array-like
        A length M array of the y values.

    Returns
    -------
    all_x : JAX array
        A length N * M array of the x values.
    all_y : JAX array
        A length N * M array of the y values.
    """
    grids = jnp.meshgrid(x_vals, y_vals, indexing="ij")
    x_all = jnp.ravel(grids[0])
    y_all = jnp.ravel(grids[1])
    return x_all, y_all


class BicubicInterpolator:
    """An object that performs bicubic interpolation over a 2-d grid.

    Parameters
    ----------
    x_vals : array-like
        The values along the x-axis of the grid. The values must be sorted
        and at regular step sizes.
    y_vals : array-like
        The values along the y-axis of the grid. The values must be sorted
        and at regular step sizes.
    z_vals : array-like
        The values along the z-axis of the grid.

    Attributes
    ----------
    x_vals : BicubicAxis
        The values along the x-axis of the grid stored with precomputed information.
    y_vals : BicubicAxis
        The values along the y-axis of the grid stored with precomputed information.
    z_vals : JAX array
        The values along the z-axis of the grid.
    """

    def __init__(self, x_vals, y_vals, z_vals):
        # Load and validate the x and y values.
        self.x_vals = BicubicAxis(jnp.asarray(x_vals))
        self.y_vals = BicubicAxis(jnp.asarray(y_vals))

        # Load an validate the z values.
        self.z_vals = jnp.asarray(z_vals)
        if len(self.z_vals.shape) != 2:
            raise ValueError(f"z values should be a 2-d array. Found shape={self.z_vals.shape}")
        if self.z_vals.shape[0] != len(self.x_vals) or self.z_vals.shape[1] != len(self.y_vals):
            raise ValueError(
                f"z values wrong shape. Expected shape=({len(self.x_vals)}, {len(self.y_vals)})."
                f" Found shape={self.z_vals.shape}."
            )
        self._compute_linear = vmap(jit(_compute_linear), in_axes=(0, 0, 0, 0, None))
        self._compute_cubic = vmap(jit(_compute_cubic), in_axes=(0, 0, 0, 0, None))

    @classmethod
    def from_grid_file(cls, filename, scale_factor=1.0):
        """Load the grid data from an ASCII file and create a BicubicInterpolator.

        Parameters
        ----------
        filename : str
            The name of the grid file.
        scale_factor : float
            A multiplicative scale factor for the z values.
            Default: 1.0
        """
        x_vals, y_vals, z_vals = read_grid_data(filename)
        z_vals *= scale_factor
        return BicubicInterpolator(x_vals, y_vals, z_vals)

    def __call__(self, x_q, y_q):
        """Evaluate the bicubic interpolation at a grid of points.

        Parameters
        ----------
        x_q : array-like
            The N-length array of x values.
        y_q : array-like
            The M-length array of y values.

        Returns
        -------
        results : jaxlib.xla_extension.ArrayImpl
            An N x M array of interpolated values for each (x, y) pair.
        """
        x_q = jnp.asarray(x_q)
        y_q = jnp.asarray(y_q)
        n_xq = len(x_q)
        n_yq = len(y_q)

        # Find the first index *before* each of the query values with
        # the n-1 index in each dimension mapped to n-2.
        ix = self.x_vals.find_indices(x_q)
        iy = self.y_vals.find_indices(y_q)
        ix_all, iy_all = expand_to_cross_products(ix, iy)

        # Compute the weights to use for linear interpolation and perform the linear interpolation.
        wx, wy = expand_to_cross_products(
            (x_q - self.x_vals.values[ix]) / (self.x_vals.values[ix + 1] - self.x_vals.values[ix]),
            (y_q - self.y_vals.values[iy]) / (self.y_vals.values[iy + 1] - self.y_vals.values[iy]),
        )
        lin_res = self._compute_linear(ix_all, iy_all, wx, wy, self.z_vals)

        # Compute the cubic kernel weights for each point and the resulting
        # cubic interpolated values.
        dx, dy = expand_to_cross_products(
            (self.x_vals.values[ix] - x_q) / (self.x_vals.values[ix + 1] - self.x_vals.values[ix]),
            (self.y_vals.values[iy] - y_q) / (self.y_vals.values[iy + 1] - self.y_vals.values[iy]),
        )
        quad_res = self._compute_cubic(ix_all, iy_all, dx, dy, self.z_vals)

        # Fill in the values from the different interpolations. Use 0.0 for anything out of
        # bounds. Use linear interpolation for anything along the edges. And use cubic
        # interpolation for the points in the middle.
        x_out, y_out = expand_to_cross_products(
            self.x_vals.out_of_bounds(x_q),
            self.y_vals.out_of_bounds(y_q),
        )
        out_of_bounds = x_out | y_out

        x_edge, y_edge = expand_to_cross_products(
            (ix == 0) | (ix > self.x_vals.num_vals - 3),
            (iy == 0) | (iy > self.y_vals.num_vals - 3),
        )
        on_edge = x_edge | y_edge

        results = jnp.where(out_of_bounds, 0.0, jnp.where(on_edge, lin_res, quad_res))
        results = results.reshape(n_xq, n_yq)
        return results


class BicubicAxis:
    """A helper class that represents values for an axis of bicubic interpolation
    with restrictions on acceptable data to match the SALT2 and SALT3 models.

    Restrictions include:
    - Data must contain at least 3 values.
    - Data must be sorted.
    - Data must be spaced at regular steps.

    Attributes
    ----------
    values : JAX Array
        The values of the range.
    min_val : float
        The starting value of the range.
    max_val : float
        The maximum value of the range.
    num_vals : int
        The number of values in the range.
    regular_steps : bool
        Indicates whether the axis uses regularly sized steps.
    step_size : float
        The step size of the range.
    """

    def __init__(self, values):
        # Load and validate the y values.
        self.values = jnp.asarray(values)
        if len(self.values.shape) != 1:
            raise ValueError(
                f"The RegularRange values should be a 1-d array. Found shape={self.values.shape}."
            )
        if len(self.values) < 3:
            raise ValueError(
                f"Insufficient points for RegularRange. Required >= 3. Found {len(self.values)}."
            )
        if jnp.any(self.values[:-1] >= self.values[1:]):
            raise ValueError("The RegularRange values must be in sorted order.")
        self.num_vals = len(self.values)
        self.min_val = self.values[0]
        self.max_val = self.values[-1]

        # Check the step sizes are regular (allowing for inprecision when things are written as
        # floats). If it is regular we can use a faster algorithm to find the indices.
        # We pick and save a single core search function, so JAX does not encounter an "if" during
        # tracing (at the cost of another function call during find_indices).
        step_sizes = self.values[1:] - self.values[:-1]
        min_step = jnp.min(step_sizes)
        max_step = jnp.max(step_sizes)
        if max_step - min_step > 1e-6:
            self.regular_steps = False
            self.step_size = None
            self._initial_search = self._search_nonregular
        else:
            self.regular_steps = True
            self.step_size = 0.5 * min_step + 0.5 * max_step
            self._initial_search = self._search_regular

    def __len__(self):
        return self.num_vals

    def __str__(self):
        return f"BicubicAxis [{self.min_val},{self.max_val}]. step={self.step_size}. size={self.num_vals}"

    def _search_regular(self, query_pts):
        """Do an initial search for the query points indices given regular steps."""
        return jnp.floor((query_pts - self.min_val) / self.step_size).astype(int)

    def _search_nonregular(self, query_pts):
        """Do an initial search for the query points indices given non-regular steps."""
        return jnp.searchsorted(self.values, query_pts, side="right") - 1

    def find_indices(self, query_pts):
        """Finds the first index *before* each of the query values with
        the n - 1 index in each dimension mapped to n - 2.

        values[idx[i]] <= query_pts[i] < values[idx[i] + 1]
        for all i where idx[i] > 0 and idx[i] < n - 2.

        Parameters
        ----------
        query_pts : array-like
            The values of the query points.

        Returns
        -------
        idx : JAX Array
            A pair of arrays with the indices.
        """
        query_pts = jnp.asarray(query_pts)
        idx = self._initial_search(query_pts)
        idx = jnp.where(idx < 0, 0, idx)
        idx = jnp.where(idx >= self.num_vals - 1, self.num_vals - 2, idx)

        return idx

    def out_of_bounds(self, values):
        """Compute a Boolean array of the values that are out of bounds.

        Parameters
        ----------
        values : array-like
            The N values to test

        Returns
        -------
        results : JAX array
            A length N array indicating whether each element is out of bounds.
        """
        values = jnp.asarray(values)
        return (values < self.min_val) | (values > self.max_val)
