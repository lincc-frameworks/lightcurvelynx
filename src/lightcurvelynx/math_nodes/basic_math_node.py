"""Nodes that perform basic math operations that can be specified as strings.

The goal of this library is to save users from needing to create a bunch of
small FunctionNodes to perform basic math.
"""

import ast

from lightcurvelynx.base_models import FunctionNode


class BasicMathNode(FunctionNode):
    """A node that evaluates basic mathematical functions.

    The BasicMathNode wraps Python's eval() function to sanitize the input string
    and thus prevent the execution of arbitrary code. It also allows the user to write
    the expression once and execute using math, numpy, or JAX. The names of the
    variables in the expression must match the input variables provided by kwargs.

    Example:
        my_node = BasicMathNode(
            "redshift + 10.0 * sin(phase)",
            redshift=host.redshift,
            phase=source.phase,
        )

    Attributes
    ----------
    expression : str
        The expression to evaluate.
    backend : str
        The math libary to use. This is auto-converted to one of (math, np, or jnp)
        depending on the input parameter.

    Parameters
    ----------
    expression : str
        The expression to evaluate.
    backend : str
        The math libary to use. Must be one of: math, numpy, np, jax, or jnp.
    node_label : str, optional
        An identifier (or name) for the current node.
    **kwargs : dict, optional
        Any additional keyword arguments. Every variable in the expression
        must be included as a kwarg.
    """

    # A list of supported Python operations. Used to prevent eval from
    # running arbitrary python expressions. The Call and Name types are special
    # cased so we can do checks and translations.
    _supported_ast_nodes = (
        ast.Module,  # Top level object when parsed as exec.
        ast.Expression,  # Top level object when parsed as eval.
        ast.Expr,  # Math expressions.
        ast.Constant,  # Constant values.
        ast.Load,  # Load a variable - must come from an approved function or variable.
        ast.Store,  # Store value - must come from an approved function or variable.
        ast.BinOp,  # Binary operations
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.UnaryOp,  # Uninary operations
        ast.UAdd,
        ast.USub,
        ast.Invert,
    )

    # A map from a very limited set of supported math constant/function names to
    # the corresponding names in [math, numpy, jax]. This is needed because
    # a very few functions have different names in different libraries.
    _math_map = {
        "abs": ["abs", "np.abs", "jnp.abs"],  # Special handling for math.
        "acos": ["math.acos", "np.acos", "jnp.acos"],
        "acosh": ["math.acosh", "np.acosh", "jnp.acosh"],
        "asin": ["math.asin", "np.asin", "jnp.asin"],
        "asinh": ["math.asinh", "np.asinh", "jnp.asinh"],
        "atan": ["math.atan", "np.atan", "jnp.atan"],
        "atan2": ["math.atan2", "np.atan2", "jnp.atan2"],
        "cos": ["math.cos", "np.cos", "jnp.cos"],
        "cosh": ["math.cosh", "np.cosh", "jnp.cosh"],
        "ceil": ["math.ceil", "np.ceil", "jnp.ceil"],
        "degrees": ["math.degrees", "np.degrees", "jnp.degrees"],
        "deg2rad": ["math.radians", "np.deg2rad", "jnp.deg2rad"],  # Special handling for math
        "e": ["math.e", "np.e", "jnp.e"],
        "exp": ["math.exp", "np.exp", "jnp.exp"],
        "fabs": ["math.fabs", "np.fabs", "jnp.fabs"],
        "floor": ["math.floor", "np.floor", "jnp.floor"],
        "log": ["math.log", "np.log", "jnp.log"],
        "log10": ["math.log10", "np.log10", "jnp.log10"],
        "log2": ["math.log2", "np.log2", "jnp.log2"],
        "max": ["max", "np.max", "jnp.max"],  # Special handling for math
        "min": ["min", "np.min", "jnp.min"],  # Special handling for math
        "pi": ["math.pi", "np.pi", "jnp.pi"],
        "pow": ["math.pow", "np.power", "jnp.power"],  # Special handling for numpy
        "power": ["math.pow", "np.power", "jnp.power"],  # Special handling for math
        "radians": ["math.radians", "np.radians", "jnp.radians"],
        "rad2deg": ["math.degrees", "np.rad2deg", "jnp.rad2deg"],  # Special handling for math
        "sin": ["math.sin", "np.sin", "jnp.sin"],
        "sinh": ["math.sinh", "np.sinh", "jnp.sinh"],
        "sqrt": ["math.sqrt", "np.sqrt", "jnp.sqrt"],
        "tan": ["math.tan", "np.tan", "jnp.tan"],
        "tanh": ["math.tanh", "np.tanh", "jnp.tanh"],
        "trunc": ["math.trunc", "np.trunc", "jnp.trunc"],
    }

    def __init__(self, expression, backend="numpy", node_label=None, **kwargs):
        if backend == "jax" or backend == "jnp":
            try:
                import jax.numpy as jnp
            except ImportError as err:  # pragma: no cover
                raise ImportError(
                    "JAX is required to use the BasicMathNode with backend='jax', please "
                    "install with `pip install jax` or `conda install conda-forge::jax`"
                ) from err

            self.backend = "jnp"
            self.backend_lib = jnp
            self.to_array = jnp.asarray
        elif backend == "numpy" or backend == "np":
            import numpy as np

            self.backend = "np"
            self.backend_lib = np
            self.to_array = np.asarray
        elif backend == "math":
            import math

            self.backend = "math"
            self.backend_lib = math
            self.to_array = lambda x: x  # No conversion
        else:
            raise ValueError(
                f"Unsupported math backend '{backend}'. Must be one of: math, numpy, np, jax, or jnp."
            )

        # Check the expression is pure math and translate it into the correct backend.
        self.expression = expression
        self._prepare(**kwargs)

        # Create a function from the expression. Note the expression has
        # already been sanitized and validated via _prepare().
        def eval_func(**kwargs):
            params = self._prepare_params(**kwargs)
            params[self.backend] = self.backend_lib

            try:
                return eval(self.expression, globals(), params)
            except Exception as problem:  # pragma: no cover
                # Provide more detailed logging, including the expression and parameters
                # used, when we encounter a math error like divide by zero.
                new_message = (
                    f"Error during math operation '{self.expression}' with args={kwargs}. "
                    f"Original error: {problem}"
                )
                raise type(problem)(new_message) from problem

        super().__init__(eval_func, node_label=node_label, **kwargs)

    def eval(self, **kwargs):
        """Evaluate the expression."""
        params = self._prepare_params(**kwargs)
        params[self.backend] = self.backend_lib
        return eval(self.expression, globals(), params)

    @staticmethod
    def list_functions():
        """Return a list of the support functions.

        Returns
        -------
        list
            A list of the supported functions.
        """
        return list(BasicMathNode._math_map.keys())

    def _prepare_params(self, **kwargs):
        """Convert all of the incoming parameters into the correct type,
        such as numpy arrays.

        Parameters
        ----------
        **kwargs : dict, optional
            The keyword arguments, including every variable in the expression.

        Returns
        -------
        params : dict
            The converted list of parameters.
        """
        params = {}
        for name, value in kwargs.items():
            params[name] = self.to_array(value)
        return params

    def _prepare(self, **kwargs):
        """Rewrite a python expression that consists of only basic math to use
        the prespecified math library. Santizes the string to prevent
        arbitrary code execution.

        Parameters
        ----------
        **kwargs : dict, optional
            Any additional keyword arguments, including the variable
            assignments.
        """
        tree = ast.parse(self.expression)

        # Walk the tree and confirm that it only contains the basic math.
        for node in ast.walk(tree):
            if isinstance(node, self._supported_ast_nodes):
                # Nothing to do, this is a valid operation for the ast.
                continue
            elif isinstance(node, ast.Call):
                # Check that function calls are only using items on the allow list.
                if node.func.id not in self._math_map:
                    raise ValueError(f"Unsupported function {node.func.id}")
            elif isinstance(node, ast.Name):
                if node.id in kwargs:
                    # This is a user supplied variable.
                    continue
                elif node.id in self._math_map:
                    # This is a math function or constant. Overwrite
                    if self.backend == "math":
                        node.id = self._math_map[node.id][0]
                    elif self.backend == "numpy" or self.backend == "np":
                        node.id = self._math_map[node.id][1]
                    elif self.backend == "jax" or self.backend == "jnp":
                        node.id = self._math_map[node.id][2]
                else:
                    raise ValueError(
                        f"Unrecognized named variable or function {node.id}. "
                        "This could be because the function is not supported or "
                        "you forgot to include the variable as an argument."
                    )
            else:
                raise ValueError(f"Invalid part of expression {type(node)}")

        # Convert the expression back into a string.
        self.expression = ast.unparse(tree)
