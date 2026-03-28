"""A simple node used for testing."""

from lightcurvelynx.base_models import ParameterizedNode


class SingleVariableNode(ParameterizedNode):
    """A ParameterizedNode holding a single pre-defined variable.

    Note
    ----
    Often used for testing, but can be used to make graph dependencies clearer.

    Parameters
    ----------
    name : str
        The parameter name.
    value : any
        The parameter value.
    **kwargs : dict, optional
        Any additional keyword arguments.
    """

    def __init__(self, name, value, **kwargs):
        super().__init__(**kwargs)
        self.add_parameter(name, value, **kwargs)
