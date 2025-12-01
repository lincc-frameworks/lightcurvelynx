"""The base models used to specify the LightCurveLynx computation graph.

The computation graph is composed of ParameterizedNodes which compute random variables
(called "model parameters" or "parameters" for short) and encode the dependencies between
them. Model parameters are different from object variables in that they are *not* stored
within the object, but rather programmatically set in an external graph_state dictionary by
sampling the graph. The user does not need to access the internals of graph_state directly,
but rather can treat it as an opaque object to pass around. The dictionary can hold either
individual values (often floats) or arrays of samples.

All model parameters (random variables in the probabilistic graph) must be added using
using the ParameterizedNode.add_parameter() function. This allows the graph to track
which parameters to update and how to set them. A parameter's values can be set from a few
sources:

1) A constant (e.g., a given standard deviation for a noise model).
2) A static function or method (which does not have variables that are resampled).
3) The result of evaluating a FunctionNode, which provides a computation using other
   parameters in the graph.
4) The parameters of another ParameterizedNode.

We say that parameter X is dependent on parameter Y if the value of Y is necessary
to compute the value of X. For example, if X is set by evaluating a FunctionNode that
uses parameter Y in the computation, X is dependent on Y. The dependencies impose an
ordering of model parameters in the graph. Y must be computed before X.

ParameterNodes provide semantic groupings of individual parameters. For example, we may
have a ParameterNode representing the information needed for a Type Ia supernova.
That node's parameters would include the variables needed to evaluate the supernova's
light curve. Each of these parameters might depend on parameters in other nodes, such
as those of the host galaxy.

The execution graph is processed by starting at the final node, examining each
model parameter for that node, and recursively proceeding 'up' the graph for any
of its parameters that has a dependency. For example, the function::

    f(a, b) = x
    g(c) = y
    h(x, y) = z

would form the graph::

    a - \\
         x -- \\
    b - /      \\
                z
    c -- y --- /

where z is the 'bottom' node. Parameters a, b, and c would be at the 'top' of the
graph because they have no dependencies.  Such parameters are set by constants or
static functions.
"""

from functools import partial

import numpy as np

from lightcurvelynx.graph_state import DependencyGraph, GraphState


class _ParameterSource:
    """_ParameterSource specifies the information about where a ParameterizedNode should
    get the value for a given parameter. These objects track internal state.
    Users should not work with these objects directly.

    Attributes
    ----------
    parameter_name : str
        The name of the parameter within the node (short name).
    node_name : str
        The name of the parent node.
    source_type : int
        The type of source as defined by the class variables.
        Default: 0
    value : any
        The information that actually sets the parameter. Either a constant
        or the attribute name of a dependency node.
    dependency : ParameterizedNode or None
        The node on which this parameter is dependent.
    description : str (optional)
        A brief description of the parameter.
    allow_gradient : bool
        Allow gradients to be computed at this variable.
        Default: False
    """

    # Class variables for the source enum.
    UNDEFINED = 0
    CONSTANT = 1
    MODEL_PARAMETER = 2
    FUNCTION_NODE = 3
    COMPUTE_OUTPUT = 4

    def __init__(self, parameter_name, source_type=0, node_name="", description=None):
        self.parameter_name = parameter_name
        self.node_name = node_name
        self.source_type = source_type
        self.allow_gradient = False
        self.value = None
        self.dependency = None
        self.description = description

    def help(self):
        """Display help information about this parameter."""
        print(f"{self.parameter_name}:")
        if self.description is not None:
            print(f"    Description: {self.description}")
        else:
            print("    Description: None")

        if self.source_type == self.UNDEFINED:
            print("    Source: UNDEFINED")
        elif self.source_type == self.CONSTANT:
            print(f"    Source: CONSTANT with value = {self.value}")
        elif self.source_type == self.MODEL_PARAMETER:
            print(f"    Source: MODEL_PARAMETER {self.value} of node {str(self.dependency)}")
        elif self.source_type == self.FUNCTION_NODE:
            print(f"    Source: Result of FUNCTION_NODE {str(self.dependency)}")
        elif self.source_type == self.COMPUTE_OUTPUT:
            print("    Source: Result of computation within this node")

    def set_as_constant(self, value, allow_gradient=True):
        """Set the parameter as a constant value.

        Parameters
        ----------
        value : any
            The constant value to use.
        allow_gradient : bool
            Allow a gradient to be computed at this variable.
            Default: True
        """
        if callable(value):
            raise ValueError(f"Using set_as_constant on callable {value}")

        self.source_type = _ParameterSource.CONSTANT
        self.allow_gradient = allow_gradient
        self.dependency = None
        self.value = value

    def set_as_parameter(self, dependency, param_name):
        """Set the parameter as a model parameter of another node.  This is
        used for chaining, such as when an object's ra depends on its host's ra.

        Parameters
        ----------
        dependency : ParameterizedNode
            The node in which to access the attribute.
        param_name : str
            The name of the parameter to access.
        """
        self.source_type = _ParameterSource.MODEL_PARAMETER
        self.allow_gradient = False
        self.dependency = dependency
        self.value = param_name

    def set_as_function(self, dependency, param_name="function_node_result"):
        """Set the parameter as the result of a FunctionNode (that is not
        the current node).

        Parameters
        ----------
        dependency : ParameterizedNode
            The node in which to access the attribute.
        param_name : str
            The name of where the result is stored in the FunctionNode.
        """
        self.source_type = _ParameterSource.FUNCTION_NODE
        self.allow_gradient = False
        self.dependency = dependency
        self.value = param_name

    def set_as_compute_output(self, param_name="function_node_result"):
        """Set the parameter the output of this current node's compute() method.

        This needs to be separate from FUNCTION_NODE type (set_as_function) because
        the sampling function needs to know to call the current node's compute()
        method after the other parameters have been sampled.

        Parameters
        ----------
        dependency : ParameterizedNode
            The node in which to access the attribute.
        param_name : str
            The name of where the result is stored in the FunctionNode.
        """
        self.source_type = _ParameterSource.COMPUTE_OUTPUT
        self.allow_gradient = False
        self.value = param_name


class _AttributeIndicatorNode:
    """A class to wrap a single attribute of an object. These objects track internal state.
    Users should not work with these objects directly.

    Attributes
    ----------
    attr_name : str
        The name of the attribute to access.
    parent : ParameterizedNode
        The parent node that owns this attribute.
    """

    def __init__(self, attr_name, parent):
        self.attr_name = attr_name
        self.parent = parent

    def __call__(self, graph_state):
        """Return the value in the graph_state of this attribute.

        Parameters
        ----------
        graph_state : GraphState
            The current graph state.

        Returns
        -------
        any
            The value of the attribute in the graph state.
        """
        return graph_state[self.parent.node_string][self.attr_name]


class ParameterizedNode:
    """Any model that uses parameters that can be set by constants,
    functions, or other parameterized nodes.

    ParameterizedNodes do not store values directly, but rather provide a recipe
    for how to generate the parameters' values.  The sampled values are read from
    and written to a GraphState object that stores all the parameter values for
    all the nodes.

    Attributes
    ----------
    node_label : str
        An optional human readable identifier (name) for the current node.
    node_string : str
        The full string used to identify a node. This is a combination of the node's position
        in the graph (if known), node_label (if provided), and class information. This is
        used to access the parameters for this node in the graph_state.
    setters : dict
        A dictionary mapping the parameters' names to information about the setters
        (_ParameterSource). The model parameters are stored in the order in which they
        need to be set.
    node_pos : int or None
        A unique ID number for each node in the graph indicating its position.
        Assigned during resampling or set_graph_positions(). This is required to resolve
        naming collisions so we do not overwrite parameters from other nodes.

    Parameters
    ----------
    node_label : str, optional
        An identifier (or name) for the current node.
    **kwargs : dict, optional
        Any additional keyword arguments.
    """

    def __init__(self, node_label=None, **kwargs):
        self.setters = {}
        self.node_label = node_label
        self.node_pos = None
        self.node_string = None

        # Give the node a temporary name.
        self._update_node_string()

    def __str__(self):
        """Return the string representation of the node."""
        return self.node_string

    def _update_node_string(self, new_str=None):
        """Update the node's string.

        Parameters
        ----------
        new_str : str, optional
            The new node string. If not provided, the node_string
            is automatically computed from the other node information.
        """
        if self.node_label is not None:
            # If a label is given, just use that. It overrides even the new_str.
            self.node_string = self.node_label
        elif new_str is not None:
            self.node_string = new_str
        else:
            # Otherwise use a combination of the node's class and position.
            pos_string = f"_{self.node_pos}" if self.node_pos is not None else ""
            self.node_string = f"{self.__class__.__name__}{pos_string}"

        # Update the node_name of all node's parameter setters.
        for _, setter_info in self.setters.items():
            setter_info.node_name = self.node_string

    def set_graph_positions(self, seen_nodes=None):
        """Force an update of the graph structure (numbering of each node).

        Parameters
        ----------
        seen_nodes : set, optional
            A set of nodes that have already been processed to prevent infinite loops.
            Caller should not set.
        """
        # Make sure that we do not process the same nodes multiple times.
        if seen_nodes is None:
            seen_nodes = set()
        if self in seen_nodes:
            return
        seen_nodes.add(self)

        # Update the node's position in the graph and its string.
        self.node_pos = len(seen_nodes) - 1
        self._update_node_string()

        # Recursively update any direct dependencies.
        for setter_info in self.setters.values():
            if setter_info.dependency is not None and setter_info.dependency is not self:
                setter_info.dependency.set_graph_positions(seen_nodes)

    def list_params(self):
        """Return a list of this node's parameterized values.

        Returns
        -------
        names : list[str]
            The name of all of the parameterized values for this node.
        """
        return list(self.setters.keys())

    def describe_params(self, names=None):
        """Print out a description of the node's parameters.

        Parameters
        ----------
        names : list[str], optional
            A list of parameter names to describe. If None, describes all parameters.
        """
        if names is None:
            names = self.list_params()

        if len(names) > 1:
            print(f"Parameters in {self.node_string}:\n")
        for name in names:
            if name in self.setters:
                self.setters[name].help()
                print()
            else:
                print(f"Parameter: {name} not found in node {self.node_string}.")

    def has_valid_param(self, name):
        """Check whether the node has a given parameterized value and that it is not
        always set to None.

        Parameters
        ----------
        name : str
            The name of the parameter.

        Returns
        -------
        contains : bool
            Whether the node contains a given parameter and it is not always None.
        """
        if name not in self.setters:
            return False

        setter = self.setters[name]
        return not (setter.source_type == _ParameterSource.CONSTANT and setter.value is None)

    def get_param(self, graph_state, name, default=None):
        """Get the value of a parameter stored in this node or a default value.

        Note
        ----
        This is an optional helper function that accesses the internals of graph_state
        using the node's information (e.g. its hash value).

        Parameters
        ----------
        graph_state : dict
            The dictionary of graph state information.
        name : str
            The parameter name to query.
        default : any
            The default value to return if the parameter is not in GraphState.

        Returns
        -------
        any
            The parameter value or the default.

        Raises
        ------
        ValueError
            If graph_state is None.
        """
        if graph_state is None:
            raise ValueError(f"Unable to look up parameter={name}. No graph_state given.")
        if self.node_string in graph_state and name in graph_state[self.node_string]:
            return graph_state[self.node_string][name]
        return default

    def get_local_params(self, graph_state):
        """Get a dictionary of all parameters local to this node.

        Note
        ----
        This is an optional helper function that accesses the internals of graph_state
        using the node's information (e.g. its hash value).

        Parameters
        ----------
        graph_state : GraphState
            An object mapping graph parameters to their values.

        Returns
        -------
        result : dict
            A dictionary mapping the parameter name to its value.

        Raises
        ------
        KeyError
            If no parameters have been set for this node.
        ValueError
            If graph_state is None.
        """
        if graph_state is None:
            raise ValueError("No graph_state given.")
        return graph_state[self.node_string]

    def set_parameter(self, name, value=None, **kwargs):
        """Set the source of a single *existing* parameter in the ParameterizedNode.

        Parameters within a node are actually a mapping of the parameter name
        to a _ParameterSource object that indicates how they are set during sampling.
        Some parameters may be set as a constant value while others may be set by
        evaluating a function that depends on other parameters.

        Notes
        -----
        * Does NOT set an initial value for the model parameter. The user must
          sample the parameters for this to be set.
        * The model parameters are stored in the order in which they are added.

        Parameters
        ----------
        name : str
            The parameter name to add.
        value : any, optional
            The information to use to set the parameter. Can be a constant,
            function, ParameterizedNode, or self.
        **kwargs : dict, optional
           All other keyword arguments, possibly including the parameter setters.

        Raises
        ------
        KeyError
            If there is a parameter collision or the parameter cannot be found.
        ValueError
            If the setter type is not supported.
        """
        # Set the node's position in the graph to None to indicate that the
        # structure might have changed. It needs to be updated with set_graph_positions().
        self.node_pos = None

        # Check for parameter has been added and if so, find the index. All parameters must
        # be added first with "add_parameter()".
        if name not in self.setters:
            raise KeyError(
                f"Tried to set parameter '{name}' that has not been added to node {self.node_string}."
            ) from None

        if value is None and name in kwargs:
            # The value wasn't set, but the name is in kwargs.
            value = kwargs[name]

        if callable(value):
            if isinstance(value, _AttributeIndicatorNode):
                # Case 1a: This is an attribute of a ParameterizedNode.
                # We set the parameter's value in this node as the extraction of the
                # parameter's value in the parent node.
                if value.attr_name in value.parent.setters:
                    self.setters[name].set_as_parameter(value.parent, value.attr_name)
                else:
                    raise ValueError(
                        f"Trying to set parameter '{name}' to the {type(value.parent)}.{value.attr_name}, "
                        f"but unable to find that parameter in class {type(value.parent)}."
                    )
            else:
                # Case 1b: This is a general function or callable method from another object.
                # We treat it as static (we don't resample the other object) and
                # wrap it in a FunctionNode.
                func_node = FunctionNode(value, **kwargs)
                self.setters[name].set_as_function(func_node)
        elif isinstance(value, FunctionNode):
            # Case 2: We are using the result of a computation of the function node.
            # If the FunctionNode has names outputs that match the variable, use that.
            output_name = name if name in value.outputs else "function_node_result"
            self.setters[name].set_as_function(value, output_name)
        elif isinstance(value, ParameterizedNode):
            # Case 3: We are trying to access a parameter of a ParameterizedNode
            # with the same name.
            if value == self:
                raise ValueError(f"Parameter '{name}' is recursively assigned to self.{name}.")
            if name not in value.setters:
                raise ValueError(f"Parameter '{name}' missing from {str(value)}.")
            self.setters[name].set_as_parameter(value, name)
        else:
            # Case 4: The value is constant (including None).
            self.setters[name].set_as_constant(value)

    def set_allow_gradient(self, name, allow_gradient):
        """Turn on or off the ability to compute a gradient for this variable.

        Parameters
        ----------
        name : str
            The parameter name to modify.
        allow_gradient : bool
            The new setting for allow_gradient.
        """
        self.setters[name].allow_gradient = allow_gradient

    def add_parameter(self, name, value=None, allow_gradient=None, description=None, **kwargs):
        """Add a single *new* parameter to the ParameterizedNode.

        Notes
        -----
        * Checks multiple sources in the following order: Manually specified value,
          an entry in kwargs, or None.
        * Does NOT set an initial value for the model parameter. The user must
          sample the parameters for this to be set.
        * The model parameters are stored in the order in which they are added.

        Parameters
        ----------
        name : str
            The parameter name to add.
        value : any, optional
            The information to use to set the parameter. Can be a constant,
            function, ParameterizedNode, or self.
        allow_gradient : bool or None
            Allow gradients to be computed for this variable. If set to None uses the default
            for the setter type (True for constant and False for everything else).
            Default: None
        description : str, optional
            A brief description of the parameter.
        **kwargs : dict, optional
           All other keyword arguments, possibly including the parameter setters.

        Raises
        ------
        KeyError
            If there is a parameter collision or the parameter cannot be found.
        """
        # Check for parameter collision and add a place holder value to the 'setters' dictionary.
        if hasattr(self, name) and name not in self.setters:
            raise KeyError(
                f"Parameter name '{name}' conflicts with a predefined model parameter "
                f"or class attribute in {self.node_string}"
            )
        if self.setters.get(name, None) is not None:
            raise KeyError(f"Duplicate parameter set: '{name}' in {self.node_string}")

        # Add an entry for the setter function and fill in the remaining information using
        # set_parameter(). We add an initial (dummy) value here to indicate that this parameter
        # exists and was added via add_parameter().
        self.setters[name] = _ParameterSource(
            parameter_name=name,
            source_type=_ParameterSource.UNDEFINED,
            node_name=str(self),
            description=description,
        )
        self.set_parameter(name, value, **kwargs)

        # Check if we should override allow_gradient.
        if allow_gradient is not None:
            self.setters[name].allow_gradient = allow_gradient

        # Create an _AttributeIndicatorNode to represent this parameter.
        # This node allows us to reference the parameter as object.parameter_name
        # for chaining without copying the value. For example, if my_node_1, is a
        # ParameterizedNode with a parameter x, we can do:
        #   my_node_2 = ParameterizedNode(y=my_node_1.x)
        # and my_node_2 will know to use the sampled values of x from my_node_1
        # (as opposed to the setter for x).
        setattr(self, name, _AttributeIndicatorNode(name, self))

    def get_parameter_indicator(self, param_name):
        """Return the _AttributeIndicatorNode for a given parameter. This replicates
        the behavior of accessing the parameter as an attribute of the object
        (e.g., obj.param_name), but in functional form.

        Parameters
        ----------
        param_name : str
            The name of the parameter indicator to get.

        Returns
        -------
        getter : _AttributeIndicatorNode
            The indicator node for the given parameter.
        """
        if param_name not in self.setters:
            raise KeyError(f"Parameter name '{param_name}' not found in node {self.node_string}.")
        return getattr(self, param_name)

    def compute(self, graph_state, rng_info=None, **kwargs):
        """Placeholder for a general compute function, which is called at the end
        of the sampling process and can produce derived parameters. This function
        is the main processing step in a FunctionNode.

        Parameters
        ----------
        graph_state : GraphState
            An object mapping graph parameters to their values. This object is modified
            in place as it is sampled.
        rng_info : numpy.random._generator.Generator, optional
            A given numpy random number generator to use for this computation. If not
            provided, the function uses the node's random number generator.
        **kwargs : dict, optional
            Additional function arguments.
        """
        return None

    def _sample_helper(self, graph_state, seen_nodes, rng_info=None):
        """Internal recursive function to sample the model's underlying parameters
        if they are provided by a function or ParameterizedNode. All sampled
        parameters for all nodes are stored in the graph_state dictionary, which is
        modified in-place.

        Parameters
        ----------
        graph_state : GraphState
            An object mapping graph parameters to their values. This object is modified
            in place as it is sampled.
        seen_nodes : dict
            A dictionary mapping nodes strings seen during this sampling run to their object.
            Used to avoid sampling nodes multiple times and to validity check the graph.
        rng_info : numpy.random._generator.Generator, optional
            A given numpy random number generator to use for this computation. If not
            provided, the function uses the node's random number generator.

        Raises
        ------
        KeyError
            If the sampling encounters an error with the order of dependencies.
        """
        node_str = str(self)
        if node_str in seen_nodes:
            if seen_nodes[node_str] != self:
                raise ValueError(
                    f"Duplicate node label '{node_str}'. Every node must have a unique label. "
                    "This most often happens when the node_label parameter is set directly."
                )
            return  # Nothing to do
        seen_nodes[node_str] = self

        # Run through each parameter and sample it based on the given recipe.
        # As of Python 3.7 dictionaries are guaranteed to preserve insertion ordering,
        # so this will iterate through model parameters in the order they were inserted.
        any_compute = False
        for name, setter in self.setters.items():
            # Check if we need to sample this parameter's dependency node.
            if setter.dependency is not None and setter.dependency != self:
                setter.dependency._sample_helper(graph_state, seen_nodes, rng_info=rng_info)

            # Set the result from the correct source.
            if setter.source_type == _ParameterSource.CONSTANT:
                if graph_state.num_samples == 1:
                    graph_state.set(self.node_string, name, setter.value)
                else:
                    repeated_value = np.array([setter.value] * graph_state.num_samples)
                    graph_state.set(self.node_string, name, repeated_value)
            elif setter.source_type == _ParameterSource.MODEL_PARAMETER:
                graph_state.set(
                    self.node_string,
                    name,
                    graph_state[setter.dependency.node_string][setter.value],
                )
            elif setter.source_type == _ParameterSource.FUNCTION_NODE:
                graph_state.set(
                    self.node_string,
                    name,
                    graph_state[setter.dependency.node_string][setter.value],
                )
            elif setter.source_type == _ParameterSource.COMPUTE_OUTPUT:
                # Computed parameters are set only after all the other (input) parameters.
                any_compute = True
            else:
                raise ValueError(f"Invalid _ParameterSource type {setter.source_type}")

        # If this is a function node and the parameters depend on the result of its own computation
        # call the compute function to fill them in.
        if any_compute:
            self.compute(graph_state, rng_info)

    def sample_parameters(self, given_args=None, num_samples=1, rng_info=None, sample_offset=0):
        """Sample the model's underlying parameters if they are provided by a function
        or ParameterizedNode.

        Parameters
        ----------
        given_args : dict, optional
            A dictionary representing the given arguments for this sample run.
            This can be used as the JAX PyTree for differentiation.
        num_samples : int
            A count of the number of samples to compute.
            Default: 1
        rng_info : numpy.random._generator.Generator, optional
            A given numpy random number generator to use for this computation. If not
            provided, the function uses the node's random number generator.
        sample_offset : int
            An optional offset to add to the graph state for any stateful nodes.
            This allows the system to better support testing and parallelized sampling.
            Default: 0 (no offset)

        Returns
        -------
        graph_state : GraphState
            A dictionary of dictionaries mapping node->hash, variable_name to either a
            value or array of values. This data structure is modified in place to represent
            the model's state(s).

        Raises
        ------
        ValueError
            If the sampling encounters a problem with the order of dependencies.
        """
        # If the graph structure has never been set, do that now.
        if self.node_pos is None:
            nodes = set()
            self.set_graph_positions(seen_nodes=nodes)

        # Create space for the results and set all the given_args as fixed parameters.
        results = GraphState(num_samples, sample_offset=sample_offset)
        if given_args is not None:
            results.update(given_args, all_fixed=True)

        # Resample the nodes. All information is stored in the returned results dictionary.
        seen_nodes = {}
        self._sample_helper(results, seen_nodes, rng_info=rng_info)
        return results

    def _dependency_graph_helper(self, dependency_graph):
        """Internal recursive function to build a directed acyclic graph (DAG) representing
        the parameters in the model and their dependencies. Each node in the graph is a parameter
        and each edge indicates nodes that parameter depends on.

        Parameters
        ----------
        dependency_graph : DependencyGraph
            An object tracking the dependencies between parameters. This object is modified
            in place to represent the current state.

        Raises
        ------
        KeyError
            If the sampling encounters an error with the order of dependencies.
        """
        # Check if we have already processed this node. The node name gets added
        # as soon as we add any of its parameters.
        node_name = str(self)
        if node_name in dependency_graph.all_nodes:
            return  # Nothing to do. We have already processed this node.

        # Add each parameter to the dependency graph.
        for param_name, setter in self.setters.items():
            full_name = GraphState.extended_param_name(node_name, param_name)
            dependency_graph.add_parameter(param_name, node_name)

            # Recursively process any dependencies first, including constants.
            dep_name = None
            if setter.dependency is not None and setter.dependency != self:
                dep_name = GraphState.extended_param_name(setter.dependency.node_string, setter.value)
                setter.dependency._dependency_graph_helper(dependency_graph)
            elif setter.source_type == _ParameterSource.CONSTANT:
                dep_name = dependency_graph.add_constant(setter.value)

            # If we have a dependency, add an edge from the dependency to this parameter.
            # dep_name will be None for parameters that are the result of internal computations
            # (i.e., _ParameterSource.COMPUTE_OUTPUT) which is handled in the subclass.
            if dep_name is not None:
                dependency_graph.add_edge(dep_name, full_name)

    def build_dependency_graph(self):
        """Build a directed acyclic graph (DAG) representing the parameters in the model
        and their dependencies.

        Returns
        -------
        dependency_graph : DependencyGraph
            An object tracking the dependencies between parameters.
        """
        # If the graph structure has never been set, do that now.
        if self.node_pos is None:
            nodes = set()
            self.set_graph_positions(seen_nodes=nodes)

        # Create space for the results and set all the given_args as fixed parameters.
        dependency_graph = DependencyGraph()

        # Recursively build the dependency graph.
        self._dependency_graph_helper(dependency_graph)
        return dependency_graph

    def build_pytree(self, graph_state, partial=None):
        """Build a JAX PyTree representation of the variables in this graph.

        Parameters
        ----------
        graph_state : dict
            A dictionary of dictionaries mapping node->hash, variable_name to value.
            This data structure is modified in place to represent the current state.
        partial : dict
            The partial results so far. This is modified in place by the function.
            A dictionary mapping node name to a dictionary mapping each variable's name
            to its value.
            Default: None

        Returns
        -------
        values : dict
            A dictionary mapping node name to a dictionary mapping each variable's name
            to its value.
        """
        # Check if the node might have incomplete information.
        if self.node_pos is None:
            raise ValueError(
                f"Node {self.node_string} is missing position. You must call "
                "set_graph_positions() before building a pytree."
            )

        # Skip nodes that we have already seen.
        if partial is None:
            partial = {}
        if self.node_string in partial:
            return partial

        # Add new values to the pytree, recursively exploring dependencies.
        partial[self.node_string] = {}
        for name, setter_info in self.setters.items():
            if setter_info.allow_gradient:
                # Anything wth allow_gradient == True goes in the PyTree.
                partial[self.node_string][name] = graph_state[self.node_string][name]
            elif setter_info.dependency is not None:
                # We only recursively check parameters above non-gradient nodes.
                partial = setter_info.dependency.build_pytree(graph_state, partial)
        return partial


class FunctionNode(ParameterizedNode):
    """A class to wrap functions and their argument settings.

    The node can compute the result using a given function (the func
    parameter) or through the compute() method. If func=None
    then the user must override compute().

    Attributes
    ----------
    func : function or method or partial
        The function to call during an evaluation. If this is None
        you must override the compute() method directly.
    args_names : list
        A list of argument names to pass to the function.
    outputs : list of str
        The output model parameters of this function.

    Parameters
    ----------
    func : function or method
        The function to call during an evaluation.
    node_label : str, optional
        An identifier (or name) for the current node.
    outputs : list of str, optional
        The output model parameters of this function. If None, uses
        a single model parameter result.
    fixed_params : dict, optional
        A dictionary mapping a parameter name in the function to its fixed value.
    **kwargs : dict, optional
        Any additional keyword arguments.

    Examples
    --------

    ::

        my_func = TDFunc(random.randint, a=1, b=10)
        value1 = my_func()      # Sample from default range
        value2 = my_func(b=20)  # Sample from extended range

    Note
    ----
    All the function's parameters that will be used need to be specified
    in either the default_args dict, object_args list, or as a kwarg in the
    constructor. Arguments cannot be first given during function call.
    For example, the following will fail (because b is not defined in the
    constructor)::

        my_func = TDFunc(random.randint, a=1)
        value1 = my_func(b=10.0)
    """

    def __init__(self, func, node_label=None, outputs=None, fixed_params=None, **kwargs):
        # We set the function before calling the parent class so we can use
        # the function's name (if needed).
        if fixed_params is not None and len(fixed_params) > 0:
            # Create a partial function with some of the parameters fixed.
            self.func = partial(func, **fixed_params)

            # We need to set the __name__ parameter because it is not preserved by partial.
            self.func.__name__ = func.__name__
        else:
            # Use the function as-is.
            self.func = func

        super().__init__(node_label=node_label, **kwargs)

        # Add all of the parameters from default_args or the kwargs.
        self.arg_names = []
        for key, value in kwargs.items():
            self.arg_names.append(key)
            self.add_parameter(key, value, description="Input argument for function.")

        # Add the output arguments.
        if not outputs:
            outputs = ["function_node_result"]
        self.outputs = outputs
        for name in outputs:
            # For output parameters we add a placeholder of None to set up the basic data, such as
            # the getter function and the entry in parameters. Then we change the
            # type to point to own result.
            self.add_parameter(name, None, description="Output result of function.")
            self.setters[name].set_as_compute_output(param_name=name)

    def _non_func(self):
        """This function does nothing. This is used for FunctionNodes where the actual computation
        happens in an overloaded compute() function."""
        pass

    def _update_node_string(self, new_str=None):
        """Update the node's string. A FunctionNode's string includes
        the function name in addition to the class name.
        """
        if new_str is None:
            pos_string = f"_{self.node_pos}" if self.node_pos is not None else ""
            fn_str = f":{self.func.__name__}" if self.func is not None else ""
            new_str = f"{self.__class__.__name__}{fn_str}{pos_string}"
        super()._update_node_string(new_str)

    def _build_inputs(self, graph_state, **kwargs):
        """Build the input arguments for the node's function.

        Parameters
        ----------
        graph_state : GraphState
            An object mapping graph parameters to their values. This object is modified
            in place as it is sampled.
        **kwargs : dict, optional
            Additional function arguments.

        Returns
        -------
        args : dict
            A dictionary mapping each input argument's name to its value.
        """
        args = {}
        for key in self.arg_names:
            if key in kwargs:
                args[key] = kwargs[key]
            else:
                args[key] = graph_state[self.node_string][key]
        return args

    def _save_results(self, results, graph_state):
        """Save the results to the graph state.

        Parameters
        ----------
        results : iterable
            The function's results.
        graph_state : GraphState
            An object mapping graph parameters to their values. This object is modified
            in place as it is sampled.
        """
        if len(self.outputs) == 1:
            graph_state.set(self.node_string, self.outputs[0], results)
        else:
            if len(results) != len(self.outputs):
                raise ValueError(
                    f"Incorrect number of results returned by {self.func.__name__}. "
                    f"Expected {len(self.outputs)}, but got {results}."
                )
            for i in range(len(self.outputs)):
                graph_state.set(self.node_string, self.outputs[i], results[i])

    def _dependency_graph_helper(self, dependency_graph):
        """Internal recursive function to build a directed acyclic graph (DAG) representing
        the parameters in the model and their dependencies.

        Parameters
        ----------
        dependency_graph : DependencyGraph
            An object tracking the dependencies between parameters. This object is modified
            in place to represent the current state.
        """
        node_name = str(self)
        if node_name in dependency_graph.all_nodes:
            return  # Nothing to do. We have already processed this node.

        # Handle the dependencies of the input features.
        super()._dependency_graph_helper(dependency_graph)

        # For each computed parameter, add the cross product of inputs to outputs.
        for param_name, setter in self.setters.items():
            if setter.source_type == _ParameterSource.COMPUTE_OUTPUT:
                out_full_name = GraphState.extended_param_name(node_name, param_name)
                for input_name in self.arg_names:
                    input_full_name = GraphState.extended_param_name(node_name, input_name)
                    dependency_graph.add_edge(input_full_name, out_full_name)

    def compute(self, graph_state, rng_info=None, **kwargs):
        """Execute the wrapped function.

        The input arguments are taken from the current graph_state and the outputs
        are written to graph_state.

        Parameters
        ----------
        graph_state : GraphState
            An object mapping graph parameters to their values. This object is modified
            in place as it is sampled.
        rng_info : numpy.random._generator.Generator, optional
            A given numpy random number generator to use for this computation. If not
            provided, the function uses the node's random number generator.
        **kwargs : dict, optional
            Additional function arguments.

        Returns
        -------
        results : any
            The result of the computation. This return value is provided so that testing
            functions can easily access the results.

        Raises
        ------
        ValueError
            If func attribute is None.
        """
        if self.func is None:
            raise ValueError(
                f"The FunctionNode {self.node_string}'s 'func' parameter is None. "
                "You need to either set func or override compute()."
            )

        # Build a dictionary of arguments for the function, call the function, and save
        # the results in the graph state.
        args = self._build_inputs(graph_state, **kwargs)
        results = self.func(**args)
        self._save_results(results, graph_state)
        return results

    def generate(self, given_args=None, num_samples=1, rng_info=None, **kwargs):
        """A helper function that regenerates the parameters for this node and the
        nodes above it, then returns the output for this individual node.

        This is used both for testing and for computing JAX gradients.

        Parameters
        ----------
        given_args : dict, optional
            A dictionary representing the given arguments for this sample run.
            This can be used as the JAX PyTree for differentiation.
        num_samples : int
            A count of the number of samples to compute.
            Default: 1
        rng_info : numpy.random._generator.Generator, optional
            A given numpy random number generator to use for this computation. If not
            provided, the function uses the node's random number generator.
        **kwargs : dict, optional
            Additional function arguments.
        """
        state = self.sample_parameters(given_args, num_samples, rng_info)

        # Get the result(s) of compute from the state object.
        if len(self.outputs) == 1:
            return self.get_param(state, self.outputs[0])

        results = []
        for output_name in self.outputs:
            results.append(self.get_param(state, output_name))
        return results
