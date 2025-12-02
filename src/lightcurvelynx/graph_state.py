"""A collection of sampled parameters from a statistic distribution.

Model parameters are random variables that are sampled together in a joint distribution
using a graph of dependencies. For example, the functions::

    f(a, b) = x
    g(c) = y
    h(x, y) = z

indicate that x depends on a and b, y depends on c, and z depends on x and y (and thus on a, b, and c
as well). These would form a graph that looks like::

    a - \\
         x -- \\
    b - /      \\
                z
    c -- y --- /

Within LightCurveLynx, variables are grouped into logical sets called nodes. The combination of node name
and variable name are used to indicate specific values, allowing us to use the same variable names
in multiple nodes. For example a node to generate samples from a Gaussian distribution may have internal
parameters called mean and scale that might take on different values depending on what the node is generating.
We could have one mean for an object's brightness and another for its positional relative to the center
of a host galaxy.
"""

import copy

import numpy as np
from astropy.io import ascii
from astropy.table import Table


class GraphState:
    """A class to hold the state(s) of the each variable for one or more samples of the random
    variables in the graph. Each entry is indexed by a combination of node's (unique) name and
    variable's name.  This allows nodes to have parameters with the same name, such as ra and dec.

    Attributes
    ----------
    states : dict
        A dictionary of dictionaries mapping the node's string and variable name to
        either a value or array of values for that parameters.
    num_samples : int
        A count of the number of samples stored in the GraphState.  If num_samples > 1, then
        all parameters are stored as arrays of length num_samples.  If num_samples == 1, then
        all parameters are stored as scalars.
    num_parameters : int
        The total number of parameters stored in a single sample within GraphState.
    fixed_vars : dict
        A dictionary mapping the node name to a set of the variable names that
        are fixed (not changed by resampling) in this GraphState instance.
    sample_offset : int
        An optional offset to add to the graph state for any stateful nodes.
        Default: 0
    sample_idx : int or None
        An optional index of the current sample within a larger set of samples. Only
        used when we have extracted a single sample from a multi-sample GraphState.
        Default: None
    """

    def __init__(self, num_samples=1, *, sample_offset=0):
        if num_samples < 1:
            raise ValueError(
                f"Invalid number of samples for GraphState ({num_samples}). Must be a positive integer."
            )
        self.num_samples = num_samples
        self.num_parameters = 0
        self.states = {}
        self.fixed_vars = {}
        self.sample_offset = sample_offset
        self.sample_idx = None

    def __len__(self):
        return self.num_parameters

    def __next__(self):
        return next(self._iterate())

    def __iter__(self):
        return self._iterate()

    def _iterate(self):
        """Returns a single sliced state, which is a GraphState object
        with num_samples==1 and all scalar values."""
        for idx in range(self.num_samples):
            yield self.extract_single_sample(idx)

    def __contains__(self, key):
        """Check if the GraphState contains an entry.

        The key can be::

        1) the name of a node (in which case we return True if the node exists),
        2) the full name of a parameter (in which case we return True if the
           combination of node and parameter exists), or
        3) the name of a parameter in a GraphState with a single node (in which case we return True
           if the parameter exists in that node).

        Parameters
        ----------
        key : str
            The name of the entry to check.
        """
        if key in self.states:
            # Check if this is a node name.
            return True
        elif "." in key:
            # Check if this is a full name in node.param format.
            tokens = key.split(".")
            if len(tokens) != 2:
                raise KeyError(f"Invalid GraphState key: {key}")
            return tokens[0] in self.states and tokens[1] in self.states[tokens[0]]
        elif len(self.states) == 1:
            # Special case when we have only a single node stored in the graph state.
            node_state = list(self.states.values())[0]
            if key in node_state:
                return True
        else:
            return False

    def __str__(self):
        str_lines = []
        for node_name, node_vars in self.states.items():
            str_lines.append(f"{node_name}:")
            for var_name, value in node_vars.items():
                str_lines.append(f"    {var_name}: {value}")
        return "\n".join(str_lines)

    def __eq__(self, other):
        if self.num_samples != other.num_samples:
            return False
        if len(self.states) != len(other.states):
            return False

        for node_name, node_params in self.states.items():
            # Check that this node exists in both GraphStates and has the same number
            # of parameters.
            if node_name not in other.states:
                return False
            other_params = other.states[node_name]
            if len(node_params) != len(other_params):
                return False

            # Check that the values of each parameter matches.
            for var_name, var_value in node_params.items():
                if var_name not in other_params:
                    return False
                if not np.allclose(var_value, other_params[var_name]):
                    return False

        # Finally check that the 'fixed' dictionary is the same.
        return self.fixed_vars == other.fixed_vars

    def __getitem__(self, key):
        """Access an entry in the GraphState.

        The key can be::

        1) the name of a node (in which case we return that node's dictionary of
           parameter_name -> value),
        2) the full name of a parameter (in which case we return the values), or
        3) the name of a parameter in a GraphState with a single node (in which case we
           return that parameter's values).

        Parameters
        ----------
        key : str
            The name of the entry to access.
        """
        if key in self.states:
            return self.states[key]
        elif "." in key:
            tokens = key.split(".")
            if len(tokens) != 2:
                raise KeyError(f"Invalid GraphState key: {key}")
            return self.states[tokens[0]][tokens[1]]
        elif len(self.states) == 1:
            # Special case when we have only a single node stored
            # in the graph state.
            node_state = list(self.states.values())[0]
            if key in node_state:
                return node_state[key]
        else:
            raise KeyError(f"Unknown GraphState key: {key}")

    def copy(self):
        """Create a deep copy of the GraphState.

        Returns
        -------
        GraphState
            The copied GraphState.
        """
        new_state = GraphState(num_samples=self.num_samples)
        new_state.num_parameters = self.num_parameters
        new_state.sample_offset = self.sample_offset
        new_state.sample_idx = self.sample_idx
        for node_name, node_vars in self.states.items():
            new_state.states[node_name] = {}
            for var_name, var_value in node_vars.items():
                if self.num_samples == 1:
                    new_state.states[node_name][var_name] = var_value
                else:
                    new_state.states[node_name][var_name] = var_value.copy()

        # Copy over the sets of fixed variables. This is a single set of strings
        # per node, so we just need to copy the sets.
        for node_name, var_set in self.fixed_vars.items():
            new_state.fixed_vars[node_name] = copy.deepcopy(var_set)
        return new_state

    @staticmethod
    def extended_param_name(node_name, param_name):
        """A helper function to create the full parameter name.

        Parameters
        ----------
        node_name : str
            The name of the node.
        param_name : str
            The name of the parameter.

        Returns
        -------
        extended : str
            A name of the form {node_name}.{param_name}
        """
        return f"{node_name}.{param_name}"

    @classmethod
    def from_table(cls, input_table):
        """Create the GraphState from an AstroPy Table with columns for each parameter
        and column names of the form '{node_name}.{param_name}'.

        Parameters
        ----------
        input_table : astropy.table.Table
            The input table.
        """
        num_samples = len(input_table)
        result = GraphState(num_samples=num_samples)
        for col in input_table.colnames:
            components = col.split(".")
            if len(components) != 2:
                raise ValueError(
                    f"Invalid name for entry '{col}'. Entries should be of the form 'node_name.param_name'."
                )

            # If we only have a single value then store that value instead of the np array.
            if num_samples == 1:
                result.set(components[0], components[1], input_table[col].data[0])
            else:
                result.set(components[0], components[1], input_table[col].data)
        return result

    @classmethod
    def from_file(cls, filename):
        """Create the GraphState from a saved file.

        Parameters
        ----------
        filename : str or Path
            The name of the file.
        """
        data_table = ascii.read(filename, format="ecsv")
        return GraphState.from_table(data_table)

    @classmethod
    def from_dict(cls, data, num_samples=1):
        """Create a GraphState from either a flattened dictionary, where the keys of the
        dictionary are {node_name}.{param_name}, or a nested dictionary, where
        data[node_name][param_name] = value.

        Parameters
        ----------
        data : dict
            The dictionary mapping the parameter identifier (node name and parameter name)
            to their values.
        num_samples : int
            The number of samples.
            Default:  1

        Returns
        -------
        GraphState
            The corresponding graph state.
        """
        state = GraphState(num_samples=num_samples)
        for id1, val1 in data.items():
            if "." in id1:
                # Handle the flattened array by splitting the key.
                node_name, param_name = id1.split(".")
                state.set(node_name, param_name, val1, force_copy=True, fixed=False)
            elif isinstance(val1, dict):
                # Handle the nested array by iterating over the second dictionary's entries.
                for param_name, values in val1.items():
                    state.set(id1, param_name, values, force_copy=True, fixed=False)
            else:
                raise ValueError("Input dictionary must either be flattened or nested.")
        return state

    @classmethod
    def from_list(cls, data):
        """Concatenate a list of GraphStates or single state dictionaries into a single GraphState.
        All the entries in the data must have the same set of parameters (keys).

        Parameters
        ----------
        data : list of GraphState or dict
            A list of the individual GraphState information to combine.

        Returns
        -------
        GraphState
            The corresponding graph state.
        """
        if len(data) == 0:
            raise ValueError("Cannot concatenate an empty list.")

        # Convert everything into GraphStates (if they are not already) and extract
        # the basic information.
        all_param_full_names = None
        graph_states = []
        total_samples = 0
        for current in data:
            if isinstance(current, dict):
                current = GraphState.from_dict(current)
            elif not isinstance(current, GraphState):
                raise TypeError(f"Concatenate takes either GraphState or dict. Got {type(current)}")

            # Check that this is either the first GraphState we have seen or has the same parameters
            # as the earlier GraphStates we have seen.
            current_full_names = set(current.get_all_params_names())
            if all_param_full_names is None:
                all_param_full_names = current_full_names
            elif all_param_full_names != current_full_names:
                raise ValueError(
                    f"The sets of parameters do not match. Expected {all_param_full_names}."
                    f"Received {current_full_names}."
                )

            total_samples += current.num_samples
            graph_states.append(current)

        # Allocate space for the concatenated states and fill that result.
        result = GraphState(num_samples=total_samples)
        for full_name in all_param_full_names:
            node_name, param_name = full_name.split(".")

            # Create a numpy array that is the concatenation of all the values
            # from each of the GraphStates.
            values_list = []
            for current in graph_states:
                values_list.append(np.atleast_1d(current[full_name]))
            values = np.concatenate(values_list)
            result.set(node_name, param_name, values, force_copy=False, fixed=False)
        return result

    def get_all_params_names(self):
        """Get the full name of all the parameters.

        Returns
        -------
        names : list
            A list of all the parameter names.
        """
        names = []
        for node_name, params in self.states.items():
            for param_name in params:
                names.append(self.extended_param_name(node_name, param_name))
        return names

    def get_node_state(self, node_name, sample_num=0):
        """Get a dictionary of all parameters local to the given node
        for a single sample state.

        Parameters
        ----------
        node_name : str
            The parent node whose variables to extract.
        sample_num : int
            The number of sample to extract.

        Returns
        -------
        values : dict
            A dictionary mapping the parameter name to its value.
        """
        if node_name not in self.states:
            raise KeyError(f"Node name '{node_name}' not found in GraphState.")
        if sample_num < 0 or sample_num >= self.num_samples:
            raise ValueError(f"Invalid index {sample_num} in GraphState with {self.num_samples} entries.")
        if self.num_samples == 1:
            values = self.states[node_name]
        else:
            values = {}
            for var_name, val in self.states[node_name].items():
                values[var_name] = val[sample_num]
        return values

    def set(self, node_name, var_name, value, force_copy=False, fixed=False):
        """Set a (new) parameter's value(s) in the GraphState from a given constant value
        or an array of length num_samples (to set all the values at once).

        Parameters
        ----------
        node_name : str
            The parent node holding this variable.
        var_name : str
            The parameter's name.
        value : any
            The new value of the parameter.
        force_copy : bool
            Make a copy of data in an array. If set to False, this will link
            to the array, saving memory and computation time.
            Default: False
        fixed : bool
            Treat this parameter as fixed and do not change it during subsequent calls to set.
            Default: False
        """
        # Check that the names do not use the separator value.
        if "." in node_name or "." in var_name:
            raise ValueError("GraphState names (node or variable) cannot contain the character '.'")

        # Update the meta data.
        if node_name not in self.states:
            self.states[node_name] = {}
            self.fixed_vars[node_name] = set()
        if var_name not in self.states[node_name]:
            self.num_parameters += 1

        # Check if this parameter is fixed. If so, skip the set.
        if var_name in self.fixed_vars[node_name]:
            return

        # Set the actual values.
        if self.num_samples == 1:
            # If this GraphState holds only a single sample, set it from the given value.
            self.states[node_name][var_name] = value
        elif np.isscalar(value):
            # If the value is a scalar, expand it to the correct number of samples.
            self.states[node_name][var_name] = np.full(self.num_samples, value)
        elif len(value) != self.num_samples:
            raise ValueError(
                f"Incompatible number of samples when setting GraphState for node={node_name}, "
                f"variable={var_name}: {self.num_samples} vs {len(value)}."
            )
        elif force_copy:
            self.states[node_name][var_name] = np.array(value.copy())
        else:
            self.states[node_name][var_name] = np.asarray(value)

        # Mark the variable as fixed if needed.
        if fixed:
            self.fixed_vars[node_name].add(var_name)

    def update(self, inputs, force_copy=False, all_fixed=False):
        """Set multiple parameters' value in the GraphState from a GraphState or a
        dictionary of the same form.

        Note
        ----
        The number of samples in input must either match the number of samples in the
        current object or be 1.

        Parameters
        ----------
        inputs : GraphState or dict
            Values to copy.
        force_copy : bool
            Make a copy of data in an array. If set to False, this will link
            to the array, saving memory and computation time.
            Default: False
        all_fixed : bool
            Treat all the parameters in inputs as fixed.
            Default: False

        Raises
        ------
        ValueError
            If the input an invalid number of samples.
        """
        if isinstance(inputs, GraphState):
            if self.num_samples != inputs.num_samples and inputs.num_samples != 1:
                raise ValueError(
                    f"GraphStates must have the same number of samples. "
                    f"Received {self.num_samples} and {inputs.num_samples}."
                )
            new_states = inputs.states
        else:
            new_states = inputs

        # Set the values one by one. The set function takes care of expanding
        # any values that are constants (e.g. float or int) to match the correct
        # number of samples.
        for node_name, node_vars in new_states.items():
            for var_name, value in node_vars.items():
                self.set(node_name, var_name, value, force_copy=force_copy, fixed=all_fixed)

    def extract_single_sample(self, sample_num):
        """Create a new GraphState with a single sample state and all scalar values.

        Parameters
        ----------
        sample_num : int
            The number of sample to extract.
        """
        if self.num_samples <= 0:
            raise ValueError("Cannot sample an empty GraphState")
        if sample_num < 0 or sample_num >= self.num_samples:
            raise ValueError(f"Invalid index {sample_num} in GraphState with {self.num_samples} entries.")

        # Make a copy of the GraphState with exactly one sample.
        new_state = GraphState(1)
        new_state.num_parameters = self.num_parameters
        new_state.sample_offset = self.sample_offset
        new_state.sample_idx = sample_num
        for node_name in self.states:
            new_state.states[node_name] = {}
            for var_name, value in self.states[node_name].items():
                if self.num_samples == 1:
                    new_state.states[node_name][var_name] = value
                else:
                    new_state.states[node_name][var_name] = value[sample_num]

        # Copy over the sets of fixed variables. This is a single set of strings
        # per node, so we just need to copy the sets.
        for node_name, var_set in self.fixed_vars.items():
            new_state.fixed_vars[node_name] = copy.deepcopy(var_set)

        return new_state

    def extract_parameters(self, params):
        """Extract the parameter value(s) by a given name. This is often used for
        recording the important parameters from an entire model (set of nodes).

        Parameters
        ----------
        params : str or list-like, optional
            The parameter names to extract. These can be full names ("node.param") or
            use the parameter names.

        Returns
        -------
        values : dict
            The resulting dictionary.
        """
        # If we are looking up a single parameter, but it into a list.
        if isinstance(params, str):
            params = [params]

        # Go through all the parameters. If a parameters full name is provided,
        # look it up now and save the result. Otherwise put it into a list to check
        # for in each node.
        single_params = set()
        results = {}
        for current in params:
            if "." in current:
                node_name, param_name = current.split(".")
                if node_name in self.states and param_name in self.states[node_name]:
                    results[current] = self.states[node_name][param_name]
                else:
                    raise KeyError(f"Parameter '{current}' not found in GraphState.")
            else:
                single_params.add(current)

        if len(single_params) == 0:
            # Nothing else to do.
            return results

        # Traverse the nested dictionaries looking for cases where the parameter names match.
        first_seen_node = {}
        for node_name, node_params in self.states.items():
            for param_name, param_value in node_params.items():
                if param_name in single_params:
                    if param_name in first_seen_node:
                        # We've already seen this parameter in another node. Time to use the
                        # expanded names.

                        # Start by expanding the result we have already seen if needed.
                        if param_name in results:
                            full_name_existing = GraphState.extended_param_name(
                                first_seen_node[param_name],
                                param_name,
                            )
                            results[full_name_existing] = results[param_name]
                            del results[param_name]

                        # Add the result from the current node.
                        full_name_current = GraphState.extended_param_name(node_name, param_name)
                        results[full_name_current] = param_value
                    else:
                        # This is the first time we have seen the node. Save it with
                        # just the parameter name. Also save the node where we saw it.
                        results[param_name] = param_value
                        first_seen_node[param_name] = node_name

        # Check that we found a match for all the short parameter names.
        for param_name in single_params:
            if param_name not in first_seen_node:
                raise KeyError(f"Parameter '{param_name}' not found in GraphState.")

        return results

    def to_table(self):
        """Flatten the graph state to an AstroPy Table with columns for each parameter.

        The column names are: {node_name}.{param_name}

        Returns
        -------
        values : astropy.table.Table
            The resulting Table.
        """
        values = Table()
        for node_name, node_params in self.states.items():
            for param_name, param_value in node_params.items():
                values[self.extended_param_name(node_name, param_name)] = np.array(param_value)
        return values

    def to_dict(self):
        """Flatten the graph state to a dictionary with columns for each parameter.

        The column names are: {node_name}.{param_name}

        Returns
        -------
        values : dict
            The resulting dictionary.
        """
        values = {}
        for node_name, node_params in self.states.items():
            for param_name, param_value in node_params.items():
                if self.num_samples == 1:
                    values[self.extended_param_name(node_name, param_name)] = param_value
                else:
                    values[self.extended_param_name(node_name, param_name)] = list(param_value)
        return values

    def save_to_file(self, filename, overwrite=False):
        """Save the GraphState to a file.

        Parameters
        ----------
        filename : str
            The name of the file to save.
        overwrite : bool
            Whether to overwrite an existing file.
            Default: False
        """
        data_table = self.to_table()
        ascii.write(data_table, filename, format="ecsv", overwrite=overwrite)


class DependencyGraph:
    """A class to hold the dependencies between parameters in a model. Used for
    analysis, documentation, testing, and visualization of the model structure.
    The full parameter names are in the same form used by GraphState.

    Attributes
    ----------
    all_params : set
        A set of all (full) parameter names in the graph.
    all_nodes : set
        A set of all node names in the graph.
    incoming : dict
        A dictionary mapping each parameter to the set of parameters that it depends on
        (the incoming edges).
    outgoing : dict
        A dictionary mapping each parameter to the set of parameters that depend on it
        (the outgoing edges).
    num_constants : int
        The number of constant parameters in the graph.
    """

    def __init__(self):
        self.all_params = set()
        self.all_nodes = set()
        self.incoming = {}
        self.outgoing = {}
        self.num_constants = 0

    def __len__(self):
        return len(self.all_params)

    def __contains__(self, full_param_name):
        return full_param_name in self.all_params

    def add_parameter(self, param_name, node_name=None):
        """Add a parameter to the dependency graph if it is not already present.

        Parameters
        ----------
        param_name : str
            The name of the parameter to add.
        node_name : str, optional
            The name of the node holding this parameter. If provided, the full parameter
            name will be in the same form used by GraphState for storage.
            Default: None
        """
        # If a node name is provided, create the expanded parameter name.
        # Also add the node name to the set of all nodes.
        if node_name is not None:
            if node_name not in self.all_nodes:
                self.all_nodes.add(node_name)
            param_name = GraphState.extended_param_name(node_name, param_name)

        # If we haven't seen the parameter before, add it to the graph.
        if param_name not in self.all_params:
            self.all_params.add(param_name)
            self.incoming[param_name] = set()
            self.outgoing[param_name] = set()

    def add_constant(self, value):
        """Add a constant parameter to the dependency graph.

        Parameters
        ----------
        value : any
            The value of the constant.

        Returns
        -------
        const_name : str
            The name of the constant parameter added to the graph.
        """
        const_name = f"const_{self.num_constants}={value}"
        self.num_constants += 1
        self.add_parameter(const_name)
        return const_name

    def add_edge(self, from_param, to_param):
        """Add a directed edge to the dependency graph.

        Parameters
        ----------
        from_param : str
            The name of the parameter that the edge is coming from (the dependency).
        to_param : str
            The name of the parameter that the edge is going to (the dependent).
        """
        if from_param not in self.all_params or to_param not in self.all_params:
            raise KeyError("Both parameters must be added to the graph before adding an edge.")
        self.incoming[to_param].add(from_param)
        self.outgoing[from_param].add(to_param)

    def build_subgraph(self, param_name, incoming=True, outgoing=True):
        """Get the DAG subgraph that contains this parameter. This can be:
        1) the parameters on which this parameter depends (incoming=True, outgoing=False),
        2) the parameters that depend on this parameter (incoming=False, outgoing=True), or
        3) all parameters in the same connected component as this parameter
        (incoming=True, outgoing=True).

        Parameters
        ----------
        param_name : str
            The name of the parameter to get the subgraph for.
        incoming : bool
            If True, include the parameters that have incoming edges to nodes in the subgraph.
            Default: True
        outgoing : bool
            If True, include the parameters that have outgoing edges from nodes in the subgraph.
            Default: True

        Returns
        -------
        subgraph : DependencyGraph
            The resulting subgraph.
        """
        if param_name not in self.all_params:
            raise KeyError(f"Parameter '{param_name}' not found in the graph.")

        # Use breadth-first search to find all the parameters that should be included in
        # the subgraph. This includes all the parameters that this parameter depends on and,
        # if deps_only is False, all the parameters that depend on this parameter.
        subgraph = DependencyGraph()
        to_visit = [param_name]
        while to_visit:
            current = to_visit.pop()
            if current not in subgraph.all_params:
                subgraph.add_parameter(current)

                if incoming:
                    for dep in self.incoming[current]:
                        to_visit.append(dep)

                if outgoing:
                    for dependent in self.outgoing[current]:
                        to_visit.append(dependent)

        # Add all edges where both nodes are in the subgraph.
        for param in subgraph.all_params:
            for dep in self.incoming[param]:
                if dep in subgraph.all_params:
                    subgraph.add_edge(dep, param)

        return subgraph

    def build_connected_components(self):
        """Get the DAG subgraphs that are the connected components of the graph.

        Returns
        -------
        components : list of DependencyGraph
            The resulting subgraphs.
        """
        components = []
        visited = set()
        for param in self.all_params:
            if param not in visited:
                component = self.build_subgraph(param, incoming=True, outgoing=True)
                components.append(component)
                visited.update(component.all_params)
        return components

    def _compute_depths_helper(self, param, depths):
        """Helper function to compute the depth of a parameter in the graph.

        Parameters
        ----------
        param : str
            The name of the parameter to compute the depth for.
        depths : dict
            A dictionary mapping parameter names to their depth.

        Returns
        -------
        depth : int
            The depth of the parameter.
        """
        if param in depths:
            return depths[param]
        elif len(self.incoming[param]) == 0:
            # This is a root node.
            depths[param] = 0
            return 0
        else:
            # Compute the depth as one more than the maximum depth of the dependencies.
            max_depth = max(self._compute_depths_helper(dep, depths) for dep in self.incoming[param])
            depths[param] = max_depth + 1
            return max_depth + 1

    def compute_depths(self):
        """Compute the depth of each parameter in the graph.

        Note
        ----
        This function is primarily used for visualization purposes.

        Returns
        -------
        depths : dict
            A dictionary mapping parameter names to their depth.
        """
        depths = {}
        for param in self.all_params:
            if param not in depths:
                depths[param] = self._compute_depths_helper(param, depths)
        return depths

    def to_networkx(self):
        """Create a NetworkX graph from the dependency graph.

        Returns
        -------
        graph : networkx.DiGraph
            The resulting directed graph.
        """
        try:
            import networkx as nx
        except ImportError as err:
            raise ImportError(
                "NetworkX is required to convert the dependency graph to a NetworkX graph. "
                "Please install it with 'pip install networkx'."
            ) from err

        # Compute a mapping from parameter names to their depths.
        depths = self.compute_depths()

        # Create the graph and add each node with its depth as an attribute.
        graph = nx.DiGraph()
        for param in depths:
            graph.add_node(param, layer=depths[param])

        # Add the edges. We do this after adding the nodes to ensure that all nodes are present.
        for param in self.all_params:
            for dep in self.incoming[param]:
                graph.add_edge(dep, param)
        return graph

    def _make_readable_graph_labels(self):
        """Create a mapping from parameter names to their labels for visualization.

        Returns
        -------
        labels : dict
            A dictionary mapping parameter names to their labels.
        """
        labels = {}
        for param in self.all_params:
            curr_label = param

            # Remove the "const_NN=" prefix from constant parameters.
            if "=" in param and param.startswith("const_"):
                curr_label = curr_label.split("=")[1]

            # Remove function node names and "function_node_result" from labels.
            if ":" in param:
                curr_label = "FN:" + curr_label.split(":")[-1]
            if curr_label.endswith(".function_node_result"):
                curr_label = curr_label[: -len(".function_node_result")]

            labels[param] = curr_label
        return labels

    def draw(self, param_name=None):
        """Draw the connected components of the graph using NetworkX and Matplotlib."""
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError as err:
            raise ImportError(
                "NetworkX and Matplotlib are required to draw the graph. "
                "Please install them with 'pip install networkx matplotlib'."
            ) from err

        # If a parameter name is provided, check that it exists in the graph.
        if param_name is not None and param_name not in self.all_params:
            raise KeyError(f"Parameter '{param_name}' not found in the graph.")

        components = self.build_connected_components()
        if param_name is not None:
            # If a parameter name is provided, only draw the component that contains that parameter.
            components = [comp for comp in components if param_name in comp.all_params]

        num_components = len(components)
        _, ax = plt.subplots(num_components, 1, figsize=(12, 6 * num_components))
        if num_components == 1:
            ax = [ax]

        for idx, component in enumerate(components):
            labels = component._make_readable_graph_labels()
            graph = component.to_networkx()
            layout = nx.multipartite_layout(graph, subset_key="layer")
            layout = nx.spring_layout(graph, pos=layout)
            nx.draw(
                graph,
                layout,
                arrows=True,
                labels=labels,
                edge_color="gray",
                with_labels=True,
                node_color="lightgray",
                node_size=2000,
                font_size=10,
                font_color="black",
                font_weight="bold",
                ax=ax[idx],
            )
        plt.show()


def transpose_dict_of_list(input_dict, num_elem):
    """Transpose a dictionary of iterables to a list of dictionaries.

    Parameters
    ----------
    input_dict : dict
        A dictionary of iterables, each of which is length num_elem.
    num_elem : int
        The length of the iterables.

    Returns
    -------
    output_list : list
        A length num_elem list of dictionaries, each with the same keys mapping
        to a single value.

    Raises
    ------
    ValueError
        If any of the iterables have different lengths.
    """
    if num_elem < 1:
        raise ValueError(f"Trying to transpose a dictionary with {num_elem} elements")

    output_list = [{} for _ in range(num_elem)]
    for key, values in input_dict.items():
        if len(values) != num_elem:
            raise ValueError(f"Entry '{key}' has length {len(values)}. Expected {num_elem}.")
        for i in range(num_elem):
            output_list[i][key] = values[i]
    return output_list
