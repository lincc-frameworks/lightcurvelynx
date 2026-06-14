"""The StateExpansionNode is a special FunctionNode that expands the graph state
to flatten out subparameters -- where a single row is transformed into multiple related
rows. This node is meant to be used in the rare case that a single step in the parameter
sampling process needs to change the number of samples (one sample branching into multiple
results).

Note
----
This class is experimental and may be removed in the future.
"""

import numpy as np

from lightcurvelynx.base_models import FunctionNode


class StateExpansionNode(FunctionNode):
    """A special FunctionNode that expands the graph state to flatten out the subparameters
    that indicate multiple different behaviors per-row. For example, strong lensing will split
    a single object into multiple light curves (with different magnifications and time delays).
    The user provides a list of subparameter dictionaries where each dictionary lists the values
    for the new columns to be added for each sample. Alternatively, the user can directly provide
    the number of repeats for each sample to produce identical copies of the rows without adding
    any new columns.

    Because of the structure of sampling, each ExpansionNode will only be triggered
    once during each sampling process.

    Note
    ----
    This class is experimental and may be removed in the future.

    Parameters
    ----------
    param_names : list of str, optional
        A list of parameters to unpack from the param_values argument.
        Default: None
    param_values : parameter
        A list of dictionaries where each dictionary contains the same keys (new column names)
        and the values for each parameter. The number of dictionaries must be the same as the
        number of samples in the graph state.
        Default: None
    repeats: int or list of int, optional
        The number of repeats for each sample. If an int is provided, it is applied to
        all samples. If a list is provided, it must be the same length as the number
        of samples in the graph state and specifies the number of repeats for each sample.
        Only used if param_values is not provided.
        Default: None
    **kwargs : dict, optional
         Additional keyword arguments.
    """

    def __init__(self, *, param_names=None, param_values=None, repeats=None, **kwargs):
        if param_values is not None:
            if repeats is not None:
                raise ValueError(
                    "You cannot provide both 'repeats' and 'param_values' to StateExpansionNode."
                )
            if param_names is None or len(param_names) == 0:
                raise ValueError(
                    "You must provide a non-empty 'param_names' list when using 'param_values' "
                    "in StateExpansionNode."
                )
            for name in param_names:
                if not isinstance(name, str):
                    raise ValueError("All entries in 'param_names' must be strings.")
            self.new_param_names = param_names
        elif param_names is not None:
            raise ValueError("You cannot provide 'param_names' without 'param_values' to StateExpansionNode.")
        elif repeats is None:
            raise ValueError("You must provide either 'repeats' or 'param_values' to StateExpansionNode.")
        else:
            # No new parameters to add, just repeat the rows.
            self.new_param_names = []

        super().__init__(
            func=self._non_func,  # We will override compute() so the function doesn't matter.
            repeats=repeats,
            param_values=param_values,
            outputs=["org_inds", "sub_inds"] + self.new_param_names,
            **kwargs,
        )

    def compute(self, graph_state, rng_info=None, **kwargs):
        """Expand the graph state by applying the repeat() method.

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
        list
            A list containing the original indices and sub-indices for each sample after expansion,
            as well as any new parameter values if provided. The order of the returned list is:
            [org_inds, sub_inds, new_param_1_values, new_param_2_values, ...] where the new parameter
            values match the order of the param_names provided during initialization.
        """
        # Get the parameters for the number of repeats and subparameters.
        args = self._build_inputs(graph_state, **kwargs)
        repeats = args.get("repeats", None)
        param_values = args.get("param_values", None)

        # If new parameters to add are not None, we compute the repeats array from the length
        # of the subparameters for each sample.
        if len(self.new_param_names) > 0:
            if len(param_values) != graph_state.num_samples:
                raise ValueError(
                    f"The number of subparameter dictionaries ({len(param_values)}) must match the "
                    f"number of samples in the graph state ({graph_state.num_samples})."
                )
            repeats = [0] * graph_state.num_samples

            # We get the first dictionary to compute the column names.
            column_names_set = set(self.new_param_names)
            concat_values = {col: [] for col in self.new_param_names}
            for idx, subparam_dict in enumerate(param_values):
                # Make sure each dictionary has the required column names (keys).
                if not column_names_set.issubset(subparam_dict.keys()):
                    missing_cols = column_names_set - set(subparam_dict.keys())
                    raise ValueError(
                        f"Each subparameter dictionary at index {idx} must contain the following keys: "
                        f"{', '.join(column_names_set)}. Missing keys: {', '.join(missing_cols)}."
                    )

                # Use the first column to compute the number of repeats for this sample.
                repeats[idx] = len(subparam_dict[self.new_param_names[0]])

                # For each entry in the dictionary, check that the length matches the number of repeats
                # and concatenate the values for this new column.
                for col in self.new_param_names:
                    if len(subparam_dict[col]) != repeats[idx]:
                        raise ValueError(
                            f"All values in each subparameter dictionary {idx} must have the same length. "
                            f"Found mismatched lengths for column '{col}': expected {repeats[idx]}, "
                            f"but got {len(subparam_dict[col])}."
                        )
                    concat_values[col].extend(subparam_dict[col])
        else:
            # We are given the repeat values directly. We do not add any new columns.
            if np.isscalar(repeats):
                repeats = [repeats] * graph_state.num_samples
            concat_values = {}

        # Use the repeats list to expand the graph state. Save the information about the
        # original indices and the sub-indices for each sample before and after expansion.
        org_inds = np.arange(graph_state.num_samples).repeat(repeats)
        sub_inds = np.concatenate([np.arange(r) for r in repeats])
        graph_state.repeat(repeats)
        results = [org_inds, sub_inds]

        # Add columns for any subparameters we need to concatenate.
        for col, values in concat_values.items():
            graph_state.set(self.node_string, col, values)
            results.append(values)

        # Save and return the old incides and the subindices as the result.
        self._save_results(results, graph_state)
        return results
