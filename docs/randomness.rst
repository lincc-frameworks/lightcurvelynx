Randomness
========================================================================================

LightCurveLynx is designed to run statistically meaningful simulations, which means dealing with randomness. The code provides several different mechanisms for controlling randomness.


Default Behavior
-------------------------------------------------------------------------------

By default, LightCurveLynx uses per-node random number generators (RNGs) that are initialized with a random seed (via `urandom` or the default numpy behavior):

.. code-block:: python
    
    from os import urandom
    from numpy.random import default_rng

    seed = int.from_bytes(urandom(4), "big")
    rng = default_rng(seed)

This means that if you run a simulation multiple times, you will get independent simulations and can use the results for statistical analysis.


Global Random Number Generator
-------------------------------------------------------------------------------

If users want to have reproducible simulations, they can provide a (seeded) global random number generator (RNG) to the ``simulate_lightcurves()`` function. This RNG will be used to initialize the per-node RNGs, noise RNGs, etc. The use of a global RNG will override any per-node seeds provided. If no global RNG is provided, the default behavior described above will be used.

**Passing a global RNG is the recommended way to control randomness for testing or reproducibility.**


Per-Node Seeds
-------------------------------------------------------------------------------

Many of the `ParameterizedNode` classes that use randomness have a ``seed`` parameter. If a seed is provided, the node will use it to initialize its **default** random number generator. A user provided (global) random number generator will override the per-node seeds.

These per-node seeds are provided for two reasons (both related to debugging only):
1. They allow deterministic node specifications for unit testing.
2. They allow the users to shut off the randomness to just a specific node (when no global RNG is provided).

As noted below, the per-node seeds are **always overridden** when running in parallel model to prevent correlations between the batches.


Parallelism and Randomness
-------------------------------------------------------------------------------

If provided a global RNG in parallel mode, LightCurveLynx will create a unique RNG for each batch of samples. This is done to avoid correlations between the batches. **Note that this will always override the per-node seeds.** If you need to control randomness while running in parallel, you should provide a global RNG to the ``simulate_lightcurves()`` function.
