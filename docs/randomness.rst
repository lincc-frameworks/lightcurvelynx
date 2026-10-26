Randomness
========================================================================================

LightCurveLynx is designed to run statistically meaningful simulations, which means dealing with randomness. The code provides several different mechanisms for controlling randomness.


Default Behavior
-------------------------------------------------------------------------------

By default, LightCurveLynx uses per-node, per-simulation random number generators (RNGs) that are initialized with a random seed (via the default numpy behavior or `urandom`), such as:

.. code-block:: python
    
    from numpy.random import default_rng
    rng = default_rng()

or:

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

Note that even when passing a (seeded) global random number generator, the results may vary if the runs use different batch sizes. This is due to the fact that each batch samples the parameters serially. If the number of parameters sampled in the batch changes, the sequence of random numbers used will also change, leading to different results.


Per-Node Seeds
-------------------------------------------------------------------------------

In previous versions of LightCurveLynx, users could control randomness by providing per-node seeds. This behavior has been deprecated because it can lead to subtle bugs when combined with batching or parallel execution. Users should either provide a single global RNG to the simulation function or leave blank to create a new random number generator.


Parallelism and Randomness
-------------------------------------------------------------------------------

If provided a global RNG in parallel mode, LightCurveLynx will create a unique RNG for each batch of samples. This is done to avoid correlations between the batches. If you need to control randomness while running in parallel, you should provide a global RNG to the ``simulate_lightcurves()`` function. As noted above, the results may still vary if the batch size changes, because the sequence of random numbers used will differ.
