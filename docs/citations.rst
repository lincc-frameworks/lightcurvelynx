Citations / Acknowledgements
===============================================================================

If you use LightCurveLynx in your research, we ask that you cite the following papers:

  * Dai et. al. 2026 "LightCurveLynx: Forward Modeling of Time-Domain Surveys with Application to ZTF SN Ia DR2" (accepted by ApJ; see `arxiv.org/abs/2604.07134 <https://arxiv.org/abs/2604.07134>`__)
  * Kubica et. al. 2026 "LightCurveLynx: Fast and Nimble Time Domain Simulation for Astronomical Surveys" (`pdf at JOSS <https://joss.theoj.org/papers/01344d3b6112431d1a4e49e41389e587>`__)

LightCurveLynx relies on numerous open source packages to perform the computation. Please make sure to cite the packages that your study uses.


Base Libraries
-------------------------------------------------------------------------------

The following libraries are used by most LightCurveLynx runs:

* `astropy <https://www.astropy.org/>`__
* `nested-pandas <https://nested-pandas.readthedocs.io/en/latest/>`__
* `numpy <https://numpy.org/>`__
* `pandas <https://pandas.pydata.org/>`__
* `pooch <https://pooch.readthedocs.io/en/stable/>`__
* `sncosmo <https://sncosmo.readthedocs.io/en/latest/>`__

Some runs may also use:

* `jax <https://jax.readthedocs.io/en/latest/>`__
* `matplotlib <https://matplotlib.org/>`__
* `scipy <https://www.scipy.org/>`__


Models, Effects, and Parameters
-------------------------------------------------------------------------------

LightCurveLynx builds on a large collection of open source astronomy software for models and
effects, each of which may have its own citation requirements and dependencies. Which models,
effects, and parameters used will vary from simulation to simulation. LightCurveLynx includes
citation information in class and function docstrings to help users identify relevant citations.

As an example, consider the dust extinction effect. LightCurveLynx supports a variety of external dustmaps libraries, but was primarily designed to work with the `dustmaps <https://github.com/gregreen/dustmaps>`__ package (Green 2018). The dust extinction is computed via the ExtinctionEffect which uses the `dust_extinction <https://github.com/karllark/dust_extinction>`__ package (Gordon 2024) or the `extinction <https://github.com/sncosmo/extinction>`__ package (Barbary et. al. 2017). In addition, each individual dustmap and extinction function should be cited from their original science paper. The `dustmaps readthedocs <https://dustmaps.readthedocs.io/en/latest/maps.html>`__ page provides more information on the available dustmaps and their citations.


Citation-Compass
-------------------------------------------------------------------------------

LightCurveLynx also uses the LINCC Frameworks `citation_compass <https://github.com/lincc-frameworks/citation-compass>`__ to track (specifically annotated) citations encountered during your simulations. Since Citation-Compass requires explicit annotations, it will include packages used by LightCurveLynx (such as dustmaps), but not the references on which they depend (such as the dustmaps themselves).

For more information on how to use this package, please refer to the :doc:`citations notebook <notebooks/citations>`.
