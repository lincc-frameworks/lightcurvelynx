Citations
===============================================================================

LightCurveLynx relies on numerous open source packages to perform the computation. In addition to
LightCurveLynx itself (citation information coming soon), please make sure to cite the packages
that your study uses.

Base Libraries
-------------------------------------------------------------------------------

The following libraries are used by most LightCurveLynx runs:

* `astropy <https://www.astropy.org/>`_
* `nested-pandas <https://nested-pandas.readthedocs.io/en/latest/>`_
* `numpy <https://numpy.org/>`_
* `pandas <https://pandas.pydata.org/>`_
* `pooch <https://pooch.readthedocs.io/en/stable/>`_
* `sncosmo <https://sncosmo.readthedocs.io/en/latest/>`_

Some runs may also use:

* `jax <https://jax.readthedocs.io/en/latest/>`_
* `matplotlib <https://matplotlib.org/>`_
* `scipy <https://www.scipy.org/>`_

Models, Effects, and Parameters
-------------------------------------------------------------------------------

LightCurveLynx builds on a large collection of open source astronomy software for models and
effects, each of which may have its own citation requirements and dependencies. Which models,
effects, and parameters used will vary from simulation to simulation. LightCurveLynx includes
citation information in class and function docstrings to help users identify relevant citations.

As an example, consider the dust extinction effect. LightCurveLynx supports a variety of external dustmaps
libraries, but was primarily designed to work with the `dustmaps <https://github.com/gregreen/dustmaps>`_
package (Green 2018). The dust extinction is computed via the DustExtinctionEffect which used the
`dust_extinction <https://github.com/karllark/dust_extinction>`_ package (Gordon 2024) or the
SncosmoExtinctionEffect which uses the `extinction <https://github.com/sncosmo/extinction>`_ package
(Barbary et. al. 2017). In addition, each individual dustmap and extinction function should be cited
from their original science paper. The `dustmaps readthedocs <https://dustmaps.readthedocs.io/en/latest/maps.html>`_
page provides more information on the available dustmaps and their citations.

Citation-Compass
-------------------------------------------------------------------------------

LightCurveLynx also uses the LINCC Frameworks `citation_compass` package to track (specifically annotated)
citations encountered during your simulations. Since Citation-Compass requires explicit annotations, it
will include packages used by LightCurveLynx (such as dustmaps), but not the references on which they depend
(such as the dustmaps themselves).

For more information on how to use this package, please refer to the :doc:`citations notebook <notebooks/citations>`.
