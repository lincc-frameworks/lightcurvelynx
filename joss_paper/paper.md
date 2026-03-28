---
title: 'LightCurveLynx: Fast and Nimble Time Domain Simulation for Astronomical Surveys'
tags:
  - Python
  - astronomy
  - time domain
  - transients
authors:
  - name: Jeremy Kubica
    orcid: 0009-0009-2281-7031
    affiliation: "1"
  - name: Mi Dai
    orcid: 0000-0002-5995-9692
    affiliation: "2"
  - name: Olivia Lynn
    orcid: 0000-0001-5028-146X
    affiliation: "1"
  - name: Konstantin Malanchev
    orcid: 0000-0001-7179-7406
    affiliation: "1"
  - name: Alex I. Malz
    orcid: 0000-0002-8676-1622
    affiliation: "3"
  - name: Andrew Connolly
    orcid: 0000-0001-5576-8189
    affiliation: "4"
  - name: Melissa DeLucchi
    orcid: 0000-0002-1074-2900
    affiliation: "1"
  - name: Katarzyna Kruszyńska
    orcid: 0000-0002-2729-5369
    affiliation: "5"
  - name: Wanqing Liu
    orcid: 0009-0008-3199-2627
    affiliation: "1"
  - name: Rachel Mandelbaum
    orcid: 0000-0003-2271-1527
    affiliation: "1"
  - name: Nikhil Sarin
    orcid: 0000-0003-2700-1030
    affiliation: "6,7"
  - name: Steve Schulze
    orcid: 0000-0001-6797-1889
    affiliation: "8"
affiliations:
  - name: McWilliams Center for Cosmology and Astrophysics, Department of Physics, Carnegie Mellon University, Pittsburgh, PA 15213, USA
    index: 1
  - name: Pittsburgh Particle Physics, Astrophysics, and Cosmology Center (PITT PACC). Physics and Astronomy Department, University of Pittsburgh, Pittsburgh, PA 15260, USA
    index: 2
  - name: Space Telescope Science Institute, 3700 San Martin Drive, Baltimore, Maryland 21218, USA
    index: 3
  - name: DiRAC Institute and the Department of Astronomy, University of Washington, 3910 15th Ave NE, Seattle, WA 98195, USA
    index: 4
  - name: Las Cumbres Observatory, 6740 Cortona Drive, Suite 102, Goleta, CA 93117, USA
    index: 5
  - name: Kavli Institute for Cosmology, University of Cambridge, Madingley Road, CB3 0HA, UK
    index: 6
  - name: Institute of Astronomy, University of Cambridge, Madingley Road, CB3 0HA, UK
    index: 7
  - name: Department of Particle Physics and Astrophysics, Weizmann Institute of Science, 234 Herzl St, 76100 Rehovot, Israel
    index: 8
date: 16 March 2026
bibliography: paper.bib
---

# Summary

`LightCurveLynx` is a Python-based forward simulation framework for catalog-level time-varying astronomical phenomena that brings together the community’s many astronomical modeling packages into a single framework. Its goal is to allow the users to mix-and-match different models, effects, and surveys within a single simulation environment. It is optimized for scalability, modularity, and extensibility. `LightCurveLynx` integrates numerous popular packages, such as `sncosmo`, `pzflow`, and `redback`, along with different sampling packages and astronomical surveys.


# Statement of need

To fully realize the value of the next generation of large-scale astronomical survey data, it will be necessary to analyze this data relative to expected observations from a range of underlying population models. Such analysis allows scientists to characterize the selection functions inherent in each survey, determine which objects could be observed/classified from the data, and to fit or refine models through simulation-based inference. Further, given these survey’s duration and depth, it is increasingly vital that such simulations are efficient. Making such simulations available early can even help with optimization of the observing strategy for different science cases.

The astronomy community has invested significant resources into developing powerful software packages to simulate and model a wide range of physical phenomena. To provide only a few concrete examples, SNANA [@Kessler2009], SNcosmo [@barbary2025], and SkySurvey [@rigault2026], provide powerful libraries for realistic simulations of supernovae; VBMicrolensing [@Bozza2025], PyLIMA [@Bachelet2017], BAGLE [@lu2025] provide approaches for modeling and simulating microlensing events; and redback [@sarin2024] provides a comprehensive library of models for physical phenomena. Although this proliferation of software allows researchers to explore a large number of phenomena, it introduces a level of fragmentation and complexity. The feature set, such as supported surveys and noise effects, varies from package to package.

`LightCurveLynx` is a package for forward simulation of time-varying astronomical phenomena that brings together this extensive ecosystem of software into a consistent framework. The user can access models from different packages, a range of effects, and a variety of surveys and instrument types. In addition, the modular API makes `LightCurveLynx` highly extensible. Users can easily introduce new variability models or wrap other Python packages, expanding the system’s capabilities to incorporate the community’s latest developments.


# State of the Field

As noted in the previous section, the astronomy community has developed a range of powerful tools for forward simulation of transient and variable sources. Individual packages often focus on specific types of astronomical phenomena such as supernovae or microlensing. Further each package may support a subset of the existing physical effects or survey strategy. `LightCurveLynx` does not aim to duplicate those efforts, but rather to augment them by providing a common framework for users to combine the existing packages.


# Software design

`LightCurveLynx` is designed to enable users to accurately and efficiently run simulations using a range of models and simulation packages. As such, it uses several core principles: (1) provide an extensible and flexible object-oriented API, (2) provide an interface to consistently sample from complex distributions, (3) allow users to save simulation state and replay all/part of the simulation for analysis and debugging, and (4) build in vectorization and parallelization for efficient runs. This modular structure (and the general program flow) are shown in \autoref{fig:flow}.

![The basic simulation flow for `LightCurveLynx`\label{fig:flow}](figure.png)

The simulation starts by sampling the model’s parameters from given statistical distributions. These parameters are combined with the information about where the survey is pointing to generate observed fluxes at either the spectral or band level. (We use “flux” as a synonym of “spectral flux density per unit frequency", while by “bandflux” we mean spectral flux density measured in the given photometric passband). `LightCurveLynx` uses the survey information, along with each object's position, to pre-filter evaluations to just those times when the object will be observed, saving significant computation over evaluating models at all times.  The simulation then applies line of sight effects, such as dust extinction, to the fluxes. If the fluxes were generated at the spectral level, they are integrated with the survey’s filter to produce band fluxes. Finally, instrument and detector noise are sampled and added based on the survey’s characteristics.

Models of physical phenomena are implemented as subclasses in a simple class hierarchy. The `BasePhysicalModel` class defines key astronomical parameters (`RA`, `Dec`, etc.) and an interface for all computations. Two subclasses provide further specializations for the different types of modeling. `SEDModel` serves as the parent class for any model that simulates output at the spectral level. `BandfluxModel` serves as the parent class for any model that simulates output at the bandflux level only. These classes handle common computations, such as correcting for redshift and extrapolating predictions. Creators of a new model can create a new subclass of one of these two types and focus on the core logic for computing fluxes.

Another advantage of this class hierarchy is that new features can be added in the parent classes and automatically applied to all models. One example of this is `LightCurveLynx`’s extrapolation functionality. Some common models, such as splines fit from data, only produce valid predictions over a finite range of times and wavelengths. By supporting bound checking and extrapolation in the parent classes, `LightCurveLynx` adds the ability to use an extrapolation function with any model.

The parameter distributions are specified using Pythonic syntax as a directed acyclic graph of parameter relations that can draw distributions from packages such as numpy [@harris2020array], scipy [@SciPy2020], or pzflow [@crenshaw2025]. All parameter (and bookkeeping) information is handled through the `ParameterizedNode` base class and saved as a `GraphState` object, which allows the user to analyze or replay the simulations. Sampling is vectorized wherever possible and can be parallelized for efficiency.

Line of sight effects are wrapped subclasses of the `EffectModel` class and can be added to any `BasePhysicalModel` object. This separates the implementation of the model from the effects and allows a single effect type (e.g. dust extinction) to be consistently applied to any physical model. This approach ensures consistency and reduces code duplication.

Survey specific information is encapsulated in subclasses of the `ObsTable`, allowing users to simulate their models under different survey conditions. `LightCurveLynx` currently supports data from the Vera C. Rubin’s LSST [@Ivezic2019] in simulated and DP1/DP2 formats, data from the Zwicky Transient Facility surveys [@Bellm2019], simulations of the Nancy Grace Roman telescope [@Spergel2015], and simulations of the Argus Array [@Law2022].

Parallelization is natively supported using Python's `ProcessPoolExecutor` as well as multi-machine parallelization such as [Dask](https://www.dask.org/) or [Ray](https://docs.ray.io/en/latest/index.html).


# Research impact statement

The software was verified by simulating populations of Type Ia supernovae under the ZTF survey and comparing the simulated population statistics to actual data [@dai2026]. The software is currently being used for multiple population studies under the proposed Rubin LSST cadence, including simulating RR Lyrae light curves and microlensing events.


# Installation and usage

`LightCurveLynx` can be installed with `pip` or through `conda-forge`. See the instructions on our [readthedocs page](https://lightcurvelynx.readthedocs.io/en/latest/). The project's [tutorial notebooks documentation page](https://lightcurvelynx.readthedocs.io/en/latest/notebooks.html) provides a variety of usage examples and technical deep dives.


# AI usage disclosure

All top level design, including control flow and class structure, was designed by humans. Github copilot (using Claude Sonnet 4) was used during the development of the software for online suggestions and small changes. All code changes, regardless of author, were reviewed by humans. No generative AI tools were used in the writing of this manuscript.


# Acknowledgements

This project is supported by Schmidt Sciences. The authors acknowledge support from the DiRAC Institute in the Department of Astronomy at the University of Washington. The DiRAC Institute is supported through generous gifts from the Charles and Lisa Simonyi Fund for Arts and Sciences, and the Washington Research Foundation.


# References
