==========================
Braincoder documentation
==========================

``braincoder`` provides tools to build encoding models, convolve them with
hemodynamic response functions, and decode stimuli from neural data.

Quick start
===========

The example below mirrors the steps you will take in most analyses: prepare a
paradigm, instantiate a model, simulate data, and fit parameters.

.. code-block:: python

    import numpy as np
    import pandas as pd
    from braincoder.models import GaussianPRF
    from braincoder.optimize import ParameterFitter

    n_tp = 200
    paradigm = pd.DataFrame(
        {"x": np.linspace(-5, 5, n_tp)},
        index=pd.Index(np.arange(n_tp), name="time"),
    )

    true_params = pd.DataFrame(
        {"mu": [-1.0, 2.0], "sd": [1.0, 2.0], "amplitude": [1.0, 2.0], "baseline": [0.0, 0.5]},
        index=pd.Index(["voxel_1", "voxel_2"], name="voxel"),
    )

    model = GaussianPRF()
    simulated = model.simulate(paradigm, true_params, noise=0.1)

    fitter = ParameterFitter(model=model, data=simulated, paradigm=paradigm)
    grid = fitter.fit_grid(mu=np.linspace(-4, 4, 17), sd=np.linspace(0.5, 3, 6),
                           amplitude=[1.0], baseline=[0.0], use_correlation_cost=True)
    init = fitter.refine_baseline_and_amplitude(grid)
    estimates = fitter.fit(init_pars=init, max_n_iterations=500, progressbar=False)

    print(estimates)

Models hold only their configuration. The paradigm, parameters, weights and noise
model are passed to each method.

More resources
==============

- :doc:`whats_new` — release notes; **0.6.0** adds calibrated decoding
  (:doc:`noise_models`).
- :doc:`concepts` — architectural overview and workflow tips.
- :doc:`noise_models` — calibrated decoding: low-rank Ω with
  cross-validated rank and temperature.
- :doc:`api_reference` — autodoc for the most used modules.
- :doc:`tutorial/index` — hands-on lessons that walk through realistic
  encoding/decode pipelines.
- :doc:`auto_examples/index` — gallery of runnable scripts.

Using an AI coding assistant?
=============================

The `braincoder skill for Claude Code <https://github.com/Gilles86/gilles-claude-skills/tree/main/skills/braincoder>`_ gives an assistant a step-by-step
manual for new projects, the common silent errors, and measured runtimes. It is also
readable on its own. Install it with:

.. code-block:: bash

    git clone https://github.com/Gilles86/gilles-claude-skills.git
    ln -s "$PWD/gilles-claude-skills/skills/braincoder" ~/.claude/skills/braincoder

Citing braincoder
=================

Please cite the Zenodo record: de Hollander, G., Renkert, M., Ruff, C. C., & Knapen,
T. H. *braincoder: A package for fitting encoding models to neural data and decoding
stimulus features*. Zenodo. `doi:10.5281/zenodo.10778413
<https://doi.org/10.5281/zenodo.10778413>`_. braincoder is released under the MIT
license, and the code is at https://github.com/Gilles86/braincoder.

Related software
================

`prfmodel <https://popylar-org.github.io/prfmodel/>`_ is a Python package for population
receptive field modelling, also built on Keras 3. It covers further pRF models, such as
CSS, delayed normalization and connective fields.

.. toctree::
   :maxdepth: 1

   whats_new
   migration
   installation
   concepts
   fisher_information
   noise_models
   api_reference
   tutorial/index
   auto_examples/index
   bibliography
