======================
Braincoder documentation
======================

``braincoder`` provides tools to build encoding models, convolve them with
hemodynamic response functions, and decode stimuli from neural data.

Quick start
===========

The example below mirrors the steps you will take in most analyses: prepare a
paradigm, instantiate a model, simulate data, and fit parameters.

.. code-block:: python

    import numpy as np
    import pandas as pd
    from braincoder.models import LinearModelWithBaseline
    from braincoder.optimize import ParameterFitter

    n_tp = 200
    paradigm = pd.DataFrame(
        {"stimulus": np.sin(np.linspace(0, 4 * np.pi, n_tp))},
        index=pd.Index(np.arange(n_tp), name="time"),
    )

    true_params = pd.DataFrame(
        {"baseline": [0.0, 0.5]},
        index=pd.Index(["voxel_1", "voxel_2"], name="voxel"),
    )

    model = LinearModelWithBaseline(paradigm=paradigm, parameters=true_params)
    simulated = model.simulate(
        paradigm=paradigm,
        parameters=true_params,
        noise=0.1,
    )

    fitter = ParameterFitter(model=model, data=simulated, paradigm=paradigm)
    estimates = fitter.fit(max_n_iterations=200, learning_rate=0.05)

    print(estimates)

More resources
==============

- :doc:`concepts` — architectural overview and workflow tips.
- :doc:`api_reference` — autodoc for the most used modules.
- :doc:`tutorial/index` — hands-on lessons that walk through realistic
  encoding/decode pipelines.
- :doc:`auto_examples/index` — gallery of runnable scripts.

.. toctree::
   :maxdepth: 1

   concepts
   api_reference
   tutorial/index
   auto_examples/index
   bibliography
