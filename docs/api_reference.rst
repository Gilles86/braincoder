=============
API Reference
=============

This section documents the most frequently used modules.  All classes operate
on numpy arrays or pandas ``DataFrame`` inputs and run on TensorFlow, JAX or
PyTorch through Keras 3. Models hold only their configuration: pass the paradigm,
parameters, weights and noise model to each method.

Encoding models
===============

.. autoclass:: braincoder.models.EncodingModel
   :members: predict, simulate, get_paradigm, get_parameter_labels, get_stimulus_pdf, predict_stimulus_grid
   :show-inheritance:

.. autoclass:: braincoder.models.LinearModelWithBaseline
   :members:
   :show-inheritance:

.. autoclass:: braincoder.models.LinearModelWithBaselineHRF
   :members:
   :show-inheritance:

HRF utilities
=============

.. autoclass:: braincoder.hrf.HRFModel
   :members:
   :show-inheritance:

.. autoclass:: braincoder.hrf.SPMHRFModel
   :members:
   :show-inheritance:

.. autofunction:: braincoder.hrf.spm_hrf

Optimization helpers
====================

.. autoclass:: braincoder.optimize.WeightFitter
   :members:
   :show-inheritance:

.. autoclass:: braincoder.optimize.ParameterFitter
   :members:
   :show-inheritance:

.. autoclass:: braincoder.optimize.ResidualFitter
   :members: fit
   :show-inheritance:

.. _api-noise:

Noise models for decoding
=========================

See :doc:`noise_models` for the workflow.

.. autoclass:: braincoder.noise.LowRankNoise
   :members:
   :show-inheritance:

.. autofunction:: braincoder.noise.fit_lowrank_omega

.. _api-calibration:

Calibration diagnostics
=======================

.. automodule:: braincoder.utils.calibration
   :members:
