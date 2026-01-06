=============
API Reference
=============

This section documents the most frequently used modules.  All classes operate
on numpy arrays or pandas ``DataFrame`` inputs and run TensorFlow under the
hood.

Encoding models
===============

.. autoclass:: braincoder.models.EncodingModel
   :members: predict, simulate, get_paradigm, get_parameter_labels, to_discrete_model
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
