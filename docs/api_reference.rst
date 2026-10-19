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

.. autoclass:: braincoder.models.GaussianConnectiveField
   :members: predict, get_cf_weights, source_properties
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

.. autoclass:: braincoder.optimize.ConnectiveFieldFitter
   :members: fit_grid, refine, fit_cv, get_rsq
   :show-inheritance:

.. autofunction:: braincoder.optimize.exclusion_from_distances

Cortical surfaces
=================

.. autofunction:: braincoder.utils.cortex.geodesic_distance_matrix

.. autoclass:: braincoder.utils.cortex.HeatMethodSolver
   :members: distances

.. autofunction:: braincoder.utils.cortex.submesh

.. autofunction:: braincoder.utils.cortex.block_distance_matrix
