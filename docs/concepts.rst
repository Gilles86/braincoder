=======================
Concepts & Architecture
=======================

``braincoder`` wraps common steps in the encoding/decoding workflow for fMRI
analysis.  The package is split into small, composable modules so you can swap
out only the parts you need or script full experimental pipelines.

Core pipeline
=============

1. **Describe the paradigm.** Provide stimulus traces for each time point as a
   pandas ``DataFrame``.  Stimulus helpers in :mod:`braincoder.stimuli` clean
   the paradigm and generate design matrices tailored to the selected model.
2. **Choose an encoding model.** Classes in :mod:`braincoder.models` turn the
   paradigm into voxel-wise predictions.  ``EncodingModel`` handles formatting
   plus utilities such as simulation, gradients, and conversion to discrete
   grids.
3. **Model the hemodynamic response.** Attach an :class:`HRFModel
   <braincoder.hrf.HRFModel>` or use :class:`SPMHRFModel
   <braincoder.hrf.SPMHRFModel>` to convolve predicted activity with a
   canonical or voxel-specific HRF.
4. **Fit parameters or weights.** :class:`braincoder.optimize.WeightFitter`
   solves linear weights via least squares, whereas
   :class:`braincoder.optimize.ParameterFitter` runs iterative TensorFlow
   optimizers with support for constraints, shared parameters, and progress
   logging.
5. **Decode or visualize.** Use the fitted encoding model to simulate
   responses, compute gradients, or invert the model (see the tutorials and
   example notebooks in ``docs/tutorial`` and ``notebooks``).

Stimulus paradigms
==================

Stimulus classes (see :mod:`braincoder.stimuli`) encapsulate domain knowledge
about the feature space.  When you pass a ``paradigm`` to an encoding model it
automatically routes through the matching ``Stimulus`` subclass to:

- ensure required columns/dimensions are present,
- optionally clean and normalize user input, and
- generate tensors that the TensorFlow models expect.

Encoding models
===============

All encoding models derive from :class:`braincoder.models.EncodingModel`,
which provides:

- ``predict`` and ``simulate`` helpers that work with numpy/pandas inputs,
- automatic broadcasting to batches/voxels, and
- convenience methods to fetch paradigms, parameters, and gradients.

Common subclasses include ``LinearModel``,
``LinearModelWithBaseline``, and HRF-aware versions such as
``LinearModelWithBaselineHRF``.  Because they share the base API you can swap
them without rewriting fitting code.

Hemodynamic response functions
==============================

``braincoder.hrf`` implements standalone HRF utilities and the
:class:`SPMHRFModel <braincoder.hrf.SPMHRFModel>`.  You can:

- precompute stimulus-specific HRFs with :func:`braincoder.hrf.spm_hrf`,
- run shared or voxel-wise convolutions via :class:`braincoder.hrf.HRFModel`,
  and
- plug custom HRFs into any :class:`braincoder.models.HRFEncodingModel`
  subclass.

Optimization, fitting, and decoding
===================================

Use :class:`braincoder.optimize.WeightFitter` for fast linear weight solves or
:class:`braincoder.optimize.ParameterFitter` when you need to estimate model
parameters themselves.  Both accept pandas inputs, work with TensorFlow
accelerators, and expose diagnostics such as R² traces and intermediate
parameter snapshots.

Where to go next
================

- Walk through the hands-on :doc:`tutorial/index`.
- Browse the gallery in :doc:`auto_examples/index`.
- Jump to the :doc:`api_reference` to see the complete API surface.
