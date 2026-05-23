=================================================
Fisher information and expected uncertainty
=================================================

Why the spiky FI curves you may have seen are *real*, and how to compute
Fisher information and decoder uncertainty reliably from an
:class:`~braincoder.models.EncodingModel`.

.. contents::
   :local:

Two questions, two APIs
-----------------------

After fitting an encoding model + noise covariance you typically want one
of two things:

**How much information does the population carry about a stimulus value?**
  → :meth:`EncodingModel.get_fisher_information`

**How well will an unbiased decoder recover the true stimulus, averaged
over noise draws?**
  → :meth:`EncodingModel.get_expected_uncertainty`

Both work on the same fitted ``(parameters, omega, dof)`` tuple. The
Fisher information is the asymptotic lower bound on decoder variance
(Cramér–Rao); the expected uncertainty is an empirical readout from the
*actual* posterior implied by your model. They agree when the decoder is
well-fit and the posterior is locally Gaussian; they diverge when the
posterior is multimodal, bounded by the grid edge, or the noise model
mismatches the data.

Closed-form Fisher information
------------------------------

For Gaussian noise the population FI at stimulus :math:`s` is

.. math::

    \mathcal{I}(s) = J(s)^\top \, \Omega^{-1} \, J(s)
    \quad
    \text{where } J(s)_v = \frac{\partial \mu_v}{\partial s}.

For multivariate Student-t noise with :math:`\nu` degrees of freedom on
a :math:`p`-voxel residual,

.. math::

    \mathcal{I}_t(s) = \frac{\nu + p}{\nu + p + 2} \cdot J(s)^\top \, \Omega^{-1} \, J(s).

As :math:`\nu \to \infty` the prefactor approaches 1 and the Gaussian
formula is recovered. Both are now available as closed-form expressions
in :meth:`get_fisher_information` (pass ``analytical=True``, default).
The Monte-Carlo branch (``analytical=False``) is kept for backwards
compatibility and as a sanity check; it adds nothing in this regime
except MC noise and an int32 overflow risk at large
``n × n_voxels × n_stim``.

Why FI curves can look spiky — and when it's real
-------------------------------------------------

Single voxels with Gaussian-like tuning have :math:`\partial \mu / \partial s = 0`
*at* the preferred stimulus and peaks on the flanks. A population FI is a
sum of *squared* per-voxel gradients, so the curve dips at every voxel's
mode and bumps on either side. When the fitted mode population *clusters*
— e.g. an ROI where many voxels prefer the same narrow range of values
— the population FI gets a sharp dip-bump-dip pattern at the cluster,
not because the model is wrong but because that's what the population
*actually* encodes.

Diagnose this in three steps:

1. Compute :meth:`get_fisher_information` with ``analytical=True`` (no
   MC noise). If the curve is still spiky, the spikes are population
   structure.
2. Plot a histogram of the fitted modes on the same x-axis. Spike
   positions in the FI should sit on the *flanks* of clusters in the
   mode histogram.
3. (Optional) call :meth:`get_expected_uncertainty` on the same grid;
   ``var_E`` should also be spiky in the same places, since the
   Cramér–Rao bound is sharp where FI is sharp.

When the spikes are *not* real:

* If they move when you change MC seed or ``n``, they're MC noise — use
  the analytical path.
* If they live only near ``stimulus_range[0]`` or ``stimulus_range[-1]``
  for a log-Gaussian model, they're boundary cliffs from
  :math:`\log x / x` blow-up; restrict the grid or use a model with no
  singularity at the lower boundary.
* If they shift when you fit ``omega`` again (different random seed in
  ResidualFitter), they're noise-model artifact; refit with
  :class:`ResidualFitter` ``lambd`` between 0.5 and 1 to bias toward a
  cleaner empirical covariance.

API reference
-------------

.. autoclass:: braincoder.models.EncodingModel
   :members: get_fisher_information, get_expected_uncertainty
   :noindex:

Worked example
--------------

.. code-block:: python

    import numpy as np
    import pandas as pd
    from braincoder.models import LogGaussianPRF

    rng = np.random.default_rng(0)
    pars = pd.DataFrame({
        "mode":      rng.normal(20, 4, 100).clip(2, 45).astype(np.float32),
        "fwhm":      rng.normal(8, 2, 100).clip(2, 25).astype(np.float32),
        "amplitude": np.ones(100, dtype=np.float32),
        "baseline":  np.zeros(100, dtype=np.float32),
    })
    model = LogGaussianPRF(parameterisation="mode_fwhm_natural", parameters=pars)
    omega = (0.25 ** 2) * np.eye(100, dtype=np.float32)
    stim = np.linspace(1, 40, 100, dtype=np.float32)

    fi_gauss = model.get_fisher_information(stim, omega=omega)
    fi_t     = model.get_fisher_information(stim, omega=omega, dof=5.0)
    out      = model.get_expected_uncertainty(
        stim, omega=omega, dof=5.0, n_simulations=500, batch_stimuli=20)

    # `out` is a DataFrame with mean_E, var_E, mean_error, mean_abs_error
    # indexed by true stimulus. 1/out["var_E"] ≈ fi_t for an unbiased
    # decoder away from the grid boundary.

Empirical audit
---------------

A 100-voxel log-Gaussian population with two mode clusters (centered at
12 and 32 CHF) was used to compare paths:

================================  =========  =========  =========
Metric                            Analytic   MC n=200   MC n=1000
================================  =========  =========  =========
Mean FI (across grid)             9.566      9.571      9.566
Std FI (across grid)              6.901      6.992      6.933
RMS error vs analytical (raw)     —          1.13       0.37
RMS error vs analytical (% of μ)  —          11.8 %     3.9 %
================================  =========  =========  =========

The MC curves *look* spikier than the analytical reference because of
noise; the analytical curve is itself non-monotonic, confirming the
population-structure origin of the bumps. MC also costs an extra factor
of ``n × n_stim`` memory and triggered an int32 overflow inside
``UnsortedSegmentSum`` at ``n × n_voxels × n_stim ≈ 5 × 10⁷`` —
historically forcing users to shrink ``n_voxels`` to keep MC viable.

See :file:`tests/test_fisher_information.py` for the regression suite
(Student-t prefactor, MC ≈ analytic agreement, FI ↔ ``1/var_E``
relationship, batched vs. unbatched ``get_expected_uncertainty``).
